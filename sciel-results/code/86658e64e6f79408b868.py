import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    # The evaluator kills at 150s. We keep a strict margin to ensure return.
    # 10s safety margin is standard, but let's use 5s to be safe for Python overhead.
    deadline = start_time + 145.0
    
    total_words = q ** n
    
    # Precompute digits for all words
    # For q=3, n=8, total_words=6561. This is fast.
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    # Precompute balls (neighborhoods) as bitmasks
    # A ball of radius r around i contains all words within Hamming distance r.
    # For q=3, n=8, radius=1, ball size is 1 + 8*(3-1) = 17.
    # We represent the ball as a Python integer (bitmask) where bit j is 1 if word j is in the ball.
    
    balls = [0] * total_words
    
    # Precompute powers of q for fast index calculation
    powers = [1] * n
    for i in range(n-2, -1, -1):
        powers[i] = powers[i+1] * q

    for i in range(total_words):
        d = digits_of[i]
        mask = 1 << i  # self
        
        if radius == 1:
            for pos in range(n):
                orig_digit = d[pos]
                for new_digit in range(q):
                    if new_digit == orig_digit:
                        continue
                    delta = (new_digit - orig_digit) * powers[pos]
                    idx = i + delta
                    mask |= (1 << idx)
        else:
            # General radius (slower, but n=8 is small)
            # For efficiency in this specific task (radius=1 is expected), 
            # we can stick to the fast path if radius==1.
            # If radius > 1, we fall back to a slower generation or just use radius=1 logic 
            # if the task implies standard football pool (radius 1).
            # The task says "at most one position", so radius=1 is the primary case.
            # However, the interface accepts 'radius'.
            for r in range(1, radius + 1):
                for pos_comb in itertools.combinations(range(n), r):
                    for variation in itertools.product(range(q), repeat=r):
                        delta = 0
                        is_diff = False
                        for k, pos in enumerate(pos_comb):
                            new_val = variation[k]
                            if new_val != d[pos]:
                                is_diff = True
                                delta += (new_val - d[pos]) * powers[pos]
                        if not is_diff:
                            continue
                        idx = i + delta
                        mask |= (1 << idx)
                        
        balls[i] = mask

    all_mask = (1 << total_words) - 1
    
    # Helper to check if a set of indices covers the space
    def is_covered(indices):
        mask = 0
        for i in indices:
            mask |= balls[i]
        return mask == all_mask

    # Helper to get the cover mask
    def get_cover_mask(indices):
        mask = 0
        for i in indices:
            mask |= balls[i]
        return mask

    # Initial Construction: Greedy Set Cover
    # This gives a good starting point for SA, much better than random.
    def greedy_construct():
        uncovered = all_mask
        code = []
        # To speed up, we can just iterate. 
        # 6561 items, bit_count is fast.
        while uncovered:
            if time.time() > deadline - 10.0:
                return code # Return partial if timeout, but greedy is fast
        
            best_idx = -1
            max_cover = 0
            candidates = []
            
            # Optimization: Only consider indices that add new coverage?
            # Actually, just scan all. 6561 is small.
            for i in range(total_words):
                gain = (balls[i] & uncovered).bit_count()
                if gain > max_cover:
                    max_cover = gain
                    best_idx = i
                    candidates = [i]
                elif gain == max_cover and gain > 0:
                    candidates.append(i)
            
            if max_cover == 0:
                break # Should not happen if balls cover self
                
            # Random tie-breaking to allow different starts if needed, 
            # but for SA init, deterministic is fine or random is fine.
            best_idx = random.choice(candidates)
            
            code.append(best_idx)
            uncovered &= ~balls[best_idx]
            
        return code

    # Simulated Annealing
    # State: set of codewords (list of indices)
    # Cost: number of codewords (we want to minimize)
    # But we must maintain validity. 
    # Strategy: 
    #   1. Start with a valid cover (from Greedy).
    #   2. Moves:
    #      a. Remove a random codeword. If valid, accept. If invalid, accept with prob exp(-delta/T). 
    #         Here delta is the "penalty" for being invalid? 
    #         Actually, standard SA for set cover often uses a penalty function:
    #         Objective = |Code| + Lambda * (Number of Uncovered Words).
    #         We start with Lambda high enough to enforce validity, then anneal.
    #         Or simpler: Only accept moves that keep the solution valid? 
    #         If we only accept valid moves, it's just local search. 
    #         To escape local minima, we need to allow invalid intermediate states.
    #         
    #   Let's use the Penalty Method:
    #   Score = len(code) + penalty_factor * uncovered_count
    #   We anneal penalty_factor down? Or keep it constant high?
    #   If penalty is high, we stay valid. If we allow invalid states, we might find a 
    #   configuration where removing one word leaves 1 uncovered, but swapping it with 
    #   another fixes it and reduces size? 
    #   
    #   Alternative: "Remove-Swap" Local Search is often sufficient for these sizes. 
    #   The current program does this. 
    #   
    #   Let's try a different angle: **Genetic Algorithm** or **Memetic Algorithm**?
    #   Or just a more aggressive **Randomized Local Search** with restarts?
    #
    #   Let's stick to Simulated Annealing with a validity constraint that is relaxed.
    #   
    #   Objective Function:
    #   F(code) = len(code) + M * (total_words - covered_count)
    #   If M is large, F is dominated by uncovered count. 
    #   We want to minimize F.
    #   
    #   Moves:
    #   1. Remove a random element.
    #   2. Add a random element.
    #   3. Swap a random element with a random other element.
    
    def simulated_annealing(initial_code, time_limit):
        code = initial_code[:]
        current_cover = get_cover_mask(code)
        current_uncovered = total_words - current_cover.bit_count()
        current_cost = len(code) + 1000 * current_uncovered
        
        best_code = code[:]
        best_cost = current_cost
        
        T = 10.0
        cooling = 0.999
        
        # Precompute a list of all indices for random selection
        all_indices = list(range(total_words))
        
        while time.time() < time_limit:
            if T < 0.1:
                break
            
            # Generate a neighbor
            move_type = random.random()
            new_code = code[:]
            
            if move_type < 0.4:
                # Remove
                if len(new_code) == 0:
                    continue
                idx_to_remove = random.randint(0, len(new_code) - 1)
                removed_word = new_code.pop(idx_to_remove)
                new_cover = current_cover & ~balls[removed_word]
            elif move_type < 0.8:
                # Add
                # Pick a random word not in code? Or just any word.
                # Adding a redundant word increases cost, so it's rarely accepted unless it helps later.
                # Better: Add a word that covers the most uncovered bits.
                # But that's greedy. For SA, random is better for exploration.
                # Let's pick a random word from the space.
                new_word = random.choice(all_indices)
                if new_word not in new_code:
                    new_code.append(new_word)
                    new_cover = current_cover | balls[new_word]
                else:
                    continue # No change
            else:
                # Swap
                if len(new_code) == 0:
                    continue
                idx_to_swap = random.randint(0, len(new_code) - 1)
                old_word = new_code[idx_to_swap]
                new_word = random.choice(all_indices)
                if new_word == old_word:
                    continue
                if new_word in new_code:
                    continue
                new_code[idx_to_swap] = new_word
                new_cover = (current_cover & ~balls[old_word]) | balls[new_word]
            
            new_uncovered = total_words - new_cover.bit_count()
            new_cost = len(new_code) + 1000 * new_uncovered
            
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Accept
                code = new_code
                current_cover = new_cover
                current_uncovered = new_uncovered
                current_cost = new_cost
                
                if new_uncovered == 0 and len(code) < len(best_code):
                    best_code = code[:]
                    best_cost = current_cost
            
            T *= cooling
            
        return best_code

    # Run Greedy to get initial solution
    initial_code = greedy_construct()
    
    # Run SA with the remaining time
    # Reserve 5 seconds for final processing
    sa_deadline = time.time() + 130.0
    
    best_code = simulated_annealing(initial_code, sa_deadline)
    
    # Final cleanup: Remove any redundant words from the best code found
    # This ensures we return the minimal valid subset of the best code
    if best_code:
        # Sort to make removal stable? Not necessary.
        # Iterate and remove if redundant
        changed = True
        while changed:
            changed = False
            cover = get_cover_mask(best_code)
            for i in range(len(best_code)):
                c = best_code[i]
                # If removing c keeps cover valid
                if (cover & ~balls[c]) == all_mask:
                    best_code.pop(i)
                    changed = True
                    break # Restart loop
    
    # Fallback if something went wrong
    if not best_code:
        # Use the greedy result if SA failed or returned empty
        best_code = initial_code
        # Clean up greedy result too
        changed = True
        while changed:
            changed = False
            cover = get_cover_mask(best_code)
            for i in range(len(best_code)):
                c = best_code[i]
                if (cover & ~balls[c]) == all_mask:
                    best_code.pop(i)
                    changed = True
                    break

    # Convert indices back to words
    return [list(digits_of[i]) for i in best_code]
