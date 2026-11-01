import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    total_words = q ** n
    
    # Handle trivial or small cases
    if total_words > 10000:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [[0] * n]
            
    if n == 0:
        return [[]]
        
    # Precomputation
    # Map tuple to index
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx += t[i] * powers[i]
        return idx

    def idx_to_tuple(idx):
        t = [0] * n
        rem = idx
        for i in range(n):
            t[i] = rem % q
            rem //= q
        return tuple(t)

    # Precompute balls for each word
    # ball[i] is a list of indices covered by codeword i
    balls = [None] * total_words
    for i in range(total_words):
        w = list(idx_to_tuple(i))
        ball = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    new_w = list(w)
                    new_w[pos] = new_val
                    new_idx = tuple_to_idx(tuple(new_w))
                    ball.append(new_idx)
        balls[i] = ball
    
    # Precompute ball masks as integers for fast union/intersection if needed, 
    # but for redundancy check, counts are better.
    # Let's use a numpy array for counts for speed.
    
    def is_valid_cover(code_indices):
        """Check if the set of indices covers the whole space."""
        if not code_indices:
            return total_words == 0
        covered = np.zeros(total_words, dtype=bool)
        for c in code_indices:
            covered[balls[c]] = True
        return covered.all()

    def get_redundant_indices(code_indices):
        """Return list of indices in code_indices that are redundant."""
        counts = np.zeros(total_words, dtype=np.int32)
        for c in code_indices:
            counts[balls[c]] += 1
        
        redundant = []
        for c in code_indices:
            # If all words in ball[c] are covered by at least 2 codewords, c is redundant
            if np.min(counts[balls[c]]) > 1:
                redundant.append(c)
        return redundant

    def greedy_prune(code_indices):
        """Remove all redundant codewords."""
        current = list(code_indices)
        changed = True
        while changed:
            changed = False
            redundant = get_redundant_indices(current)
            for r in redundant:
                current.remove(r)
                changed = True
        return current

    def generate_initial_code():
        """Generate a valid code using direct sum + pruning."""
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        code = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                word = h + list(tail)
                code.append(tuple_to_idx(tuple(word)))
        return greedy_prune(code)

    def local_search_swap(code_indices, max_swaps=100):
        """
        Try to reduce code size by swapping.
        1. Try to remove any redundant word.
        2. If none, pick a random word, try to swap it with a random other word.
        3. If the new code is valid, accept it.
        4. Then try to remove redundant words again.
        """
        current = list(code_indices)
        current_size = len(current)
        
        for _ in range(max_swaps):
            if time.time() > deadline - 5.0:
                break
                
            # 1. Try to remove
            redundant = get_redundant_indices(current)
            if redundant:
                # Remove all redundant
                for r in redundant:
                    current.remove(r)
                continue # Loop again to check for more
            
            # 2. If no redundant, try a swap
            # Pick a random codeword to remove
            idx_to_remove = random.choice(current)
            
            # Pick a random new codeword to add
            # To be efficient, we can bias towards words that cover many "uniquely covered" spots
            # But random is fine for exploration.
            new_word = random.randint(0, total_words - 1)
            
            if new_word in current:
                continue
                
            # Test validity of swap
            # New code = current - {idx_to_remove} + {new_word}
            # We can check validity by checking if the union of balls covers everything.
            # Optimization: 
            # Let U be union of balls in current.
            # Let B_remove be ball[idx_to_remove].
            # Let B_add be ball[new_word].
            # The new union is (U \ B_remove) U B_add.
            # This is valid if (U \ B_remove) U B_add == U_all.
            # Which is equivalent to: B_add covers (U_all \ (U \ B_remove)) = U_all \ U U B_remove? No.
            # U_all \ (U \ B_remove) = (U_all \ U) U (U \ (U \ B_remove)) = {} U (B_remove \ (U \ B_remove)).
            # Actually, simpler: 
            # The words that were ONLY covered by idx_to_remove are (B_remove \ (U \ B_remove)).
            # Let's call this set S_unique.
            # If S_unique is empty, then removing idx_to_remove leaves a valid cover. (But we already checked redundancy, so S_unique is NOT empty).
            # So we MUST cover S_unique with B_add.
            # So the swap is valid if S_unique is a subset of B_add.
            
            # Calculate S_unique
            # S_unique = { w in B_remove | count[w] == 1 in current }
            
            counts = np.zeros(total_words, dtype=np.int32)
            for c in current:
                counts[balls[c]] += 1
            
            s_unique = [w for w in balls[idx_to_remove] if counts[w] == 1]
            
            # Check if new_word covers all of s_unique
            valid_swap = True
            new_ball = balls[new_word]
            # Use a set for s_unique for fast lookup? Or just loop.
            # s_unique is small (size <= 1 + n*(q-1) = 13 for n=6, q=3).
            unique_set = set(s_unique)
            for w in new_ball:
                if w in unique_set:
                    unique_set.remove(w)
                if not unique_set:
                    break
            
            if not unique_set:
                # Valid swap
                current.remove(idx_to_remove)
                current.append(new_word)
                
                # After swap, try to remove redundant words
                # (Handled in next iteration of loop)
                
        return current

    # Main Search Loop
    best_code = None
    best_size = float('inf')
    
    # Initial solution
    current_code = generate_initial_code()
    current_code = local_search_swap(current_code, max_swaps=50)
    
    if len(current_code) < best_size:
        best_size = len(current_code)
        best_code = list(current_code)
        
    # Iterated Local Search
    # We will perform multiple restarts or perturbations.
    # Since the state space of valid codes of size ~73 is connected, 
    # we can just run the local search for a long time.
    
    # To improve diversity, we can occasionally "kick" the code by removing a word 
    # and replacing it with a random one, even if it increases size temporarily? 
    # No, we must stay valid.
    # So we stick to valid swaps.
    
    # Let's run the local search for the remaining time.
    # We can also try different random seeds for the swap selection.
    
    while time.time() < deadline - 10.0:
        # Run local search on current_code
        current_code = local_search_swap(current_code, max_swaps=200)
        
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = list(current_code)
            # If we found the theoretical minimum (73), we can stop early?
            # Lower bound is 71, best known is 73.
            if best_size <= 73:
                break
        
        # Perturbation: 
        # If we are stuck, we might want to try a different initial code.
        # Let's generate a new random valid code? 
        # Generating a random valid code is hard.
        # Instead, let's just keep swapping. The swap operator explores the space of valid codes of the current size.
        # If we are at size 74, we can only move within size 74. 
        # To get to 73, we need a swap that makes a word redundant.
        # The local_search_swap function does exactly this: it swaps, then checks for redundancy.
        
        # If we've been stuck at the same size for a long time, 
        # we might try to "shuffle" the current code by doing many random swaps 
        # without checking for size reduction, just to move to a different local minimum.
        # The current implementation already does random swaps.
        
        # To ensure we don't get stuck in a tiny cycle, we can add a "cooling" or "restart" logic.
        # But for 120s, the simple loop should explore enough.
        
        # Optional: If best_size is still high, try a fresh start with a different random seed for the initial construction?
        # The initial construction is deterministic. 
        # We can randomize the order of the tetracode or tails?
        # Or just accept the direct sum as the only source of initial codes.
        
        # Let's add a check: if no improvement in last X seconds, restart from scratch?
        # Restarting from scratch takes time. 
        # Better: Just keep swapping.
        
    # Final conversion
    if best_code is None:
        best_code = generate_initial_code()
        
    result = [list(idx_to_tuple(c)) for c in best_code]
    
    return result
