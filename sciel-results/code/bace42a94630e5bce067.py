import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    total_words = q ** n
    
    # Precompute digits for all words
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    # Precompute balls (neighbors) as bitmasks
    balls = [0] * total_words
    
    # Precompute powers of q for fast index calculation
    powers = [1] * (n + 1)
    for i in range(1, n + 1):
        powers[i] = powers[i-1] * q

    for i in range(total_words):
        d = digits_of[i]
        mask = 1 << i
        
        if radius == 1:
            for pos in range(n):
                orig_digit = d[pos]
                for new_digit in range(q):
                    if new_digit == orig_digit:
                        continue
                    delta = (new_digit - orig_digit) * powers[n - 1 - pos]
                    idx = i + delta
                    mask |= (1 << idx)
        else:
            # General radius
            for r in range(1, radius + 1):
                for pos_comb in itertools.combinations(range(n), r):
                    orig_digits = [d[p] for p in pos_comb]
                    for variation in itertools.product(range(q), repeat=r):
                        is_diff = False
                        delta = 0
                        for k, pos in enumerate(pos_comb):
                            new_val = variation[k]
                            if new_val != d[pos]:
                                is_diff = True
                                delta += (new_val - d[pos]) * powers[n - 1 - pos]
                        if not is_diff:
                            continue
                        idx = i + delta
                        mask |= (1 << idx)
                        
        balls[i] = mask

    all_mask = (1 << total_words) - 1
    
    # Helper to get cover mask of a set
    def get_cover_mask(code_indices):
        mask = 0
        for c in code_indices:
            mask |= balls[c]
        return mask

    # Initial Solution: Greedy Construction
    def greedy_construct(seed):
        random.seed(seed)
        uncovered = all_mask
        code = []
        
        while uncovered:
            if time.time() > deadline - 10.0:
                return code # Return partial if timeout, though unlikely to be invalid if we just return list
                # Actually, if we return partial, it might not cover. 
                # But greedy is fast. 6561 items, ~400 steps. 6561*400 = 2.6M ops. Fast.
                
            best_idx = -1
            max_cover = 0
            candidates = []
            
            # Optimization: We can maintain a list of active candidates?
            # For now, iterate all.
            for i in range(total_words):
                gain = (balls[i] & uncovered).bit_count()
                if gain > max_cover:
                    max_cover = gain
                    best_idx = i
                    candidates = [i]
                elif gain == max_cover and gain > 0:
                    candidates.append(i)
                    
            if max_cover == 0:
                break
                
            if len(candidates) > 1:
                best_idx = random.choice(candidates)
            else:
                best_idx = candidates[0]
                
            code.append(best_idx)
            uncovered &= ~balls[best_idx]
            
        return code

    # Simulated Annealing
    def simulated_annealing(initial_code, max_time=120.0):
        if not initial_code:
            return initial_code
            
        current_code = initial_code[:]
        current_cover = get_cover_mask(current_code)
        current_size = len(current_code)
        
        # Track best found
        best_code = current_code[:]
        best_size = current_size
        
        # Temperature parameters
        T = 1.0
        cooling_rate = 0.9995
        min_T = 0.01
        
        # Precompute set for fast membership check
        current_set = set(current_code)
        
        steps = 0
        start_sa = time.time()
        
        while time.time() < deadline - 5.0 and T > min_T:
            steps += 1
            
            # Choose a move: Remove, Add, or Swap
            # Probabilities: 
            # If we have redundant words, Remove is good.
            # If we are large, Remove is good.
            # Add is risky but might allow better structure? No, we want to minimize size.
            # Swap is the main driver for structure change.
            
            # Let's bias towards Swap and Remove.
            # If current_size > target, Remove is more likely.
            # For now: 40% Swap, 30% Remove, 30% Add (Add is rare, usually bad, but helps escape?)
            # Actually, Add increases size, which is bad for objective. 
            # But in SA, we accept worse solutions. 
            # However, we only care about valid solutions. 
            # If we Add, we become larger. If we then Remove a different one, we might get smaller.
            
            move_type = random.random()
            
            new_code = current_code[:]
            new_set = current_set.copy()
            new_cover = current_cover
            new_size = current_size
            
            if move_type < 0.5:
                # Swap: Pick one to remove, one to add
                if not current_code:
                    continue
                idx_remove = random.randint(0, len(current_code) - 1)
                c_remove = current_code[idx_remove]
                
                # Pick c_add
                # Sample from all words? Or from uncovered? 
                # If we sample from all, we might pick one that is already covered.
                # Better: Sample from words that are "critical" or random.
                # Let's just sample random.
                c_add = random.randint(0, total_words - 1)
                
                if c_add in new_set:
                    continue
                
                # Apply swap
                new_code[idx_remove] = c_add
                new_set.remove(c_remove)
                new_set.add(c_add)
                
                # Update cover
                # new_cover = (current_cover & ~balls[c_remove]) | balls[c_add]
                # This is efficient.
                new_cover = (current_cover & ~balls[c_remove]) | balls[c_add]
                # Size unchanged
                new_size = current_size
                
            elif move_type < 0.8:
                # Remove
                if not current_code:
                    continue
                idx_remove = random.randint(0, len(current_code) - 1)
                c_remove = current_code[idx_remove]
                
                new_code.pop(idx_remove)
                new_set.remove(c_remove)
                new_cover = current_cover & ~balls[c_remove]
                new_size = current_size - 1
                
            else:
                # Add
                c_add = random.randint(0, total_words - 1)
                if c_add in new_set:
                    continue
                
                new_code.append(c_add)
                new_set.add(c_add)
                new_cover = current_cover | balls[c_add]
                new_size = current_size + 1
            
            # Evaluate Objective
            # We want to minimize size, but MUST be valid.
            # Objective = size + penalty * (number of uncovered words)
            # If invalid, penalty is huge.
            
            uncovered_count = (all_mask ^ new_cover).bit_count()
            
            # Current objective
            if current_cover == all_mask:
                obj_curr = current_size
            else:
                obj_curr = current_size + 1000 * (all_mask ^ current_cover).bit_count()
                
            if new_cover == all_mask:
                obj_new = new_size
            else:
                obj_new = new_size + 1000 * uncovered_count
                
            delta_obj = obj_new - obj_curr
            
            # Acceptance probability
            if delta_obj < 0:
                accept = True
            else:
                # Boltzmann factor
                # T is temperature
                # If T is small, delta_obj > 0 is rejected.
                prob = np.exp(-delta_obj / T)
                accept = random.random() < prob
            
            if accept:
                current_code = new_code
                current_set = new_set
                current_cover = new_cover
                current_size = new_size
                
                # Update best
                if current_cover == all_mask and current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
            
            T *= cooling_rate
            
        return best_code

    # Run Greedy to get a starting point
    # Then run SA
    # We can run multiple restarts if time permits
    
    best_candidate = None
    best_size = float('inf')
    
    # Run one greedy + SA
    # Greedy is fast (~1-2s)
    code = greedy_construct(42)
    
    if code:
        # Check validity of greedy result
        cover = get_cover_mask(code)
        if cover == all_mask:
            # Run SA
            sa_code = simulated_annealing(code)
            if sa_code:
                sa_cover = get_cover_mask(sa_code)
                if sa_cover == all_mask and len(sa_code) < best_size:
                    best_size = len(sa_code)
                    best_candidate = sa_code
    
    # If we have time, try another seed
    if time.time() < deadline - 60.0:
        code2 = greedy_construct(123)
        if code2:
            cover2 = get_cover_mask(code2)
            if cover2 == all_mask:
                sa_code2 = simulated_annealing(code2)
                if sa_code2:
                    sa_cover2 = get_cover_mask(sa_code2)
                    if sa_cover2 == all_mask and len(sa_code2) < best_size:
                        best_size = len(sa_code2)
                        best_candidate = sa_code2

    # If we have even more time, try a third
    if time.time() < deadline - 30.0:
        code3 = greedy_construct(456)
        if code3:
            cover3 = get_cover_mask(code3)
            if cover3 == all_mask:
                sa_code3 = simulated_annealing(code3)
                if sa_code3:
                    sa_cover3 = get_cover_mask(sa_code3)
                    if sa_cover3 == all_mask and len(sa_code3) < best_size:
                        best_size = len(sa_code3)
                        best_candidate = sa_code3

    if best_candidate is None:
        # Fallback to tetracode direct sum
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(digits_of[i]) for i in best_candidate]
