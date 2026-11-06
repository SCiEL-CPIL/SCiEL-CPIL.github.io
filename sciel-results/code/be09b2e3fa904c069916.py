import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 58.0  # Safety margin for 60s limit
    
    n = 256
    # Precompute masks for 3APs? 
    # A 3AP is {a, b, c} with b = (a+c)/2.
    # For a set represented as a bitmask `mask`, adding `x` is valid if:
    # 1. x is not already in mask.
    # 2. No {a, x, c} exists in mask where a+c=2x.
    # 3. No {x, b, c} exists in mask where x+c=2b.
    # 4. No {a, b, x} exists in mask where a+x=2b.
    
    # To speed up checks, we can precompute for each x, the pairs (a, c) that would form a 3AP with x.
    # However, since n=256 is small, bit operations are very fast.
    # Let's represent the set as an integer bitmask.
    
    def is_3_free(mask):
        # This is for full validation if needed, but we will do incremental checks.
        # Actually, we just need to check if adding x to mask creates a 3AP.
        pass

    def can_add(mask, x):
        # Check if adding x to mask creates a 3AP.
        # x is 1-indexed in problem, 0-indexed in bit.
        # Let's use 0-indexed for bits: bit i represents element i+1.
        # So element x corresponds to bit x-1.
        bit_x = 1 << (x - 1)
        if mask & bit_x:
            return False
        
        # Check if x is middle: a, x, c. a+c = 2x.
        # a and c are in mask.
        # We need to check if there exists a, c in mask such that a + c = 2x.
        # Since a, c < n, we can iterate a.
        # a must be in mask. c = 2x - a must be in mask.
        # Also a != c (non-constant progression) and a != x, c != x (x not in mask yet).
        
        # Check 1: x is middle.
        for a in range(1, n + 1):
            if mask & (1 << (a - 1)):
                c = 2 * x - a
                if c == x: continue # a=x, not in mask
                if c < 1 or c > n: continue
                if c == a: continue # constant progression, but a != c usually? 
                # Wait, non-constant 3AP means a != c.
                # If a == c, then a = x, which is not in mask. So a != c is guaranteed if a, c in mask and x not in mask?
                # If a = x, then mask has x, which is false.
                # So a != x and c != x.
                # If a = c, then 2a = 2x => a=x, contradiction. So a != c.
                if mask & (1 << (c - 1)):
                    return False
        
        # Check 2: x is first term. x, b, c. x+c = 2b.
        # b, c in mask. c = 2b - x.
        for b in range(1, n + 1):
            if mask & (1 << (b - 1)):
                c = 2 * b - x
                if c == x: continue
                if c < 1 or c > n: continue
                if mask & (1 << (c - 1)):
                    return False
                    
        # Check 3: x is last term. a, b, x. a+x = 2b.
        # a, b in mask. a = 2b - x.
        for b in range(1, n + 1):
            if mask & (1 << (b - 1)):
                a = 2 * b - x
                if a == x: continue
                if a < 1 or a > n: continue
                if mask & (1 << (a - 1)):
                    return False
                    
        return True

    # Precompute reciprocal weights
    weights = [1.0 / i for i in range(1, n + 1)]
    
    best_score = -1.0
    best_set = None
    
    # Strategy: Multiple restarts of Simulated Annealing
    # Initial set: Greedy construction biased towards small numbers?
    # Or random small set?
    
    # Let's define a function to generate a random valid initial set
    def generate_initial_set():
        # Greedy: pick smallest available that fits?
        # This produces a specific set. 
        # To diversify, we can randomize the order of selection slightly or pick random valid elements.
        
        # Let's try a "biased greedy": 
        # Create a list of candidates 1..n
        # Shuffle with a bias towards smaller numbers?
        # Or just iterate 1..n and add if valid. This is deterministic.
        # To get diversity, we can randomly remove some elements from the greedy set?
        # Or start with a random subset and try to add?
        
        # Let's use a random restart strategy:
        # 1. Start with empty set.
        # 2. Create a list of all numbers.
        # 3. Shuffle the list, but weight smaller numbers higher?
        #    Actually, simpler: 
        #    Pick a random subset of size ~ n/3?
        #    Then try to add elements from 1..n if valid.
        
        # Let's try: Start with empty mask.
        # Iterate x from 1 to n.
        # With probability p (e.g., 0.8), if valid, add x.
        # This creates diverse initial sets.
        
        mask = 0
        for x in range(1, n + 1):
            if random.random() < 0.7: # Bias towards including small numbers
                if can_add(mask, x):
                    mask |= (1 << (x - 1))
        return mask

    def get_score(mask):
        # Sum of 1/x for x in mask
        s = 0.0
        # Iterate bits
        m = mask
        while m:
            # Get lowest set bit
            lb = m & -m
            idx = lb.bit_length() # 1-indexed position of the bit
            s += weights[idx - 1]
            m ^= lb
        return s

    def get_set_from_mask(mask):
        s = []
        m = mask
        while m:
            lb = m & -m
            idx = lb.bit_length()
            s.append(idx)
            m ^= lb
        return s

    # Simulated Annealing
    def simulated_annealing(initial_mask):
        current_mask = initial_mask
        current_score = get_score(current_mask)
        best_mask = current_mask
        best_score = current_score
        
        # Parameters
        T = 10.0 # Initial temperature
        cooling_rate = 0.995
        min_T = 0.01
        
        # Precompute potential moves?
        # Moves: 
        # 1. Add a missing element x.
        # 2. Remove an element x.
        # 3. Swap x (in set) with y (not in set).
        
        # To be efficient, we should focus on "good" moves.
        # Adding small numbers is good.
        # Removing large numbers is good.
        
        # Let's generate moves on the fly.
        
        while T > min_T and time.time() < deadline - 2.0:
            # Generate a random move
            move_type = random.randint(0, 2)
            
            if move_type == 0: # Add
                # Pick a random missing element
                missing = [x for x in range(1, n + 1) if not (current_mask & (1 << (x - 1)))]
                if not missing: continue
                x = random.choice(missing)
                if can_add(current_mask, x):
                    new_mask = current_mask | (1 << (x - 1))
                    delta = weights[x - 1]
                    if delta > 0 or random.random() < pow(2.718281828, -delta / T):
                        current_mask = new_mask
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
            elif move_type == 1: # Remove
                # Pick a random element in set
                # Prefer removing large elements? 
                # Random is fine for SA.
                in_set = [x for x in range(1, n + 1) if (current_mask & (1 << (x - 1)))]
                if not in_set: continue
                x = random.choice(in_set)
                new_mask = current_mask & ~(1 << (x - 1))
                delta = -weights[x - 1]
                if delta > 0 or random.random() < pow(2.718281828, -delta / T):
                    current_mask = new_mask
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
            else: # Swap
                # Pick x in set, y not in set
                in_set = [x for x in range(1, n + 1) if (current_mask & (1 << (x - 1)))]
                missing = [x for x in range(1, n + 1) if not (current_mask & (1 << (x - 1)))]
                if not in_set or not missing: continue
                
                # Bias: try to swap large x with small y
                # Just pick random for now, or weighted?
                # Let's pick random x from in_set, random y from missing.
                # Optimization: If y > x, delta is negative. We might want to bias towards y < x.
                # Let's just do random, SA will handle it.
                x = random.choice(in_set)
                y = random.choice(missing)
                
                # Check if swap is valid:
                # Remove x, add y.
                # (current_mask without x) + y must be 3-free.
                # This is equivalent to checking if y is valid in (current_mask without x).
                # And we already know current_mask is 3-free.
                # So we just need to check if adding y to (current_mask ^ bit_x) creates a 3AP.
                
                mask_without_x = current_mask & ~(1 << (x - 1))
                
                # Check if y can be added to mask_without_x
                # Reuse can_add logic but on mask_without_x
                # Note: can_add checks if y is in mask. y is not in current_mask, so not in mask_without_x.
                
                # We need a helper that takes a mask and an element y
                def can_add_to_mask(mask, y):
                    bit_y = 1 << (y - 1)
                    if mask & bit_y:
                        return False
                    
                    # Check 1: y is middle.
                    for a in range(1, n + 1):
                        if mask & (1 << (a - 1)):
                            c = 2 * y - a
                            if c == y: continue
                            if c < 1 or c > n: continue
                            if mask & (1 << (c - 1)):
                                return False
                            
                    # Check 2: y is first.
                    for b in range(1, n + 1):
                        if mask & (1 << (b - 1)):
                            c = 2 * b - y
                            if c == y: continue
                            if c < 1 or c > n: continue
                            if mask & (1 << (c - 1)):
                                return False
                            
                    # Check 3: y is last.
                    for b in range(1, n + 1):
                        if mask & (1 << (b - 1)):
                            a = 2 * b - y
                            if a == y: continue
                            if a < 1 or a > n: continue
                            if mask & (1 << (a - 1)):
                                return False
                            
                    return True

                if can_add_to_mask(mask_without_x, y):
                    new_mask = mask_without_x | (1 << (y - 1))
                    delta = weights[y - 1] - weights[x - 1]
                    if delta > 0 or random.random() < pow(2.718281828, -delta / T):
                        current_mask = new_mask
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
            
            T *= cooling_rate
        
        return best_mask, best_score

    # Run multiple restarts
    num_restarts = 10 # Adjust based on time
    
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        init_mask = generate_initial_set()
        mask, score = simulated_annealing(init_mask)
        
        if score > best_score:
            best_score = score
            best_set = get_set_from_mask(mask)
            
    # Final Polish: Try to add any missing small numbers to best_set
    # Convert best_set to mask
    final_mask = 0
    for x in best_set:
        final_mask |= (1 << (x - 1))
        
    # Try adding any missing x in ascending order (small x first)
    for x in range(1, n + 1):
        if not (final_mask & (1 << (x - 1))):
            if can_add(final_mask, x):
                final_mask |= (1 << (x - 1))
                
    # Try swapping large x with small missing y
    # We can do a few passes of this
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        for y in range(1, n + 1):
            if final_mask & (1 << (y - 1)):
                continue
            # Try to find x in final_mask such that x > y and swap is valid
            # Iterate x descending
            for x in range(n, y, -1):
                if not (final_mask & (1 << (x - 1))):
                    continue
                mask_without_x = final_mask & ~(1 << (x - 1))
                if can_add_to_mask(mask_without_x, y):
                    # Check if swap improves score
                    # 1/y > 1/x is always true if y < x
                    # So it always improves score if valid.
                    final_mask = mask_without_x | (1 << (y - 1))
                    improved = True
                    break # Restart y loop? Or just continue?
                    # If we swap, the set changes. We should restart the y loop to be safe?
                    # Or just continue to next y?
                    # Since y is now in set, we can move to next y.
                    # But other swaps might be possible.
                    # Let's break out of x loop and continue y loop.
                    # But the mask changed.
                    # It's safer to restart the whole while loop if we made a change.
                    break
            if improved:
                break
        
    # Final validation and return
    final_set = get_set_from_mask(final_mask)
    
    # Double check validity
    # This is slow, but n=256 is small.
    # O(n^3) check?
    # Let's just trust the incremental checks.
    # But for safety, if time permits, run a full check.
    
    return final_set

