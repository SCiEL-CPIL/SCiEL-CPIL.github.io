import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    best_set = []
    best_size = 0
    
    # Helper to check if adding x to S is valid, using incremental sums
    # S: list of elements
    # triple_sums: set of all sums of 3 distinct elements in S
    # x: candidate
    # Returns: (is_valid, new_triple_sums)
    def try_add(S, triple_sums, x):
        # 1. Check if x divides any existing triple sum
        for s in triple_sums:
            if s % x == 0:
                return False, None
        
        # 2. Generate new triple sums involving x
        # New sums are x + a + b for a, b in S, a != b
        new_sums = []
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                s_new = x + S[i] + S[j]
                new_sums.append(s_new)
        
        # 3. Check if any a in S divides any new sum
        for s_new in new_sums:
            for a in S:
                if s_new % a == 0:
                    return False, None
        
        # 4. If valid, update triple_sums
        new_triple_sums = set(triple_sums)
        new_triple_sums.update(new_sums)
        
        return True, new_triple_sums

    # Greedy construction
    def greedy_construct(order):
        S = []
        triple_sums = set()
        for x in order:
            valid, new_sums = try_add(S, triple_sums, x)
            if valid:
                S.append(x)
                triple_sums = new_sums
        return S, triple_sums

    # Local search: try to improve size
    def local_search(S, triple_sums):
        S = list(S)
        triple_sums = set(triple_sums)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add any element
            candidates = [x for x in range(1, 193) if x not in S]
            random.shuffle(candidates)
            added = False
            for x in candidates:
                valid, new_sums = try_add(S, triple_sums, x)
                if valid:
                    S.append(x)
                    triple_sums = new_sums
                    improved = True
                    added = True
                    break
            
            if added:
                continue
            
            # If no add possible, try swaps to escape local optimum
            # Swap x in S with y not in S
            if S:
                x_idx = random.randint(0, len(S) - 1)
                x = S[x_idx]
                
                # Try a few random y
                for _ in range(10):
                    y = random.randint(1, 192)
                    if y in S:
                        continue
                    
                    # Create S' = (S \ {x}) U {y}
                    S_temp = S[:x_idx] + S[x_idx+1:]
                    # We need triple_sums for S_temp
                    # This is expensive to recompute from scratch.
                    # Alternative: Just check validity of S_temp U {y} from scratch?
                    # Or approximate?
                    # Let's just do a full check for the new set S_new = S_temp + [y]
                    S_new = S_temp + [y]
                    if len(S_new) != len(S): continue # Should be same size
                    
                    # Full validity check for S_new
                    is_valid = True
                    # Check if any a in S_new divides sum of 3 others
                    # We can reuse the incremental logic if we had sums for S_temp.
                    # But we don't.
                    # Let's just do a fast check:
                    # Since we are swapping, maybe just check if S_new is valid.
                    # This might be slow.
                    
                    # Optimization: Only do swaps if we have time and stuck.
                    # Given the speed of try_add, maybe we can reconstruct sums for S_temp?
                    # No, too complex.
                    
                    # Let's just try to add elements in different orders in the main loop.
                    # Swaps are risky for time.
                    pass
                    
        return S, triple_sums

    # Main loop: Multiple restarts
    # Strategy: 
    # 1. Try sorted descending (seed behavior)
    # 2. Try random shuffles
    # 3. Try "smart" orders (e.g., primes first, or numbers with few divisors)
    
    # 1. Descending
    order = list(range(192, 0, -1))
    S, sums = greedy_construct(order)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
        best_sums = sums
        
    if time.time() > deadline:
        return best_set

    # 2. Random restarts
    num_restarts = 0
    while time.time() < deadline - 5.0:
        order = list(range(1, 193))
        random.shuffle(order)
        S, sums = greedy_construct(order)
        
        # Local search: try to add more
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            candidates = [x for x in range(1, 193) if x not in S]
            random.shuffle(candidates)
            for x in candidates:
                valid, new_sums = try_add(S, sums, x)
                if valid:
                    S.append(x)
                    sums = new_sums
                    improved = True
                    break
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            best_sums = sums
            # If we reached 48, we can stop? No, max is 48.
            if best_size == 48:
                break
        num_restarts += 1
        
    return best_set
