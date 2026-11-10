# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    
    Strategy:
    1. Compute forbidden differences F = {x^2 mod 205 : x in 1..204} \ {0}.
    2. Represent the set of selected elements as a bitmask (integer).
    3. Precompute conflict masks: conflict_mask[i] has bit j set if i and j conflict.
    4. Use Greedy Initialization with multiple orderings.
    5. Use Iterated Local Search (ILS) with bitmask operations for O(1) conflict checks.
       - Moves: Add, Swap, Remove.
       - Greedy Fill after moves to maximize size quickly.
    6. Deadline guard to ensure completion within 600s.
    """
    m = 205
    start_time = time.time()
    deadline = start_time + 570  # Safety margin of 30s
    
    # 1. Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        val = (x * x) % m
        if val != 0:
            forbidden.add(val)
    
    # 2. Precompute conflict masks
    # conflict_mask[i] is an integer where bit j is 1 if (i-j) % m or (j-i) % m is in forbidden
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden or d2 in forbidden:
                mask |= (1 << j)
        conflict_mask[i] = mask
    
    # Helper to check if candidate 'c' can be added to set represented by 'mask'
    def can_add_bitmask(c, mask):
        # If c is already in mask, it's not a valid "add" in the sense of increasing size, 
        # but for conflict check: does c conflict with any element in mask?
        # We want to know if (mask & conflict_mask[c]) == 0
        return (mask & conflict_mask[c]) == 0

    def popcount(x):
        return bin(x).count('1')

    def mask_to_list(mask):
        return [i for i in range(m) if (mask >> i) & 1]

    # Greedy initialization with bitmask
    def greedy_init_bitmask(order):
        mask = 0
        for cand in order:
            if can_add_bitmask(cand, mask):
                mask |= (1 << cand)
        return mask

    # Generate initial masks
    best_init_mask = 0
    best_init_size = -1
    
    orders = [
        list(range(m)),
        list(range(m-1, -1, -1)),
        list(range(0, m, 2)),
        list(range(1, m, 2)),
        list(range(0, m, 3)),
        list(range(1, m, 3)),
        list(range(2, m, 3)),
        list(range(0, m, 5)),
        list(range(1, m, 5)),
        list(range(2, m, 5)),
        list(range(3, m, 5)),
        list(range(4, m, 5)),
        list(range(0, m, 7)),
        list(range(1, m, 7)),
        list(range(2, m, 7)),
        list(range(0, m, 11)),
        list(range(1, m, 11)),
        list(range(0, m, 13)),
        list(range(1, m, 13)),
        list(range(0, m, 17)),
        list(range(1, m, 17)),
        list(range(0, m, 19)),
        list(range(1, m, 19)),
    ]
    
    # Add random orderings
    for seed in range(50):
        rng = random.Random(seed)
        order = list(range(m))
        rng.shuffle(order)
        orders.append(order)
        
    for order in orders:
        init_mask = greedy_init_bitmask(order)
        size = popcount(init_mask)
        if size > best_init_size:
            best_init_size = size
            best_init_mask = init_mask
            
    if best_init_mask == 0:
        best_init_mask = 1 # Start with {0}
        best_init_size = 1

    # Global best
    global_best_mask = best_init_mask
    global_best_size = best_init_size
    
    # Local search function
    def local_search(initial_mask, rng, time_limit):
        current_mask = initial_mask
        current_size = popcount(current_mask)
        best_local_mask = current_mask
        best_local_size = current_size
        
        # Greedy fill first
        improved = True
        while improved:
            improved = False
            for c in range(m):
                if not (current_mask & (1 << c)):
                    if can_add_bitmask(c, current_mask):
                        current_mask |= (1 << c)
                        current_size += 1
                        improved = True
            if current_size > best_local_size:
                best_local_size = current_size
                best_local_mask = current_mask
                
        # Randomized local search
        # We use a simple loop until time limit
        last_time = time.time()
        while time.time() < time_limit:
            move_type = rng.random()
            
            if move_type < 0.5:
                # Try to add a random element
                # Find candidates not in mask
                # For efficiency, we can just pick a random c and check
                c = rng.randint(0, m - 1)
                if not (current_mask & (1 << c)):
                    if can_add_bitmask(c, current_mask):
                        current_mask |= (1 << c)
                        current_size += 1
                        if current_size > best_local_size:
                            best_local_size = current_size
                            best_local_mask = current_mask
                            # Greedy fill after improvement
                            improved = True
                            while improved:
                                improved = False
                                for k in range(m):
                                    if not (current_mask & (1 << k)):
                                        if can_add_bitmask(k, current_mask):
                                            current_mask |= (1 << k)
                                            current_size += 1
                                            improved = True
                                if current_size > best_local_size:
                                    best_local_size = current_size
                                    best_local_mask = current_mask
                                    
            elif move_type < 0.85:
                # Swap: remove one, add one
                # Pick a random element in mask to remove
                # To pick a random set bit, we can use a list of indices or random guessing
                # Given m=205, a list is fast enough, but let's try to be efficient
                # We'll maintain a list of selected elements for this local search? 
                # No, let's just pick a random index and check if it's in mask.
                # If not, retry.
                # Or better: generate a list of set bits once per iteration? That's O(N).
                # Let's just pick random indices.
                a = rng.randint(0, m - 1)
                if (current_mask & (1 << a)):
                    # Try to find a b not in mask to add
                    b = rng.randint(0, m - 1)
                    if not (current_mask & (1 << b)):
                        # Check if b can be added if a is removed
                        # New mask would be current_mask ^ (1 << a)
                        # We need ( (current_mask ^ (1 << a)) & conflict_mask[b] ) == 0
                        # Note: conflict_mask[b] does not include b itself.
                        # If a is in conflict_mask[b], then removing a helps.
                        # If a is NOT in conflict_mask[b], then removing a doesn't change the conflict status with other elements.
                        # So we just check:
                        temp_mask = current_mask ^ (1 << a)
                        if (temp_mask & conflict_mask[b]) == 0:
                            # Valid swap
                            current_mask = temp_mask | (1 << b)
                            # Size remains same
                            # Check if this new configuration allows greedy fill to improve
                            improved = True
                            while improved:
                                improved = False
                                for k in range(m):
                                    if not (current_mask & (1 << k)):
                                        if can_add_bitmask(k, current_mask):
                                            current_mask |= (1 << k)
                                            current_size += 1
                                            improved = True
                                if current_size > best_local_size:
                                    best_local_size = current_size
                                    best_local_mask = current_mask
            else:
                # Remove one to break local optimum
                a = rng.randint(0, m - 1)
                if (current_mask & (1 << a)):
                    current_mask ^= (1 << a)
                    current_size -= 1
                    
            # Periodic greedy fill
            if rng.random() < 0.1:
                improved = True
                while improved:
                    improved = False
                    for k in range(m):
                        if not (current_mask & (1 << k)):
                            if can_add_bitmask(k, current_mask):
                                current_mask |= (1 << k)
                                current_size += 1
                                improved = True
                    if current_size > best_local_size:
                        best_local_size = current_size
                        best_local_mask = current_mask

        return best_local_mask, best_local_size

    # Main search loop with restarts
    # We will run local search on the best initial mask, and then on perturbed versions
    rng_main = random.Random(42)
    
    # Run local search on the best initial mask
    local_limit = deadline - 30
    best_mask, best_size = local_search(best_init_mask, rng_main, local_limit)
    if best_size > global_best_size:
        global_best_size = best_size
        global_best_mask = best_mask
        
    # If we have time, try other initial masks or perturbations
    # Since local_search takes a time limit, we should allocate time carefully.
    # Let's run a few more iterations if time permits.
    
    # For simplicity and robustness, we rely on the first deep search.
    # However, if the first search finishes quickly, we can try others.
    # The local_search function runs until time_limit. 
    # So the first call will consume most of the time.
    
    # If we want to be safer, we can split the time.
    # But typically, one long run of ILS is better than many short ones for this size.
    
    # Convert best mask to array
    result_list = mask_to_list(global_best_mask)
    
    # Verify
    # The problem requires distinct residues in 0..204.
    # The bitmask ensures distinctness and range.
    # We should double check no forbidden differences.
    # This is a safety check.
    for i in range(len(result_list)):
        for j in range(i+1, len(result_list)):
            a, b = result_list[i], result_list[j]
            d1 = (a - b) % m
            d2 = (b - a) % m
            if d1 in forbidden or d2 in forbidden:
                # This should not happen if logic is correct
                # Fallback to a known valid small set
                result_list = [0]
                break
    
    return np.array(result_list, dtype=np.int64)

# EVOLVE-BLOCK-END
