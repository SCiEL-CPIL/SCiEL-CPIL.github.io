import itertools
import math
import time
import numpy as np

def _score(U):
    U = list(set(U))
    if 0 not in U:
        return 0.0
    mx = max(U)
    if mx == 0:
        return 0.0
    U_arr = np.array(U)
    plus = np.add.outer(U_arr, U_arr)
    minus = np.subtract.outer(U_arr, U_arr)
    n_plus = len(np.unique(plus))
    n_minus = len(np.unique(minus))
    if n_plus == 0:
        return 0.0
    ratio = n_minus / n_plus
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    best_score = 0.0
    best_set = [0, 1, 3]

    def try_set(U):
        nonlocal best_score, best_set
        s = _score(U)
        if s > best_score:
            best_score = s
            best_set = list(U)

    # Phase 1: Try structured sets
    structured_sets = []
    # Known good small sets
    structured_sets.append([0, 1, 2, 4])
    structured_sets.append([0, 1, 2, 4, 8])
    structured_sets.append([0, 1, 3, 4, 8])
    
    # Powers of 2
    for n in range(2, 12):
        structured_sets.append([0] + [2**i for i in range(n)])
    # Small arithmetic progressions with gaps
    for a in range(1, 8):
        for d in range(1, 8):
            structured_sets.append([0] + [a * i * d for i in range(1, 7)])
    # Mixed: small + large
    for small in range(1, 6):
        for large in range(10, 50, 3):
            structured_sets.append([0, small, large])
            structured_sets.append([0, small, large, 2*large])
            
    # Fibonacci-like
    fib = [0, 1]
    for i in range(10):
        fib.append(fib[-1] + fib[-2])
    for i in range(2, len(fib)):
        structured_sets.append(fib[:i+1])

    for U in structured_sets:
        if time.time() > deadline:
            return best_set
        try_set(U)

    # Phase 2: Exhaustive search over small ranges
    max_val = 30
    candidates = list(range(1, max_val + 1))

    for size in range(2, 8):
        if time.time() > deadline:
            return best_set
        for combo in itertools.combinations(candidates, size - 1):
            if time.time() > deadline:
                return best_set
            U = [0] + list(combo)
            try_set(U)

    # Phase 3: Robust Local Search
    # Start from the best set found so far
    current = list(best_set)
    current_score = best_score
    
    # Define a range of candidates for moves
    # We want to explore adding numbers that are not too large but also not too small
    max_candidate = 100
    
    while time.time() < deadline - 2.0:
        improved = False
        
        # 1. Try adding a new element
        # Check a diverse set of candidates: small integers, multiples of existing elements, etc.
        candidates_to_try = set()
        for x in range(1, max_candidate + 1):
            candidates_to_try.add(x)
        # Add some structured candidates based on current set
        for x in current:
            for y in current:
                if x != y:
                    candidates_to_try.add(x + y)
                    candidates_to_try.add(abs(x - y))
                    if x * y < max_candidate:
                        candidates_to_try.add(x * y)
        
        # Filter candidates to those not in current set and within reasonable bounds
        valid_candidates = [x for x in candidates_to_try if x not in current and x <= max_candidate]
        
        best_add_score = current_score
        best_add_x = None
        for x in valid_candidates:
            if time.time() > deadline - 2.0: break
            new_U = sorted(current + [x])
            s = _score(new_U)
            if s > best_add_score:
                best_add_score = s
                best_add_x = x
        
        if best_add_x is not None:
            current = sorted(current + [best_add_x])
            current_score = best_add_score
            improved = True
            try_set(current)
            continue # Restart loop to re-evaluate other moves with new state

        # 2. Try removing an element
        best_rem_score = current_score
        best_rem_idx = None
        for i in range(1, len(current)):
            if time.time() > deadline - 2.0: break
            new_U = current[:i] + current[i+1:]
            s = _score(new_U)
            if s > best_rem_score:
                best_rem_score = s
                best_rem_idx = i
        
        if best_rem_idx is not None:
            current = current[:best_rem_idx] + current[best_rem_idx+1:]
            current_score = best_rem_score
            improved = True
            try_set(current)
            continue # Restart loop

        # 3. Try swapping an element
        best_swap_score = current_score
        best_swap_i = None
        best_swap_x = None
        for i in range(1, len(current)):
            if time.time() > deadline - 2.0: break
            # Try swapping current[i] with a candidate
            # To save time, only try a subset of candidates for swaps
            swap_candidates = [x for x in range(1, max_candidate + 1) if x not in current]
            # Sort by some heuristic? Or just iterate?
            # Let's iterate but break early if no improvement found in first pass
            for x in swap_candidates:
                if time.time() > deadline - 2.0: break
                new_U = current[:i] + [x] + current[i+1:]
                s = _score(new_U)
                if s > best_swap_score:
                    best_swap_score = s
                    best_swap_i = i
                    best_swap_x = x
        
        if best_swap_i is not None:
            current = current[:best_swap_i] + [best_swap_x] + current[best_swap_i+1:]
            current = sorted(current)
            current_score = best_swap_score
            improved = True
            try_set(current)
            continue # Restart loop

        if not improved:
            # No single move improves the score. 
            # Try a "pair move": add one, remove one simultaneously?
            # This is more expensive. Let's do a simple version:
            # For each element to remove, find the best element to add.
            best_pair_score = current_score
            best_pair_rem = None
            best_pair_add = None
            
            # To keep it fast, limit the number of removal candidates
            removal_candidates = list(range(1, len(current)))
            if len(removal_candidates) > 10:
                # Random sample or just first 10? Let's try all if small, else sample
                removal_candidates = removal_candidates[:10]
            
            for i in removal_candidates:
                if time.time() > deadline - 2.0: break
                # Remove current[i]
                base_U = current[:i] + current[i+1:]
                # Try adding a new element to base_U
                # Use a smaller candidate set for this inner loop
                add_candidates = [x for x in range(1, 50) if x not in base_U]
                for x in add_candidates:
                    if time.time() > deadline - 2.0: break
                    new_U = sorted(base_U + [x])
                    s = _score(new_U)
                    if s > best_pair_score:
                        best_pair_score = s
                        best_pair_rem = i
                        best_pair_add = x
            
            if best_pair_rem is not None:
                # Apply the pair move
                new_U = current[:best_pair_rem] + current[best_pair_rem+1:]
                new_U = sorted(new_U + [best_pair_add])
                current = new_U
                current_score = best_pair_score
                improved = True
                try_set(current)
                continue # Restart loop
            
            # If no pair move improved, we are stuck in a local optimum.
            # Break the local search loop.
            break

    # Final polish: Ensure we return the best set found
    return best_set
