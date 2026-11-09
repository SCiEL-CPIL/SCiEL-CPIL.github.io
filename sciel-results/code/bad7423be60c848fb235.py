import time
import itertools
import random

_LCM = 232792560

def _get_sums(subset):
    """Calculate all subset sums of the scaled reciprocals."""
    scaled = [_LCM // x for x in subset]
    sums = {0}
    for v in scaled:
        sums |= {s + v for s in sums}
    return sums

def _is_dissociated(subset):
    """Check if the set is dissociated."""
    sums = _get_sums(subset)
    return len(sums) == (1 << len(subset))

def _greedy_construct(order):
    """Construct a dissociated set using a specific order of candidates."""
    S = []
    sums = {0}
    for x in order:
        v = _LCM // x
        # Check if adding v causes a collision
        # New sums would be sums U {s+v for s in sums}
        # Collision if (s+v) in sums for any s in sums
        # This is equivalent to checking if (s1 - s2) == v for any s1, s2 in sums
        # Or simpler: just generate new sums and check size
        new_sums = sums | {s + v for s in sums}
        if len(new_sums) == len(sums) * 2:
            S.append(x)
            sums = new_sums
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    universe = list(range(1, 23))
    universe_set = set(universe)
    
    best_S = []
    best_size = 0
    
    # 1. Initialize with multiple greedy strategies
    orders = [
        sorted(universe, reverse=True),
        sorted(universe),
        sorted(universe, key=lambda x: _LCM // x), # Smallest scaled values first
        sorted(universe, key=lambda x: -(_LCM // x)) # Largest scaled values first
    ]
    
    # Add some random orders
    for _ in range(20):
        r_order = universe.copy()
        random.shuffle(r_order)
        orders.append(r_order)
        
    for order in orders:
        if time.time() > deadline - 30: # Reserve time for search
            break
        S = _greedy_construct(order)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
            
    # 2. Local Search
    # We maintain the current set S and its sums
    # To be efficient, we don't recompute sums from scratch for every candidate if possible,
    # but given the small size, recomputation is fine and less error-prone.
    
    current_S = best_S.copy()
    current_size = best_size
    
    # Precompute scaled values for quick access
    scaled_vals = {x: _LCM // x for x in universe}
    
    improved = True
    no_improve_count = 0
    max_no_improve = 5000 # Limit sideways moves before restarting or stopping
    
    while time.time() < deadline - 5:
        improved = False
        in_set = set(current_S)
        not_in_set = [x for x in universe if x not in in_set]
        
        # --- Phase 1: Try to Increase Size ---
        
        # 1.1: Try adding 1 element (1-for-0)
        # This is the cheapest move
        for x in not_in_set:
            if time.time() > deadline - 5:
                return best_S
            candidate = current_S + [x]
            if _is_dissociated(candidate):
                current_S = candidate
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_S = current_S.copy()
                improved = True
                break # Restart loop with new set
                
        if improved:
            continue
            
        # 1.2: Try swapping 1 element for 2 (1-for-2)
        # Cost: O(N * C(N,2)) checks
        moves_1_for_2 = []
        for a in current_S:
            for x, y in itertools.combinations(not_in_set, 2):
                moves_1_for_2.append((a, x, y))
        
        random.shuffle(moves_1_for_2)
        
        for a, x, y in moves_1_for_2:
            if time.time() > deadline - 5:
                return best_S
            # Create candidate
            # Remove a, add x, y
            cand_list = [z for z in current_S if z != a] + [x, y]
            if _is_dissociated(cand_list):
                current_S = cand_list
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_S = current_S.copy()
                improved = True
                break # Restart loop
                
        if improved:
            continue
            
        # 1.3: Try swapping 2 elements for 3 (2-for-3)
        # Cost: O(C(N,2) * C(N,3)) checks -- Expensive, only if time allows
        # We might skip this if 1-for-2 is frequent, but it's a good escape hatch.
        # Let's try a limited number of these.
        
        moves_2_for_3 = []
        for a, b in itertools.combinations(current_S, 2):
            for x, y, z in itertools.combinations(not_in_set, 3):
                moves_2_for_3.append((a, b, x, y, z))
                
        if len(moves_2_for_3) > 0:
            random.shuffle(moves_2_for_3)
            # Limit the number of trials to avoid wasting time
            trials = min(100, len(moves_2_for_3))
            for a, b, x, y, z in moves_2_for_3[:trials]:
                if time.time() > deadline - 5:
                    return best_S
                cand_list = [z_elem for z_elem in current_S if z_elem != a and z_elem != b] + [x, y, z]
                if _is_dissociated(cand_list):
                    current_S = cand_list
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_S = current_S.copy()
                    improved = True
                    break # Restart loop
                    
        if improved:
            continue
            
        # --- Phase 2: Sideways Moves (Maintain Size) ---
        # If we can't increase size, we try to change the configuration.
        # This helps escape local optima.
        
        # 2.1: Try swapping 1 element for 1 (1-for-1)
        moves_1_for_1 = []
        for a in current_S:
            for x in not_in_set:
                moves_1_for_1.append((a, x))
                
        random.shuffle(moves_1_for_1)
        # Try a subset of these
        trials = min(100, len(moves_1_for_1))
        for a, x in moves_1_for_1[:trials]:
            if time.time() > deadline - 5:
                return best_S
            cand_list = [z for z in current_S if z != a] + [x]
            if _is_dissociated(cand_list):
                current_S = cand_list
                # Size remains same, so no update to best_size unless we track best_S separately
                # But this move might enable future size increases.
                improved = True
                break
                
        if improved:
            no_improve_count = 0
            continue
            
        # 2.2: Try swapping 2 elements for 2 (2-for-2)
        moves_2_for_2 = []
        for a, b in itertools.combinations(current_S, 2):
            for x, y in itertools.combinations(not_in_set, 2):
                moves_2_for_2.append((a, b, x, y))
                
        if len(moves_2_for_2) > 0:
            random.shuffle(moves_2_for_2)
            trials = min(100, len(moves_2_for_2))
            for a, b, x, y in moves_2_for_2[:trials]:
                if time.time() > deadline - 5:
                    return best_S
                cand_list = [z for z in current_S if z != a and z != b] + [x, y]
                if _is_dissociated(cand_list):
                    current_S = cand_list
                    improved = True
                    break
                    
        if improved:
            no_improve_count = 0
            continue
            
        # No move improved or changed the set.
        no_improve_count += 1
        
        # If we are stuck for too long, restart with a new random greedy
        if no_improve_count > max_no_improve:
            # Try a fresh start
            r_order = universe.copy()
            random.shuffle(r_order)
            S = _greedy_construct(r_order)
            if len(S) > current_size:
                current_S = S
                current_size = len(S)
                if current_size > best_size:
                    best_size = current_size
                    best_S = current_S.copy()
            no_improve_count = 0
            # Even if not better, we reset the current state to explore a different region
            # Actually, if it's not better, we should probably keep the best known and try to improve it,
            # but if we are stuck, restarting is a common heuristic.
            # Let's just reset current_S to the best_S found so far and try again.
            current_S = best_S.copy()
            current_size = best_size

    return best_S
