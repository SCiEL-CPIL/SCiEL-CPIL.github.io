"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 85.0  # 5s safety margin
    n = 256
    
    def get_sums(S):
        """Compute the set of all sums b+c for b,c in S with b<=c."""
        S_list = sorted(S)
        sums = set()
        L = len(S_list)
        for i in range(L):
            b = S_list[i]
            for j in range(i, L):
                c = S_list[j]
                sums.add(b + c)
        return sums

    def can_add_small(x, S, sums_S):
        """
        Check if x can be added to S, assuming x < min(S).
        Condition: x must not divide any sum in sums_S.
        """
        for s in sums_S:
            if s % x == 0:
                return False
        return True

    def is_valid_set(S):
        """Full validity check for arbitrary S."""
        S_list = sorted(S)
        L = len(S_list)
        for i in range(L):
            a = S_list[i]
            for j in range(i + 1, L):
                b = S_list[j]
                for k in range(j, L):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    # Phase 1: Find the best starting interval [L, 256]
    best_start_L = 256
    for L in range(1, n + 1):
        if time.time() > deadline:
            break
        S_interval = set(range(L, n + 1))
        if is_valid_set(S_interval):
            best_start_L = L
            break

    # Start with the largest valid interval
    current_set = set(range(best_start_L, n + 1))
    
    # Phase 2: Greedy Extension
    sums_S = get_sums(current_set)
    
    for x in range(best_start_L - 1, 0, -1):
        if time.time() > deadline:
            break
        if x in current_set:
            continue
        if can_add_small(x, current_set, sums_S):
            current_set.add(x)
            sums_S = get_sums(current_set)

    best_set = set(current_set)
    best_size = len(best_set)

    # Phase 3: Conflict Graph Branch-and-Bound
    # Build a conflict graph where an edge between i and j means they cannot coexist
    # in a valid set (when combined with any third element from the set).
    
    # Precompute for each pair (i, j) with i < j whether they conflict
    # Two elements a < b conflict if there exists c in the set such that:
    #   - a divides b+c (when a is the smallest)
    #   - b divides a+c (when b is the smallest, but a<b so this won't happen as a+b < 2b)
    # Actually, the condition is: for a < b <= c in S, a must NOT divide b+c.
    # So for a fixed pair (a, b) with a < b, they conflict if there exists c >= b in S
    # such that a divides (b + c).
    
    # For branch-and-bound, we'll use a different approach:
    # We maintain a set S and try to add elements greedily, but with backtracking.
    
    # Let's use a smarter local search: try all possible additions and removals
    # and use the sums optimization.
    
    improved = True
    iteration = 0
    while improved and time.time() < deadline - 5:
        improved = False
        iteration += 1
        current_set = set(best_set)
        sums_S = get_sums(current_set)
        
        # Step 3a: Try to Add 1 element
        added_any = False
        for x in range(1, n + 1):
            if x in current_set:
                continue
            if x < min(current_set):
                if can_add_small(x, current_set, sums_S):
                    current_set.add(x)
                    added_any = True
                    break
            else:
                test_set = current_set | {x}
                if is_valid_set(test_set):
                    current_set = test_set
                    added_any = True
                    break
        
        if added_any:
            best_set = current_set
            best_size = len(best_set)
            improved = True
            continue

        # Step 3b: Try Add 2, Remove 1 (swap strategy)
        candidates_x = [x for x in range(1, n + 1) if x not in current_set]
        random.shuffle(candidates_x)
        
        for x in candidates_x:
            if time.time() > deadline - 5:
                break
            
            y_candidates = list(current_set)
            random.shuffle(y_candidates)
            
            for y in y_candidates:
                if time.time() > deadline - 5:
                    break
                
                S_minus_y = current_set - {y}
                
                if x < min(S_minus_y):
                    sums_minus_y = get_sums(S_minus_y)
                    if can_add_small(x, S_minus_y, sums_minus_y):
                        S_temp = S_minus_y | {x}
                        sums_temp = get_sums(S_temp)
                        
                        for z in range(1, n + 1):
                            if z in S_temp:
                                continue
                            if z < min(S_temp):
                                if can_add_small(z, S_temp, sums_temp):
                                    best_set = S_temp | {z}
                                    best_size = len(best_set)
                                    improved = True
                                    break
                            else:
                                test_set = S_temp | {z}
                                if is_valid_set(test_set):
                                    best_set = test_set
                                    best_size = len(best_set)
                                    improved = True
                                    break
                        
                        if improved:
                            break
                else:
                    test_set = S_minus_y | {x}
                    if is_valid_set(test_set):
                        S_temp = test_set
                        sums_temp = get_sums(S_temp)
                        
                        for z in range(1, n + 1):
                            if z in S_temp:
                                continue
                            if z < min(S_temp):
                                if can_add_small(z, S_temp, sums_temp):
                                    best_set = S_temp | {z}
                                    best_size = len(best_set)
                                    improved = True
                                    break
                            else:
                                test_set_z = S_temp | {z}
                                if is_valid_set(test_set_z):
                                    best_set = test_set_z
                                    best_size = len(best_set)
                                    improved = True
                                    break
                        
                        if improved:
                            break

        # Step 3c: Multi-swap - try Add k, Remove k for small k
        # This allows escaping deeper local minima
        if not improved and iteration % 3 == 0:
            # Try Add 2, Remove 2
            candidates_x = [x for x in range(1, n + 1) if x not in current_set]
            random.shuffle(candidates_x)
            candidates_x = candidates_x[:20]  # Limit for speed
            
            for x in candidates_x:
                if time.time() > deadline - 5:
                    break
                
                y_candidates = list(current_set)
                random.shuffle(y_candidates)
                y_candidates = y_candidates[:15]  # Limit for speed
                
                for y in y_candidates:
                    if time.time() > deadline - 5:
                        break
                    
                    S_minus_y = current_set - {y}
                    
                    # Try to add x to S_minus_y
                    x_valid = False
                    if x < min(S_minus_y):
                        sums_minus_y = get_sums(S_minus_y)
                        if can_add_small(x, S_minus_y, sums_minus_y):
                            x_valid = True
                    else:
                        if is_valid_set(S_minus_y | {x}):
                            x_valid = True
                    
                    if not x_valid:
                        continue
                    
                    S_temp = S_minus_y | {x}
                    
                    # Now try to find z and w such that removing w and adding z works
                    # Or just try to add one more element
                    sums_temp = get_sums(S_temp)
                    
                    for z in range(1, n + 1):
                        if z in S_temp:
                            continue
                        if z < min(S_temp):
                            if can_add_small(z, S_temp, sums_temp):
                                best_set = S_temp | {z}
                                best_size = len(best_set)
                                improved = True
                                break
                        else:
                            test_set = S_temp | {z}
                            if is_valid_set(test_set):
                                best_set = test_set
                                best_size = len(best_set)
                                improved = True
                                break
                        
                        if improved:
                            break
                    
                    if improved:
                        break
                
                if improved:
                    break

    if best_set is None:
        best_set = list(range(192, 257))
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
