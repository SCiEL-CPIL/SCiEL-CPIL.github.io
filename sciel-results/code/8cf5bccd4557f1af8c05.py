def construct():
    import time
    from math import gcd
    import random
    import sys

    deadline = time.time() + 50.0  # Safety margin
    N = 128
    all_numbers = list(range(1, N + 1))
    
    # Precompute GCD table
    g = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g[i][j] = gcd(i, j)
            g[j][i] = g[i][j]

    # Precompute conflict masks
    # conflict_mask[a][b] is a bitmask of all c such that (a,b,c) is a bad triple
    # i.e., gcd(a,b) == gcd(a,c) == gcd(b,c) and c != a, c != b
    # Using 1-indexed bits, so bit k corresponds to number k
    conflict_mask = [[0] * (N + 1) for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = g[a][b]
            mask = 0
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if g[a][c] == gab and g[b][c] == gab:
                    mask |= (1 << c)
            conflict_mask[a][b] = mask
            conflict_mask[b][a] = mask

    def is_valid_addition(x, S_mask, S_list):
        """Check if x can be added to S without creating a bad triple."""
        # For every pair (a,b) in S, check if x is in conflict_mask[a][b]
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                if conflict_mask[a][b] & (1 << x):
                    return False
        return True

    def greedy_construct(order):
        """Greedily construct a valid set from the given order."""
        S = set()
        S_mask = 0
        S_list = []
        for x in order:
            if x in S:
                continue
            if is_valid_addition(x, S_mask, S_list):
                S.add(x)
                S_mask |= (1 << x)
                S_list.append(x)
        return S_list

    def local_improve(S_list, time_budget):
        """Try to improve the set by adding elements or swapping."""
        S = set(S_list)
        S_list = list(S_list)
        S_mask = 0
        for x in S_list:
            S_mask |= (1 << x)
        
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            # Try to add any missing element
            for x in range(1, N + 1):
                if x in S:
                    continue
                if is_valid_addition(x, S_mask, S_list):
                    S.add(x)
                    S_list.append(x)
                    S_mask |= (1 << x)
                    improved = True
                    break # Restart loop to try adding more
            
            if time.time() >= time_budget:
                break
                
            if not improved:
                # Try swap: remove one, add one (to maintain size, useful for escaping local minima in structure, 
                # though we want to increase size, sometimes removing a "blocking" element allows adding two)
                # Let's try Remove 1, Add 2
                found_swap = False
                for rem in S_list:
                    S_trial = S - {rem}
                    S_trial_list = [x for x in S_list if x != rem]
                    S_trial_mask = S_mask & ~(1 << rem)
                    
                    # Try to add as many as possible from complement
                    added = 0
                    for x in range(1, N + 1):
                        if x in S_trial:
                            continue
                        if is_valid_addition(x, S_trial_mask, S_trial_list):
                            S_trial.add(x)
                            S_trial_list.append(x)
                            S_trial_mask |= (1 << x)
                            added += 1
                            
                    if len(S_trial) > len(S):
                        S = S_trial
                        S_list = list(S)
                        S_mask = S_trial_mask
                        improved = True
                        found_swap = True
                        break
                        
                if found_swap:
                    continue
                # If no improvement, break to avoid infinite loop on this instance
                # Or break and let the outer loop restart
                break
                
        return S_list

    best_S = []
    best_size = 0
    start_time = time.time()
    
    # 1. Initial Greedy Passes with different orderings
    random.seed(42)
    
    # Natural order
    S = greedy_construct(list(range(1, N + 1)))
    if len(S) > best_size:
        best_size = len(S)
        best_S = S
    
    # Reverse order
    S = greedy_construct(list(range(N, 0, -1)))
    if len(S) > best_size:
        best_size = len(S)
        best_S = S
        
    # Random orderings
    for i in range(200):
        if time.time() > deadline - 20:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        S = greedy_construct(order)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
            
    # 2. Local Improvement on best found
    if time.time() < deadline - 10:
        best_S = local_improve(best_S, deadline - 5)
        best_size = len(best_S)
        
    # 3. Iterated Local Search (ILS)
    # Perturb best_S and re-optimize
    for i in range(50):
        if time.time() > deadline - 5:
            break
        # Perturb: remove 3 random elements
        S_perturbed = set(best_S)
        if len(S_perturbed) > 3:
            to_remove = random.sample(list(S_perturbed), 3)
            for x in to_remove:
                S_perturbed.discard(x)
        else:
            S_perturbed = set()
            
        S_opt = local_improve(list(S_perturbed), deadline - 5)
        if len(S_opt) > best_size:
            best_size = len(S_opt)
            best_S = S_opt
            
    # Final Verification
    if not best_S:
        best_S = [1]
        
    # Verify no violations in best_S
    S_set = set(best_S)
    valid = True
    for i in range(len(best_S)):
        a = best_S[i]
        for j in range(i + 1, len(best_S)):
            b = best_S[j]
            if conflict_mask[a][b] & (sum(1 << x for x in best_S if x != a and x != b)):
                # Check if there is any c in S, c!=a,b such that (a,b,c) is bad
                # The mask contains all such c. If intersection with S\{a,b} is non-zero, invalid.
                # Simpler: just check the mask against S_mask with a,b cleared
                mask_ab = conflict_mask[a][b]
                S_mask = 0
                for x in best_S:
                    S_mask |= (1 << x)
                if mask_ab & (S_mask & ~(1 << a) & ~(1 << b)):
                    valid = False
                    break
        if not valid:
            break
            
    if not valid:
        # Fallback: remove elements until valid (should not happen if logic is correct)
        while not valid:
            # Find a violation
            viol = None
            for i in range(len(best_S)):
                a = best_S[i]
                for j in range(i + 1, len(best_S)):
                    b = best_S[j]
                    mask_ab = conflict_mask[a][b]
                    S_mask = 0
                    for x in best_S:
                        S_mask |= (1 << x)
                    if mask_ab & (S_mask & ~(1 << a) & ~(1 << b)):
                        # Find the specific c
                        c_mask = mask_ab & (S_mask & ~(1 << a) & ~(1 << b))
                        # Get lowest set bit
                        c = (c_mask & -c_mask).bit_length()
                        viol = (a, b, c)
                        break
                if viol:
                    break
            if not viol:
                break
            # Remove one of the violating elements
            a, b, c = viol
            best_S.remove(c)
            valid = True

    if not best_S:
        best_S = [1]

    return best_S
