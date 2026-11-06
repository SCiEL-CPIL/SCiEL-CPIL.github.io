"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    N = 128
    
    # Precompute GCD table
    gcd_table = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
    
    # Precompute bad pairs for each x
    # bad_pairs_for_x[x] is a list of (a, b) such that (a, b, x) is a bad triple
    # i.e., gcd(a,b) == gcd(a,x) == gcd(b,x)
    bad_pairs_for_x = [[] for _ in range(N+1)]
    
    for x in range(1, N+1):
        for a in range(1, N+1):
            if a == x:
                continue
            g_ax = gcd_table[a][x]
            for b in range(a+1, N+1):
                if b == x:
                    continue
                g_bx = gcd_table[b][x]
                if g_ax == g_bx:
                    if gcd_table[a][b] == g_ax:
                        bad_pairs_for_x[x].append((a, b))
    
    def can_add_bitmask(x, S_mask):
        """Check if x can be added to set represented by bitmask S_mask."""
        for (a, b) in bad_pairs_for_x[x]:
            if (S_mask & (1 << a)) and (S_mask & (1 << b)):
                return False
        return True

    def greedy_add(order):
        """Build set by adding elements in given order."""
        S = 0
        for x in order:
            if can_add_bitmask(x, S):
                S |= (1 << x)
        return S

    def local_search_improve(S_mask, time_limit):
        """Try to improve by adding elements and swapping."""
        S = S_mask
        rng = random.Random(int(time.time() * 1000) % 10000)
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try adding elements
            # Randomize order of addition attempts to avoid bias
            candidates = [x for x in range(1, N+1) if not (S & (1 << x))]
            rng.shuffle(candidates)
            
            for x in candidates:
                if can_add_bitmask(x, S):
                    S |= (1 << x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2. Try 1-for-1 swap (remove y, add x) to potentially enable future adds
            # Or 1-for-2 (remove y, add x, add z)
            
            # First, try simple swaps to escape local minima (size stays same)
            # This might not increase size immediately, but can change the set structure.
            # However, we want to maximize size. So 1-for-2 is better.
            
            # Try 1-for-2: Remove y, add x and z
            # Iterate over y in S
            elements_in_S = [i for i in range(1, N+1) if S & (1 << i)]
            rng.shuffle(elements_in_S)
            
            for y in elements_in_S:
                S_temp = S & ~(1 << y)
                
                # Try to add as many as possible from candidates not in S_temp
                # We specifically want to add 2 elements to get net +1
                # Greedily add first compatible, then next compatible
                
                added = 0
                new_S = S_temp
                # To be efficient, just try to add one, then try to add another
                # If we can add 2, we keep the change.
                
                # Collect candidates
                cands = [x for x in range(1, N+1) if not (new_S & (1 << x)) and x != y]
                rng.shuffle(cands)
                
                added_list = []
                for x in cands:
                    if can_add_bitmask(x, new_S):
                        new_S |= (1 << x)
                        added_list.append(x)
                        if len(added_list) >= 2:
                            break
                
                if len(added_list) == 2:
                    S = new_S
                    improved = True
                    break
            
            # If no 1-for-2 found, try 1-for-1 swap with a "better" element?
            # Or just random perturbation: remove a random element, don't add back immediately, 
            # and let the "add" phase try to fill it up.
            if not improved:
                # Random perturbation: remove 1 element
                if elements_in_S:
                    y = rng.choice(elements_in_S)
                    S &= ~(1 << y)
                    # Don't mark as improved, let the loop continue to try adding
                    # But we must ensure we don't get stuck in an infinite loop of removing.
                    # The "add" phase at the start of the next iteration will try to add.
                    # If it can't add, we remove again.
                    # To prevent oscillation, we can track a "temperature" or just accept it.
                    # Given the small N, this should converge.
                    pass

        return S

    best_S = 0
    best_size = 0
    
    # Phase 1: Multiple greedy approaches
    # Order 1: Ascending
    S1 = greedy_add(range(1, N+1))
    size1 = bin(S1).count('1')
    if size1 > best_size:
        best_S = S1
        best_size = size1
        
    # Order 2: Descending
    S2 = greedy_add(range(N, 0, -1))
    size2 = bin(S2).count('1')
    if size2 > best_size:
        best_S = S2
        best_size = size2
        
    # Order 3: By number of conflicts (fewer conflicts first)
    def conflict_count(x):
        return len(bad_pairs_for_x[x])
        
    order_conflicts = sorted(range(1, N+1), key=lambda x: (conflict_count(x), x))
    S3 = greedy_add(order_conflicts)
    size3 = bin(S3).count('1')
    if size3 > best_size:
        best_S = S3
        best_size = size3
        
    # Order 4: Random restarts
    rng = random.Random(42)
    for trial in range(50):
        if time.time() > deadline - 15:
            break
        order = list(range(1, N+1))
        rng.shuffle(order)
        S = greedy_add(order)
        size = bin(S).count('1')
        if size > best_size:
            best_S = S
            best_size = size

    # Phase 2: Local Search on the best found
    # Run local search on the current best
    best_S = local_search_improve(best_S, deadline - 5)
    best_size = bin(best_S).count('1')
    
    # Phase 3: More Local Search Restarts
    # Try to perturb the best solution and re-optimize
    for trial in range(20):
        if time.time() > deadline - 2:
            break
        S_temp = best_S
        # Remove 2 random elements
        elements = [i for i in range(1, N+1) if S_temp & (1 << i)]
        if len(elements) >= 2:
            r1, r2 = rng.sample(elements, 2)
            S_temp &= ~(1 << r1)
            S_temp &= ~(1 << r2)
            
        S_new = local_search_improve(S_temp, min(time.time() + 1.5, deadline - 2))
        size_new = bin(S_new).count('1')
        if size_new > best_size:
            best_S = S_new
            best_size = size_new

    result = [i for i in range(1, N+1) if best_S & (1 << i)]
    return result if result else [1]
# EVOLVE-BLOCK-END
