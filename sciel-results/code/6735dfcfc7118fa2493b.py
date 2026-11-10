import random
import time
from math import gcd

def construct():
    start_time = time.time()
    # 45 seconds is a safe margin for 60s limit
    deadline = start_time + 45
    
    N = 128
    # Precompute GCD table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    def can_add(S, x):
        """
        Check if x can be added to set S.
        S is a set of integers.
        Returns True if adding x does not create a triple (a,b,x) 
        such that gcd(a,b)==gcd(a,x)==gcd(b,x).
        """
        s_list = list(S)
        n = len(s_list)
        for i in range(n):
            a = s_list[i]
            g_ax = gcd_table[a][x]
            for j in range(i + 1, n):
                b = s_list[j]
                g_ab = gcd_table[a][b]
                if g_ab == g_ax:
                    # Only check g_bx if g_ab == g_ax
                    if g_ab == gcd_table[b][x]:
                        return False
        return True

    def local_search(S):
        """
        Improve set S by adding elements and performing swaps.
        """
        S = set(S)
        improved = True
        # Limit iterations to prevent infinite loops in case of oscillation (unlikely with size-maximizing swaps)
        max_iters = 100
        iters = 0
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # Phase 1: Try to add any element
            # Iterate in random order to break symmetries? 
            # Or just 1..N. Random is better for diversity.
            candidates_add = list(range(1, N + 1))
            random.shuffle(candidates_add)
            
            for x in candidates_add:
                if x not in S:
                    if can_add(S, x):
                        S.add(x)
                        improved = True
                        break 
            
            # Phase 2: If no addition possible, try swaps
            if not improved:
                s_list = list(S)
                random.shuffle(s_list)
                
                swapped = False
                for a in s_list:
                    if swapped:
                        break
                    
                    # Try replacing a with b
                    # Optimization: Pre-calculate S_minus_a? No, just pass S and exclude a in check?
                    # can_add takes a set. Creating S_minus_a is O(k).
                    S_minus_a = S - {a}
                    
                    candidates_swap = list(range(1, N + 1))
                    random.shuffle(candidates_swap)
                    
                    for b in candidates_swap:
                        if b in S:
                            continue
                        
                        if can_add(S_minus_a, b):
                            S.remove(a)
                            S.add(b)
                            swapped = True
                            improved = True
                            break
                
        return S

    best_set = set()
    
    # Run multiple random greedy passes
    # We don't know how long each pass takes, so we loop until deadline
    while time.time() < deadline:
        # Greedy construction
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        current_set = set()
        for x in candidates:
            if can_add(current_set, x):
                current_set.add(x)
        
        # Local search improvement
        improved_set = local_search(current_set)
        
        if len(improved_set) > len(best_set):
            best_set = improved_set

    return list(best_set)
