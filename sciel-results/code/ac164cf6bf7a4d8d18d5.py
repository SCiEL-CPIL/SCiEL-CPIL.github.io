def construct():
    import time
    from math import gcd
    import numpy as np
    from itertools import combinations
    
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    N = 512
    nodes = list(range(1, N + 1))
    
    # Precompute compatibility matrix using numpy for speed
    # compat[i, j] = 1 if lcm(i+1, j+1) <= 512
    def build_compatibility():
        # Use vectorized gcd via numpy
        arr = np.arange(1, N + 1, dtype=np.int64)
        # Build matrix: lcm(i,j) = i*j//gcd(i,j)
        # For memory efficiency, compute row by row
        compat = np.zeros((N, N), dtype=np.bool_)
        for i in range(N):
            a = arr[i]
            g = np.gcd(a, arr)
            lcms = (a * arr) // g
            compat[i, :] = (lcms <= N)
        return compat
    
    compat = build_compatibility()
    
    # Build neighbor sets for Bron-Kerbosch
    neighbors = [set() for _ in range(N)]
    for i in range(N):
        # Find all j where compat[i,j] is True
        mask = compat[i].copy()
        mask[i] = False  # No self-loops
        neighbors[i] = set(np.where(mask)[0].tolist())
    
    # Helper: check if set is valid (all pairs have lcm <= 512)
    def is_valid_set(s):
        s_list = list(s)
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                a, b = s_list[i] + 1, s_list[j] + 1
                if a * b // gcd(a, b) > N:
                    return False
        return True
    
    # Greedy expansion: repeatedly add compatible elements
    def greedy_expand(s):
        s = set(s)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Try to add each candidate
            for x in range(N):
                if x in s:
                    continue
                # Check compatibility with all in s
                can_add = True
                for y in s:
                    if not compat[x, y]:
                        can_add = False
                        break
                if can_add:
                    s.add(x)
                    changed = True
                    break
        return s
    
    # Bron-Kerbosch with pivot for maximum clique
    def bron_kerbosch_max_clique():
        best_clique = set()
        best_size = 0
        
        # Initialize with a good seed: divisors of highly composite number
        # 504 = 2^3 * 3^2 * 7 has 32 divisors
        divs_504 = set()
        for k in range(1, N + 1):
            if 504 % (k) == 0:
                divs_504.add(k - 1)  # 0-indexed
        if len(divs_504) > best_size:
            best_clique = set(divs_504)
            best_size = len(divs_504)
        
        # Also try divisors of other highly composite numbers
        for hcn in [360, 420, 480, 240, 180, 120, 72, 48, 36, 24, 12, 8, 6, 4, 2, 1]:
            divs = set()
            for k in range(1, N + 1):
                if hcn % k == 0:
                    divs.add(k - 1)
            if len(divs) > best_size:
                best_clique = set(divs)
                best_size = len(divs)
        
        # Bron-Kerbosch with pivot
        def bk(R, P, X):
            nonlocal best_clique, best_size
            
            if time.time() > deadline:
                return
            
            if not P and not X:
                if len(R) > best_size:
                    best_clique = set(R)
                    best_size = len(R)
                return
            
            if not P:
                return
            
            # Upper bound: even if we add all of P, can we beat best?
            if len(R) + len(P) <= best_size:
                return
            
            # Pivot selection: choose vertex in P ∪ X with max degree in P
            all_candidates = P | X
            if not all_candidates:
                return
            
            pivot = None
            max_count = -1
            for u in all_candidates:
                count = len(P & neighbors[u])
                if count > max_count:
                    max_count = count
                    pivot = u
            
            if pivot is None:
                return
            
            # Only need to process vertices in P that are NOT neighbors of pivot
            candidates = P - neighbors[pivot]
            
            for v in list(candidates):
                if time.time() > deadline:
                    return
                
                new_R = R | {v}
                new_P = P & neighbors[v]
                new_X = X & neighbors[v]
                
                # Prune: if even adding all new_P can't beat best, skip
                if len(new_R) + len(new_P) <= best_size:
                    continue
                
                bk(new_R, new_P, new_X)
                
                P = P - {v}
                X = X | {v}
        
        # Run BK starting from empty set with all vertices
        # But this is too slow for 512 nodes. Instead, use a smarter approach:
        # Start from the best seed and expand
        
        # First, greedily expand the best seed
        expanded_seed = greedy_expand(set(best_clique))
        if len(expanded_seed) > best_size:
            best_clique = expanded_seed
            best_size = best_size  # Already updated by greedy_expand? No, let's track
            best_size = len(expanded_seed)
        
        # Now try BK from the expanded seed's complement to find improvements
        # Actually, let's do a different approach: use BK to find max clique in the graph
        # But limit the search space
        
        # For now, return the best we have from greedy expansion
        return best_clique
    
    # Local search: try removing one element and adding multiple new ones
    def local_search(s):
        s = set(s)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for remove_candidate in list(s):
                if time.time() > deadline:
                    break
                
                trial_set = s - {remove_candidate}
                original_size = len(trial_set)
                
                # Try to greedily add elements back
                expanded = greedy_expand(set(trial_set))
                
                if len(expanded) > len(s):
                    s = expanded
                    improved = True
                    break
        return s
    
    # Strategy 1: Greedy from divisor seeds
    best_set = set()
    best_size = 0
    
    # Try multiple highly composite number divisor sets as seeds
    hcn_candidates = [504, 360, 420, 480, 240, 180, 120, 72, 48, 36, 24, 12, 8, 6, 4, 2, 1,
                      512, 256, 128, 64, 32, 16]
    
    for hcn in hcn_candidates:
        if time.time() > deadline:
            break
        
        seed = set()
        for k in range(1, N + 1):
            if hcn % k == 0:
                seed.add(k - 1)  # 0-indexed
        
        expanded = greedy_expand(set(seed))
        if len(expanded) > best_size:
            best_set = expanded
            best_size = len(expanded)
    
    # Strategy 2: Local search on the best set
    if time.time() < deadline:
        improved_set = local_search(set(best_set))
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(best_set)
    
    # Strategy 3: Random restarts with greedy expansion
    import random
    random.seed(42)
    num_restarts = 30
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Build a random valid set by adding small numbers probabilistically
        random_set = set()
        for x in range(64):  # Focus on small numbers first
            if random.random() < 0.4:
                # Check if x can be added
                can_add = True
                for y in random_set:
                    if not compat[x, y]:
                        can_add = False
                        break
                if can_add:
                    random_set.add(x)
        
        expanded = greedy_expand(random_set)
        if len(expanded) > best_size:
            best_set = expanded
            best_size = len(expanded)
    
    # Strategy 4: One more local search pass
    if time.time() < deadline:
        final_set = local_search(set(best_set))
        if len(final_set) > best_size:
            best_set = final_set
            best_size = len(best_set)
    
    # Final greedy expansion
    if time.time() < deadline:
        final_expanded = greedy_expand(set(best_set))
        if len(final_expanded) > best_size:
            best_set = final_expanded
            best_size = len(best_set)
    
    # Convert 0-indexed to 1-indexed
    result = sorted([x + 1 for x in best_set])
    
    # Ensure nonempty
    if not result:
        result = [1]
    
    return result
