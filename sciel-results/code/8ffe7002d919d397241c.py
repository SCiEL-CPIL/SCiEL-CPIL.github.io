import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s margin
    
    n = 12
    max_m = 1000000
    
    # Function to check if a perfect matching exists for a given m and L
    def has_matching(m, L):
        # Generate sets S_i for i in 1..n
        # S_i = {x in (m, m+L] | i divides x}
        # We build a bipartite graph and check for perfect matching
        # Left side: 1..n
        # Right side: values in (m, m+L]
        
        # To speed up, we can use DFS for bipartite matching (Kuhn's algorithm)
        # Right nodes are integers. We map them to indices 0..L-1 for the matching array?
        # Or just use the values directly in a dict/set?
        # Since L is small (<= 144), we can just list the values.
        
        # Optimization: If the union of all S_i has size < n, no matching.
        union_vals = set()
        for i in range(1, n + 1):
            # First multiple of i greater than m
            start = m + 1
            # Adjust to next multiple
            rem = start % i
            if rem == 0:
                first = start
            else:
                first = start + (i - rem)
            
            # Add multiples
            x = first
            while x <= m + L:
                union_vals.add(x)
                x += i
        
        if len(union_vals) < n:
            return False
            
        # Kuhn's Algorithm
        # match_to[y] = x, where x is matched to y
        # We iterate x in 1..n and try to find a free y
        # But y is not contiguous 1..M, it's specific values.
        # We'll use a dictionary for matching: match_val[y] = x
        
        match_val = {}
        
        def dfs(x, visited):
            for y in range(m + 1, m + L + 1):
                if x % 1 == 0 and y % x == 0 and y not in visited:
                    visited.add(y)
                    if y not in match_val or dfs(match_val[y], visited):
                        match_val[y] = x
                        return True
            return False
        
        # Wait, the edge condition is i divides b_i.
        # So for a given i (left node), it connects to y if y % i == 0.
        # My dfs above checks y % x == 0 where x is the index. Correct.
        
        # Optimization: Only iterate y if y % i == 0.
        # We can precompute candidates for each i.
        candidates = {}
        for i in range(1, n + 1):
            cands = []
            rem = (m + 1) % i
            if rem == 0:
                first = m + 1
            else:
                first = m + 1 + (i - rem)
            x = first
            while x <= m + L:
                cands.append(x)
                x += i
            candidates[i] = cands
            
        # Check Hall's condition? Or just run matching.
        # Matching is O(VE) which is small.
        
        match_val = {}
        
        def dfs(i, visited):
            for y in candidates[i]:
                if y not in visited:
                    visited.add(y)
                    if y not in match_val or dfs(match_val[y], visited):
                        match_val[y] = i
                        return True
            return False
        
        for i in range(1, n + 1):
            visited = set()
            if not dfs(i, visited):
                return False
        return True

    def calc_L(m):
        # Binary search or linear search?
        # L is small. Linear is fine.
        # Max L is 144.
        L = 1
        while L <= 150:
            if has_matching(m, L):
                return L
            L += 1
        return 150 # Should not happen

    # Generate candidates
    candidates_m = set()
    
    # Add small m
    for m in range(0, 100):
        candidates_m.add(m)
        
    # Add m based on LCMs of subsets
    # Subsets of {1, ..., 12}
    # We want subsets with large LCMs to create "bottlenecks"
    # Let's just use LCMs of sets like {7, 11, 12}, {5, 7, 11, 12}, etc.
    
    import math
    
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    def lcm_list(lst):
        result = 1
        for x in lst:
            result = lcm(result, x)
        return result
        
    # Key subsets
    key_subsets = [
        [11, 12],
        [7, 11, 12],
        [5, 7, 11, 12],
        [4, 7, 11, 12],
        [3, 7, 11, 12],
        [2, 7, 11, 12],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [5, 6, 7, 8, 9, 10, 11, 12],
        [7, 8, 9, 10, 11, 12]
    ]
    
    for sub in key_subsets:
        K = lcm_list(sub)
        if K > max_m:
            continue
        # m = K*q - r
        # q from 1 to max_m // K
        # r from 1 to 50 (small offset)
        max_q = max_m // K
        for q in range(1, max_q + 1):
            base = K * q
            for r in range(1, 51):
                m = base - r
                if 0 <= m <= max_m:
                    candidates_m.add(m)
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # Also add some random or structured probes
    # e.g. m = 27720*q - r
    K_full = 27720
    max_q = max_m // K_full
    for q in range(1, max_q + 1):
        base = K_full * q
        for r in range(1, 101):
            m = base - r
            if 0 <= m <= max_m:
                candidates_m.add(m)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # Evaluate candidates
    best_m = 0
    best_L = 0
    
    # Sort candidates? Maybe evaluate promising ones first.
    # Heuristic: m closer to a multiple of large LCM is more promising.
    # We'll just iterate.
    
    for m in candidates_m:
        if time.time() > deadline:
            break
        L = calc_L(m)
        if L > best_L:
            best_L = L
            best_m = m
            
    # If we have time, do a local search around best_m
    if time.time() < deadline - 5:
        current_best_L = best_L
        # Check neighbors of best_m
        for delta in range(-100, 101):
            m_new = best_m + delta
            if 0 <= m_new <= max_m:
                if time.time() > deadline:
                    break
                L_new = calc_L(m_new)
                if L_new > current_best_L:
                    current_best_L = L_new
                    best_m = m_new
        best_L = current_best_L

    return int(best_m)

construct()
