import time

def construct():
    """
    Finds m in [0, 1000000] maximizing f(12, m).
    f(12, m) is the least L > 0 such that (m, m+L] contains distinct b_1..b_12 with i|b_i.
    """
    N = 12
    PERIOD = 27720  # LCM(1..12)
    MAX_L = 144
    
    # Precompute valid offsets for each i and residue r
    # valid_offsets[i][r] is a list of j in [0, MAX_L-1] such that (r+1+j) % i == 0
    # i ranges 1..12
    valid_offsets = [[None] * PERIOD for _ in range(N + 1)]
    
    for i in range(1, N + 1):
        for r in range(PERIOD):
            # We need j in [0, MAX_L-1] such that (r + 1 + j) % i == 0
            # Let val = r + 1 + j
            # val % i == 0 => j % i == (-r - 1) % i
            target_res = (-r - 1) % i
            offsets = []
            # Find first j >= 0 with j % i == target_res
            if target_res == 0:
                start_j = 0
            else:
                start_j = target_res
            
            for j in range(start_j, MAX_L, i):
                offsets.append(j)
            valid_offsets[i][r] = offsets

    def has_matching(r, L):
        # match_to[j] = i if right node j is matched to left node i, else -1
        match_to = [-1] * L
        
        # Pre-filter candidates for this L
        # cand[i] = list of j in valid_offsets[i][r] such that j < L
        # We can just iterate valid_offsets[i][r] and break when j >= L
        # Or slice if it's a list. Slicing creates a new list, which is slow.
        # Better to pass the full list and L, and check in DFS.
        
        # To optimize DFS, sort left nodes by number of candidates (heuristic)
        # But candidate count depends on L.
        # Let's just use fixed order 1..12 or precomputed static order?
        # Static order might be fine.
        
        def dfs(i, visited):
            # i is 1..N
            # visited is a list of booleans for right nodes 0..L-1
            # Iterate through candidates for i
            # candidates are j in valid_offsets[i][r] where j < L
            
            # Optimization: local variable lookup
            v_offs = valid_offsets[i][r]
            for j in v_offs:
                if j >= L:
                    break # Since offsets are sorted
                if not visited[j]:
                    visited[j] = True
                    if match_to[j] == -1 or dfs(match_to[j], visited):
                        match_to[j] = i
                        return True
            return False

        # Standard Kuhn's algorithm
        # Order of processing left nodes can affect speed.
        # Nodes with fewer options are harder to match, so process them first.
        # However, computing the degree for each L is overhead.
        # Let's try processing 1..12.
        
        for i in range(1, N + 1):
            visited = [False] * L
            if not dfs(i, visited):
                return False
        return True

    best_m = 0
    max_L_found = 0
    
    start_time = time.time()
    deadline = start_time + 85.0 # 5s margin
    
    for r in range(PERIOD):
        if time.time() > deadline:
            break
            
        # Binary search for minimal L
        # We know L >= 12 (pigeonhole, need 12 distinct numbers)
        # We know L <= 144
        lo, hi = 12, MAX_L
        # Find minimum L in [lo, hi] where matching exists
        # Predicate: has_matching(r, L) is monotone non-decreasing in L
        # If has_matching(r, L) is True, then has_matching(r, L+1) is True.
        # We want the smallest L such that it is True.
        
        # Check if even MAX_L works?
        # If not, then this r is invalid? But problem says max is 144, implying it always works for some L <= 144?
        # "The raw value is f(12, m), which is at most 144."
        # So has_matching(r, 144) should always be True.
        
        while lo < hi:
            mid = (lo + hi) // 2
            if has_matching(r, mid):
                hi = mid
            else:
                lo = mid + 1
        
        if lo <= MAX_L:
            L_min = lo
            if L_min > max_L_found:
                max_L_found = L_min
                best_m = r
                if max_L_found == MAX_L:
                    return best_m

    return best_m
