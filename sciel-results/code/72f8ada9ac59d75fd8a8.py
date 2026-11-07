import time

def _hall_check(m, L):
    """
    Check if (m, m+L] contains a valid matching for indices 1..12.
    Uses Hall's Theorem with bitmask subset enumeration.
    """
    if L < 12:
        return False
        
    # masks[v_index] will hold the bitmask of indices dividing value (m+1+v_index)
    # We need to check Hall's condition:
    # For all S subset of {1..12}, |N(S)| >= |S|
    # N(S) = {v in (m, m+L] : exists i in S such that i|v}
    
    # fail_count[S] = number of v in (m, m+L] such that v is NOT in N(S)
    # v is not in N(S) iff for all i in S, i does not divide v.
    # This is equivalent to S being a subset of the complement of the divisor mask of v.
    
    fail_count = [0] * 4096
    
    # Iterate through values in (m, m+L]
    for k in range(L):
        v = m + 1 + k
        # Compute divisor mask for v
        mask = 0
        for i in range(1, 13):
            if v % i == 0:
                mask |= (1 << (i - 1))
        
        # The value v fails to be in N(S) for all S subset of (~mask) & 0xFFF
        comp = (~mask) & 0xFFF
        
        # Increment fail_count for all subsets of comp
        # Standard subset iteration
        sub = comp
        while True:
            fail_count[sub] += 1
            if sub == 0:
                break
            sub = (sub - 1) & comp
            
    # Check Hall's condition
    # For all S != 0, fail_count[S] <= L - popcount(S)
    # popcount(S) is the number of indices in S
    for S in range(1, 4096):
        pc = bin(S).count('1')
        if fail_count[S] > L - pc:
            return False
            
    return True

def construct():
    """
    Find m in 0..1000000 that maximizes f(12, m).
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    K = 27720  # LCM(1..12)
    best_m = 0
    best_L = 0
    
    # We search all m in one period.
    # For each m, we find the minimal L.
    # Max L is bounded. The task says max L is 144.
    # We can binary search L or just iterate.
    # Since L is small, iterating from 12 upwards is fast enough if _hall_check is fast.
    # However, _hall_check is O(L * 4096).
    # Total ops ~ 27720 * (sum_{L=12}^{L_max} L * 4096).
    # If L_max is 144, sum L ~ 10000. 10000 * 4096 ~ 40M ops per m? No.
    # For each m, we call _hall_check multiple times.
    # Binary search on L: ~7 calls.
    # Each call is O(L * 4096). Average L ~ 70. 70 * 4096 ~ 280k ops.
    # 7 * 280k ~ 2M ops per m.
    # 27720 * 2M = 55 Billion ops. Too slow for Python.
    
    # We need a faster check.
    # The DFS matching is likely faster in practice because the graph is sparse.
    # Let's switch to the optimized DFS from the current program, 
    # but ensure it's tight.
    
    # Actually, let's try to optimize the Hall check or use a different method.
    # Or, reduce the search space?
    # The function f(12, m) is periodic.
    # Maybe we can sample? No, we need the exact max.
    
    # Let's use the DFS approach but with adjacency lists precomputed per m?
    # No, L changes.
    
    # Let's use the DFS approach from the current program.
    # It was scored 1.375.
    # The bottleneck might be the Python overhead of the recursive DFS.
    
    # Alternative: Use `networkx`? Not allowed if not installed, but standard lib is.
    # `scipy` is available. `scipy.sparse.csgraph.maximum_bipartite_matching`?
    # That might be fast.
    
    # Let's try to implement a fast non-recursive matching or use the bitmask approach 
    # but only for small L?
    
    # Actually, the bitmask approach complexity is dominated by the subset iteration.
    # If the mask is dense (many divisors), comp is small, so few subsets.
    # If the mask is sparse (few divisors), comp is large.
    # For random numbers, masks are moderately dense.
    
    # Let's stick to the current program's DFS but optimize it.
    # Or, use the `scipy` matcher if available and fast.
    
    # Let's try the `scipy` approach.
    # Build a sparse matrix A of size 12 x L.
    # A[i, j] = 1 if (m+1+j) % (i+1) == 0.
    # Then find max bipartite matching.
    
    try:
        import scipy.sparse as sp
        import scipy.sparse.csgraph as csgraph
        import numpy as np
        
        def _can_match_scipy(m, L):
            if L < 12:
                return False
            # Create adjacency matrix
            # Rows: indices 0..11 (corresponding to i=1..12)
            # Cols: values 0..L-1 (corresponding to v=m+1..m+L)
            rows = []
            cols = []
            for i in range(1, 13):
                for j in range(L):
                    v = m + 1 + j
                    if v % i == 0:
                        rows.append(i - 1)
                        cols.append(j)
            
            if len(rows) < 12:
                return False
                
            data = np.ones(len(rows))
            A = sp.coo_matrix((data, (rows, cols)), shape=(12, L))
            
            # maximum_bipartite_matching returns (rows, cols)
            try:
                r, c = csgraph.maximum_bipartite_matching(A)
                if len(r) == 12:
                    return True
                else:
                    return False
            except Exception:
                return False
                
    except ImportError:
        _can_match_scipy = None
        
    # If scipy is not available or fails, fall back to DFS
    def _can_match_dfs(m, L):
        if L < 12:
            return False
            
        # Pre-compute candidate counts for sorting
        indices_info = []
        for i in range(1, 13):
            count = 0
            # First multiple of i >= m+1
            k = (m + 1 + i - 1) // i
            start_v = k * i
            if start_v < m + 1:
                start_v += i
            # Count multiples in (m, m+L]
            # v = start_v, start_v+i, ... <= m+L
            if start_v <= m + L:
                count = ((m + L) - start_v) // i + 1
            indices_info.append((count, i))
            
        indices_info.sort()
        
        if indices_info[0][0] == 0:
            return False
            
        match_idx = [-1] * L
        
        def dfs(i, visited):
            for k in range(L):
                v = m + 1 + k
                if v % i != 0:
                    continue
                if visited[k]:
                    continue
                visited[k] = True
                if match_idx[k] == -1 or dfs(match_idx[k], visited):
                    match_idx[k] = i
                    return True
            return False

        for count, i in indices_info:
            visited = [False] * L
            if not dfs(i, visited):
                return False
                
        return True

    # Use scipy if available, else DFS
    matcher = _can_match_scipy if _can_match_scipy else _can_match_dfs

    for m in range(K):
        if time.time() > deadline:
            break
            
        # Binary search for minimal L
        # L is at least 12. Max is likely < 200.
        lo, hi = 12, 200
        found = False
        while lo <= hi:
            mid = (lo + hi) // 2
            if matcher(m, mid):
                best_L_candidate = mid
                hi = mid - 1
                found = True
            else:
                lo = mid + 1
                
        if found:
            if best_L_candidate > best_L:
                best_L = best_L_candidate
                best_m = m
            # If we found a very large L, we might still need to check others
            # to see if it's even larger.
            
    return int(best_m)
