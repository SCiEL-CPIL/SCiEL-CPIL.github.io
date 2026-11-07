import time
import sys

# EVOLVE-BLOCK-START

def construct():
    """
    Finds the integer m in [0, 1000000] that maximizes f(12, m).
    f(12, m) is the least positive L such that (m, m+L] contains distinct 
    b_1, ..., b_12 with i dividing b_i.
    """
    N = 12
    PERIOD = 27720  # LCM(1, ..., 12)
    
    # Deadline guard: 90s limit, leave 10s margin
    deadline = time.time() + 80.0
    
    def has_matching(cands):
        """
        Checks if there is a perfect matching for indices 0..11 
        to distinct values in the candidate lists.
        cands is a list of 12 lists of integers (offsets 1..W).
        """
        # Sort indices by number of candidates to prune early
        # We need to keep track of original indices? 
        # The matching just needs to assign distinct values.
        # The left nodes are 0..11.
        # Let's reorder the left nodes for the DFS.
        order = sorted(range(N), key=lambda i: len(cands[i]))
        
        # match_val[val] = left_index
        # We use a dict or list. Since values are 1..W (W<=144), list is faster.
        # Initialize match array with -1
        # Max W is 144.
        max_val = 0
        for cl in cands:
            if cl:
                max_val = max(max_val, max(cl))
        match = [-1] * (max_val + 1)
        
        # To speed up, we can use a visited array with a timestamp or a set.
        # Since DFS is recursive, a set is standard.
        # But we can use a list of size max_val+1 and a 'seen' flag with a counter.
        
        def dfs(i, visited):
            for v in cands[order[i]]:
                if visited[v]:
                    continue
                visited[v] = True
                if match[v] == -1 or dfs(match[v], visited):
                    match[v] = order[i]
                    return True
            return False
        
        # We need to map the sorted order back to original?
        # No, `match[v]` stores the original index?
        # In `dfs(i, ...)`, `i` is the position in `order`.
        # `cands[order[i]]` are the candidates for that node.
        # `match[v]` should store the node index that is currently matched to v.
        # If we store `order[i]` in `match[v]`, then `dfs(match[v], ...)` will use the position in `order`.
        # This is consistent.
        
        for i in range(N):
            visited = [False] * (max_val + 1)
            if not dfs(i, visited):
                return False
        return True

    def check_K(K):
        """
        Returns True if there exists m such that f(m) >= K.
        f(m) >= K iff no matching exists for window size W = K-1.
        Also returns the m if found, else None.
        """
        W = K - 1
        if W == 0:
            return True, 0
        
        # Precompute candidates for each i and residue r
        # cands_res[i][r] = list of offsets j in 1..W such that (r+j) % i == 0
        cands_res = [[None] * i for i in range(N + 1)]
        for i in range(1, N + 1):
            for r in range(i):
                lst = []
                for j in range(1, W + 1):
                    if (r + j) % i == 0:
                        lst.append(j)
                cands_res[i][r] = lst
        
        # Iterate m
        for m in range(PERIOD):
            if time.time() > deadline:
                return False, None # Timeout, assume False to be safe? Or break?
                # Better to break outer loop in construct, but here we return False.
            
            # Build cands for this m
            cands = []
            for i in range(1, N + 1):
                r = m % i
                cands.append(cands_res[i][r])
            
            # Quick check: if any list is empty, no matching
            # Also, if sum of lengths < N, no matching? No, distinct values.
            # Just run matching.
            
            if not has_matching(cands):
                return True, m
        
        return False, None

    # Binary search for max K
    # P(K) is True for K=1..K_max, False for K > K_max
    low, high = 1, 145
    best_m = 0
    
    # We know P(1) is True (W=0, no match)
    while low < high:
        mid = (low + high + 1) // 2
        exists, m_found = check_K(mid)
        if exists:
            low = mid
            best_m = m_found
        else:
            high = mid - 1
            
    # The max K is low.
    # best_m is the m found for this K (or a smaller one if we didn't update in the last step? 
    # We updated best_m whenever exists was True.
    # Since the search converges to the largest K, and we update best_m when True,
    # best_m corresponds to the current low which is the max K.
    # However, if the last check was False, we didn't update.
    # But we updated for all True cases. The last True case was for `low`.
    # So best_m is valid for `low`.
    
    return int(best_m)

# EVOLVE-BLOCK-END
