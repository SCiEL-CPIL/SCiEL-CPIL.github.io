import time

def construct():
    start_time = time.time()
    limit = 85  # seconds, safety margin
    
    N = 12
    period = 27720
    
    best_m = 0
    best_f = 0
    
    def can_match(m, L):
        """Check if we can match 12 distinct values in (m, m+L] where i divides b_i."""
        N_local = 12
        
        # Build candidate lists and sort by scarcity
        cands_info = []
        for i in range(1, N_local + 1):
            r = (-m) % i
            if r == 0:
                first_d = i
            else:
                first_d = r
            if first_d > L:
                return False
            lst = list(range(first_d, L + 1, i))
            cands_info.append((len(lst), lst, i))
        
        # Sort by number of candidates (most constrained first)
        cands_info.sort(key=lambda x: x[0])
        
        # Quick Hall check: for the k most constrained, union must be >= k
        # This is expensive, skip for now
        
        # DFS with array-based used tracking for speed
        used = [False] * (L + 1)
        
        def dfs(k):
            if k == N_local:
                return True
            _, lst, _ = cands_info[k]
            for d in lst:
                if not used[d]:
                    used[d] = True
                    if dfs(k + 1):
                        return True
                    used[d] = False
            return False
        
        return dfs(0)

    def compute_f(m):
        """Find the minimum L >= 1 such that matching exists in (m, m+L]."""
        # L must be at least 12 (12 distinct values needed)
        # Maximum relevant L is around 144
        # Use linear search from 1 upward (or from 12 for speed)
        # Since we want to MAXIMIZE f over m, and f is the MINIMUM L that works,
        # a larger f means the matching is "harder" to achieve
        
        # Start from L=1 for correctness, but we know L>=12
        # Actually, let's start from 1 to be safe
        for L in range(1, 150):
            if can_match(m, L):
                return L
        return 149

    # Search m in one period
    # Check time periodically
    check_interval = 200
    
    for m in range(period):
        if m % check_interval == 0:
            if time.time() - start_time > limit:
                break
        
        f_val = compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
        # Early termination: if best_f is already very high, we can stop
        # The theoretical max is around 144
        if best_f >= 140:
            # Still check remaining m's but with tighter time limit
            pass

    return best_m
