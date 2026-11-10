import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    n = 12
    P = 27720  # LCM of 1..12
    
    def has_perfect_matching(m, L):
        # candidates[i] contains indices idx (0-based) such that (m + 1 + idx) % i == 0
        candidates = [[] for _ in range(n + 1)]
        
        # Precompute remainders for speed? 
        # (m+1) % i is fixed for a given m, i.
        rems = [0] * (n + 1)
        for i in range(1, n + 1):
            rems[i] = (m + 1) % i
            
        for idx in range(L):
            for i in range(1, n + 1):
                if (rems[i] + idx) % i == 0:
                    candidates[i].append(idx)
        
        for i in range(1, n + 1):
            if not candidates[i]:
                return False
        
        # Sort indices by constraint tightness (fewest candidates first)
        indices = list(range(1, n + 1))
        indices.sort(key=lambda i: len(candidates[i]))
        
        def dfs(pos, used_mask):
            if pos == n:
                return True
            i = indices[pos]
            for b_idx in candidates[i]:
                bit = 1 << b_idx
                if not (used_mask & bit):
                    if dfs(pos + 1, used_mask | bit):
                        return True
            return False
        
        return dfs(0, 0)

    def compute_f(m):
        # Binary search for minimum L
        lo = 1
        hi = 200  # Safe upper bound for n=12
        best_L = hi
        
        # Check if hi is sufficient
        if not has_perfect_matching(m, hi):
            # If not, increase hi, though unlikely
            hi = 1000
            if not has_perfect_matching(m, hi):
                hi = P
                
        while lo <= hi:
            mid = (lo + hi) // 2
            if has_perfect_matching(m, mid):
                best_L = mid
                hi = mid - 1
            else:
                lo = mid + 1
                
        return best_L

    best_m = 0
    best_f = 0
    
    for m in range(P):
        if time.time() > deadline:
            break
            
        f = compute_f(m)
        if f > best_f:
            best_f = f
            best_m = m
            
    return best_m
