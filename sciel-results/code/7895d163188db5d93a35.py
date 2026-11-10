# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s
    
    n = 12
    lcm_1_12 = 27720  # lcm(1..12)
    
    def can_assign(m, L):
        """Check if we can assign distinct b_i in (m, m+L] with i | b_i."""
        # For each i, the candidates are multiples of i in (m, m+L]
        # Number of candidates for i is floor((m+L)/i) - floor(m/i)
        
        # Quick feasibility: each i must have at least 1 candidate
        # And total distinct values needed is 12, window size is L
        # So L >= 12 is necessary (but not sufficient)
        
        cands = [None] * n
        
        for i in range(1, n + 1):
            first_mult = (m // i + 1) * i
            end_val = m + L
            if first_mult > end_val:
                return False
            cands[i - 1] = list(range(first_mult, end_val + 1, i))
        
        # Kuhn's algorithm
        match_right = {}
        
        def dfs(left, visited):
            for v in cands[left]:
                if v in visited:
                    continue
                visited.add(v)
                if v not in match_right or dfs(match_right[v], visited):
                    match_right[v] = left
                    return True
            return False
        
        for left in range(n):
            visited = set()
            if not dfs(left, visited):
                return False
        return True
    
    def f_binary(m):
        """Compute f(12, m) using binary search on L."""
        lo, hi = 1, 144
        while lo < hi:
            mid = (lo + hi) // 2
            if can_assign(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo
    
    # Two-stage search:
    # Stage 1: coarse scan every k-th residue
    # Stage 2: fine scan around best regions
    
    best_m = 0
    best_f = 0
    
    # Stage 1: coarse scan
    step = 10
    for m in range(0, lcm_1_12, step):
        if time.time() > deadline - 10.0:
            break
        val = f_binary(m)
        if val > best_f:
            best_f = val
            best_m = m
            if best_f == 144:
                break
    
    # Stage 2: fine scan around the best region
    if best_f < 144:
        # Scan a window around the best found
        window = step * 3
        start = max(0, best_m - window)
        end = min(lcm_1_12, best_m + window)
        
        for m in range(start, end):
            if time.time() > deadline - 3.0:
                break
            if m == best_m:
                continue
            val = f_binary(m)
            if val > best_f:
                best_f = val
                best_m = m
                if best_f == 144:
                    break
    
    # Stage 3: if still not 144, do a full scan with remaining time
    if best_f < 144 and time.time() < deadline - 5.0:
        for m in range(0, lcm_1_12):
            if time.time() > deadline - 2.0:
                break
            if m == best_m:
                continue
            val = f_binary(m)
            if val > best_f:
                best_f = val
                best_m = m
                if best_f == 144:
                    break
    
    return int(best_m)
# EVOLVE-BLOCK-END
