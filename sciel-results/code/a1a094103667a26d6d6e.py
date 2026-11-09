import time
import math

def _has_perfect_matching(m, L, n=12):
    """Check if (m, m+L] contains distinct b_1,...,b_n with i | b_i."""
    # For each i in 1..n, find which positions j in 1..L have (m+j) % i == 0
    # Then check if there's a perfect matching
    # adj[i] = bitmask of positions j where (m+j) is divisible by i
    adj = [0] * (n + 1)
    for i in range(1, n + 1):
        mask = 0
        for j in range(1, L + 1):
            if (m + j) % i == 0:
                mask |= (1 << (j - 1))
        adj[i] = mask
        if mask == 0:
            return False
    
    # Hopcroft-Karp style or simple DFS matching
    # match_r[j] = which i is matched to position j
    match_r = [0] * L
    
    def dfs(i, visited):
        candidates = adj[i] & ~visited
        while candidates:
            low_bit = candidates & -candidates
            j_idx = low_bit.bit_length() - 1
            candidates ^= low_bit
            if match_r[j_idx] == 0 or dfs(match_r[j_idx], visited | low_bit):
                match_r[j_idx] = i
                return True
        return False
    
    for i in range(1, n + 1):
        if not dfs(i, 0):
            return False
    return True

def _compute_f(m, n=12, max_L=144):
    """Compute f(n, m): least positive L such that (m, m+L] has distinct b_i with i|b_i."""
    # Binary search on L
    lo, hi = 1, max_L
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_perfect_matching(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    n = 12
    max_m = 1000000
    
    best_m = 0
    best_f = 0
    
    # Strategy: sample m values, focusing on regions likely to have high f
    # f(12,m) is bounded by ~144. The period structure is related to LCM(1..12)=27720
    # Let's scan a good range of m values
    
    # First, do a coarse scan
    step = 100
    m = 0
    while m <= max_m and time.time() < deadline - 5:
        f_val = _compute_f(m, n)
        if f_val > best_f:
            best_f = f_val
            best_m = m
        m += step
    
    # Now refine around the best region found
    if time.time() < deadline - 3:
        # Check around best_m with finer steps
        lo_bound = max(0, best_m - step * 5)
        hi_bound = min(max_m, best_m + step * 5)
        m = lo_bound
        while m <= hi_bound and time.time() < deadline - 2:
            if m % 10 == 0:  # skip already checked
                f_val = _compute_f(m, n)
                if f_val > best_f:
                    best_f = f_val
                    best_m = m
            m += 1
    
    # Also check some structured candidates: m = k*27720 + offset
    if time.time() < deadline - 1:
        for k in range(0, 37):  # 37 * 27720 < 1000000
            base = k * 27720
            for offset in range(0, 27720, 50):
                m = base + offset
                if m > max_m:
                    break
                f_val = _compute_f(m, n)
                if f_val > best_f:
                    best_f = f_val
                    best_m = m
    
    return int(best_m)
