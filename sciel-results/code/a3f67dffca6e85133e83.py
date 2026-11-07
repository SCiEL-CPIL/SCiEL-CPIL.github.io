"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def _can_match(m, L, n=12):
    """
    Check if (m, m+L] contains distinct b_1...b_n with i | b_i.
    Uses bitmask DFS with most-constrained-first ordering.
    """
    options = []
    for i in range(1, n + 1):
        mask = 0
        # We need v in (m, m+L] such that v % i == 0
        # v = m + 1 + j for j in 0..L-1
        # (m + 1 + j) % i == 0  =>  j % i == (-(m+1)) % i
        target = (-(m + 1)) % i
        for j in range(L):
            if j % i == target:
                mask |= (1 << j)
        if mask == 0:
            return False
        options.append((mask.bit_count(), mask))
    
    # Sort by number of options (most constrained first)
    options.sort(key=lambda x: x[0])
    
    n_opts = n
    
    def dfs(k, used_mask):
        if k == n_opts:
            return True
        mask = options[k][1]
        available = mask & ~used_mask
        while available:
            bit = available & (-available)
            available -= bit
            if dfs(k + 1, used_mask | bit):
                return True
        return False
    
    return dfs(0, 0)

def _f_val(m, n=12):
    """
    Find the least L such that (m, m+L] allows a matching.
    Binary search on monotone property.
    """
    lo = n
    hi = 144
    
    # Check if even the max window works (should always be true for reasonable L)
    # But just in case, if it doesn't, we return hi or higher? 
    # The problem states L is at most 144.
    if not _can_match(m, hi, n):
        # If 144 is not enough, something is wrong or L is larger.
        # But task says L is at most 144.
        return hi
    
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    best_m = 0
    best_L = 0
    
    # The divisibility pattern repeats every LCM(1..12) = 27720.
    # f(12, m) depends only on m mod 27720.
    # So we only need to search m in [0, 27720).
    period = 27720
    
    # We can optimize the search by noting that for a given L,
    # if a match exists, it exists for any larger L.
    # We want to maximize f(12, m).
    
    # Let's just iterate through the period.
    # 27720 iterations * binary search (log2(144) ~ 8 steps) * DFS (fast)
    # This should be very fast.
    
    for m in range(period):
        if time.time() > deadline:
            break
        
        L = _f_val(m)
        if L > best_L:
            best_L = L
            best_m = m
    
    # Ensure best_m is within [0, 1000000]
    # Since period is 27720, best_m is already in range.
    
    return int(best_m)
# EVOLVE-BLOCK-END
