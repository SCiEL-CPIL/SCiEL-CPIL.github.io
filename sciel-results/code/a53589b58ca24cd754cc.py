"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

N = 12
MAX_START = 1000000

def _can_match(m, L):
    """Check if (m, m+L] contains distinct b_1..b_N with i | b_i."""
    n = N
    # Build adjacency: for each i (1-indexed), list of candidate values in (m, m+L]
    adj = [[] for _ in range(n)]
    for i in range(1, n + 1):
        first = ((m // i) + 1) * i
        end = m + L
        v = first
        while v <= end:
            adj[i - 1].append(v)
            v += i
    
    # If any i has no candidates, return False
    for i in range(n):
        if not adj[i]:
            return False
    
    # Kuhn's algorithm for bipartite matching
    # Left side: indices 0..n-1 (representing i=1..n)
    # Right side: values (multiples)
    match_val = {}  # value -> left index assigned
    
    def dfs(i, visited):
        for v in adj[i]:
            if v in visited:
                continue
            visited.add(v)
            if v not in match_val or dfs(match_val[v], visited):
                match_val[v] = i
                return True
        return False
    
    for i in range(n):
        visited = set()
        if not dfs(i, visited):
            return False
    
    return True

def _f_val(m):
    """Compute f(N, m): least positive L such that (m, m+L] has distinct b_1..b_N with i|b_i."""
    # Lower bound: at least N (need N distinct values)
    # Upper bound: try increasing L
    # The maximum L is bounded; for N=12, L is at most ~144 per task description
    lo = 1
    hi = 200  # safe upper bound
    
    # First check if hi is feasible
    if not _can_match(m, hi):
        # Try larger
        hi = 500
        if not _can_match(m, hi):
            hi = 1000
            if not _can_match(m, hi):
                return 0  # shouldn't happen
    
    # Binary search for minimum L
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid):
            hi = mid
        else:
            lo = mid + 1
    
    return lo

def construct():
    """Find m in [0, MAX_START] that maximizes f(12, m)."""
    start_time = time.time()
    deadline = start_time + 85  # 85s out of 90s budget
    
    best_m = 0
    best_f = 0
    
    def try_m(m):
        nonlocal best_m, best_f
        if m < 0 or m > MAX_START:
            return
        if time.time() > deadline:
            return
        f = _f_val(m)
        if f > best_f:
            best_f = f
            best_m = m
    
    # First pass: coarse sampling with multiple strides to cover different patterns
    # Use several stride sizes to catch different periodic structures
    strides = [100, 60, 30, 12, 6]
    for step in strides:
        if time.time() > deadline:
            break
        for m in range(0, MAX_START + 1, step):
            if time.time() > deadline:
                break
            try_m(m)
    
    # Second pass: fine sampling around the best region with larger window
    if best_f > 0:
        window = 2000
        lo_m = max(0, best_m - window)
        hi_m = min(MAX_START, best_m + window)
        for m in range(lo_m, hi_m + 1):
            if time.time() > deadline:
                break
            try_m(m)
    
    # Third pass: even finer around the best
    if best_f > 0:
        window = 100
        lo_m = max(0, best_m - window)
        hi_m = min(MAX_START, best_m + window)
        for m in range(lo_m, hi_m + 1):
            if time.time() > deadline:
                break
            try_m(m)
    
    # Fourth pass: try values near multiples of lcm(1..12)=27720 and related numbers
    lcm_12 = 27720
    for mult in range(0, MAX_START // lcm_12 + 2):
        base = mult * lcm_12
        for offset in range(-50, 51):
            if time.time() > deadline:
                break
            try_m(base + offset)
        if time.time() > deadline:
            break
    
    # Final check: also try some specific interesting values
    special_ms = [0, 1, 2, 11, 12, 13, 60, 120, 252, 504, 1008, 
                  2520, 5040, 10080, 25200, 50400, 100800, 252000, 504000, 1000000,
                  27720, 55440, 83160, 110880, 138600, 166320, 194040, 221760,
                  249480, 277200, 304920, 332640, 360360, 388080, 415800, 443520,
                  471240, 498960, 526680, 554400, 582120, 609840, 637560, 665280,
                  693000, 720720, 748440, 776160, 803880, 831600, 859320, 887040,
                  914760, 942480, 970200, 997920]
    for m in special_ms:
        if time.time() > deadline:
            break
        try_m(m)
    
    return int(best_m)

# EVOLVE-BLOCK-END
