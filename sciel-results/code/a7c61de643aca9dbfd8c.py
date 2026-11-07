import time
import math

# EVOLVE-BLOCK-START
# Strategy:
# f(12, m) is periodic with period 27720 = LCM(1..12).
# We exhaustively search all residues to find the one maximizing f(12, r).
# Early termination if we find value 144 (the theoretical maximum).
# Uses optimized Kuhn's algorithm with timestamp visited arrays.

PERIOD = 27720
N = 12
MAX_M = 1000000
MAX_L = 144  # Upper bound given in problem

def has_valid_set(m, L):
    """
    Check if there exists a perfect matching: assign each index i in 1..12
    to a distinct number in (m, m+L] that is divisible by i.
    Uses Kuhn's algorithm (DFS augmenting paths) with timestamp visited.
    """
    # Precompute neighbors: for each i (1-indexed), which positions j in 0..L-1
    # have (m+1+j) divisible by i
    neighbors = [[] for _ in range(N)]
    start = m + 1
    for i in range(1, N + 1):
        # Find smallest j >= 0 such that (start + j) % i == 0
        rem = (-start) % i
        if rem < L:
            # j = rem, rem+i, rem+2i, ... < L
            for j in range(rem, L, i):
                neighbors[i - 1].append(j)
    
    # match_right[j] = u means position j is matched to index u (0-indexed, i=u+1)
    match_right = [-1] * L
    visited = [0] * L
    stamp = 0

    def dfs(u):
        nonlocal stamp
        for j in neighbors[u]:
            if visited[j] == stamp:
                continue
            visited[j] = stamp
            if match_right[j] == -1 or dfs(match_right[j]):
                match_right[j] = u
                return True
        return False

    for u in range(N):
        stamp += 1
        if not dfs(u):
            return False
    return True

def f_12(m):
    """
    Binary search for the least L such that (m, m+L] has a valid matching.
    """
    lo, hi = 12, MAX_L
    while lo < hi:
        mid = (lo + hi) // 2
        if has_valid_set(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    deadline = 80  # Safety margin before 90s limit
    
    best_r = 0
    best_val = 0
    
    # First pass: scan all residues, early exit if we hit 144
    for r in range(PERIOD):
        if time.time() - start_time > deadline:
            break
        
        val = f_12(r)
        if val > best_val:
            best_val = val
            best_r = r
            if best_val >= MAX_L:
                # Found the theoretical maximum, no need to continue
                break
    
    # Map best_r to the largest m in [0, MAX_M] with m % PERIOD == best_r
    if best_r > MAX_M:
        m = best_r
    else:
        k_max = (MAX_M - best_r) // PERIOD
        m = k_max * PERIOD + best_r
    
    # Ensure m is within bounds
    if m < 0:
        m = 0
    if m > MAX_M:
        m = MAX_M
    
    # Verify the result is a valid Python int (not bool, not numpy int)
    if not isinstance(m, int) or isinstance(m, bool):
        m = int(m)
    
    return m
# EVOLVE-BLOCK-END
