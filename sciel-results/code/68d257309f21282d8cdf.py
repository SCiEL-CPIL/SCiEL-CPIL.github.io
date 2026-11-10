"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

P = {'n': 12, 'maximum_start': 1000000}

def _has_perfect_matching(m, L):
    """Check if in (m, m+L] we can assign distinct b_i to each i in 1..12 with i|b_i."""
    n = 12
    # Collect candidates for each i
    # For i, candidates are multiples of i in (m, m+L]
    # Build bipartite graph: left = indices 1..12, right = numbers in (m, m+L]
    
    # Get all numbers in (m, m+L]
    # But we only care about numbers that are multiples of at least one of 1..12
    # Actually every number is a multiple of 1, so all numbers in range are candidates
    
    # For efficiency, build adjacency: for each i, which numbers in (m, m+L] are divisible by i
    # Use DFS matching (Hopcroft-Karp is overkill for n=12)
    
    # Right side: numbers m+1 to m+L
    # Map each number to an index 0..L-1
    # For each left node i (1-indexed), find right nodes j where (m+1+j) % i == 0
    
    # Build adjacency list for left side
    adj = [[] for _ in range(n)]
    for i in range(1, n+1):
        # First multiple of i greater than m
        first = m + 1
        rem = first % i
        if rem == 0:
            first_mult = first
        else:
            first_mult = first + (i - rem)
        for b in range(first_mult, m + L + 1, i):
            # b is in (m, m+L], index is b - m - 1
            idx = b - m - 1
            if 0 <= idx < L:
                adj[i-1].append(idx)
    
    # DFS-based bipartite matching
    match_r = [-1] * L  # which left node is matched to right node j
    match_l = [-1] * n  # which right node is matched to left node i
    
    def dfs(u, visited):
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if match_r[v] == -1 or dfs(match_r[v], visited):
                match_r[v] = u
                match_l[u] = v
                return True
        return False
    
    for u in range(n):
        visited = [False] * L
        if not dfs(u, visited):
            return False
    return True

def _compute_f(m):
    """Compute f(12, m): least positive L such that (m, m+L] has distinct b_i with i|b_i."""
    # L is at most 144
    # Try L from 1 upward, but use binary search for efficiency
    # Actually max is 144, so linear scan is fine too, but let's binary search
    lo, hi = 1, 144
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_perfect_matching(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    best_m = 12
    best_f = _compute_f(12)
    
    # Strategy: coarse scan + local refinement
    # The function f(12, m) has structure related to the distribution of multiples
    # Let's scan m in steps, then refine
    
    # Coarse scan: step through m values
    # f(12, m) is periodic-ish with period LCM(1..12)=27720, but let's not assume
    # Do a coarse scan over [0, 1000000]
    
    # First, try some strategic points
    # m near multiples of LCM(1..12) - 1 might be interesting
    lcm_12 = 27720
    
    # Coarse scan with step size
    step = 50
    m = 0
    while m <= 1000000:
        if time.time() > deadline - 5:
            break
        f_val = _compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
        m += step
    
    # Local refinement around best_m
    if time.time() < deadline - 15:
        for dm in range(-500, 501):
            if time.time() > deadline - 5:
                break
            m2 = best_m + dm
            if 0 <= m2 <= 1000000:
                f_val = _compute_f(m2)
                if f_val > best_f:
                    best_f = f_val
                    best_m = m2
    
    # Additional targeted search: try m values near (k*lcm_12 - 1)
    if time.time() < deadline - 15:
        for k in range(0, 37):  # k*lcm_12 <= 1000000
            m_base = k * lcm_12
            for offset in range(-100, 101):
                if time.time() > deadline - 5:
                    break
                m2 = m_base - 1 + offset
                if 0 <= m2 <= 1000000:
                    f_val = _compute_f(m2)
                    if f_val > best_f:
                        best_f = f_val
                        best_m = m2
    
    # Final: try a few more random-ish spots if time permits
    if time.time() < deadline - 15:
        import random
        random.seed(42)
        for _ in range(1000):
            if time.time() > deadline - 10:
                break
            m2 = random.randint(0, 1000000)
            f_val = _compute_f(m2)
            if f_val > best_f:
                best_f = f_val
                best_m = m2
    
    # Final fine-grained scan around best m
    if time.time() < deadline - 10:
        for dm in range(-20, 21):
            if time.time() > deadline - 5:
                break
            m2 = random.randint(0, 1000000)
            f_val = _compute_f(m2)
            if f_val > best_f:
                best_f = f_val
                best_m = m2
    
    return int(best_m)
# EVOLVE-BLOCK-END
