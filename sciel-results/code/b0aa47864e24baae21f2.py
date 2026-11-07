"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _has_matching(m, L):
    """Check if interval (m, m+L] has distinct b_1..b_12 with i|b_i."""
    # Collect all candidate numbers in (m, m+L]
    candidates = set()
    for i in range(1, 13):
        start = (m // i + 1) * i
        for k in range(start, m + L + 1, i):
            candidates.add(k)
    
    if len(candidates) < 12:
        return False
    
    cand_list = sorted(candidates)
    cand_idx = {v: i for i, v in enumerate(cand_list)}
    n_right = len(cand_list)
    
    # Adjacency: for each left node i-1 (i=1..12), which right nodes are valid
    adj = [set() for _ in range(12)]
    for i in range(1, 13):
        start = (m // i + 1) * i
        for k in range(start, m + L + 1, i):
            if k in cand_idx:
                adj[i-1].add(cand_idx[k])
    
    # Kuhn's algorithm (DFS-based bipartite matching)
    match_r = [-1] * n_right
    
    def dfs(u, visited):
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if match_r[v] == -1 or dfs(match_r[v], visited):
                match_r[v] = u
                return True
        return False
    
    for u in range(12):
        visited = [False] * n_right
        if not dfs(u, visited):
            return False
    return True


def _compute_f(m):
    """Compute f(12, m): least L such that (m, m+L] has valid assignment."""
    lo, hi = 1, 200
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_matching(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    best_m = 12
    best_f = _compute_f(12)
    
    tested = set()
    
    def try_m(mm):
        nonlocal best_m, best_f
        if 0 <= mm <= 1000000 and mm not in tested:
            tested.add(mm)
            if time.time() > deadline:
                return
            f_val = _compute_f(mm)
            if f_val > best_f:
                best_f = f_val
                best_m = mm
    
    # Structured search: m near multiples of LCM(1..12)=27720
    for k in range(0, 37):
        m_val = k * 27720
        if m_val > 1000000:
            break
        for offset in range(-3, 4):
            try_m(m_val + offset)
    
    # m near multiples of 2520 = LCM(1..10)
    for k in range(0, 400):
        m_val = k * 2520
        if m_val > 1000000:
            break
        for offset in range(-2, 3):
            try_m(m_val + offset)
    
    # m near multiples of 840 = LCM(1..8)
    for k in range(0, 1200):
        m_val = k * 840
        if m_val > 1000000:
            break
        for offset in range(-1, 2):
            try_m(m_val + offset)
    
    # m near multiples of 420 = LCM(1..7)
    for k in range(0, 2400):
        m_val = k * 420
        if m_val > 1000000:
            break
        try_m(m_val)
        try_m(m_val - 1)
    
    # Random search
    rng = random.Random(42)
    for _ in range(10000):
        if time.time() > deadline:
            break
        try_m(rng.randint(0, 1000000))
    
    return int(best_m)
# EVOLVE-BLOCK-END
