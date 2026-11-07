import time

N = 12
MAX_M = 1000000
DEADLINE = 82.0  # seconds, safety margin below 90s

def _can_match(m, L, n=N):
    """Exact: does (m, m+L] contain distinct b_1..b_n with i | b_i?"""
    adj = []
    for i in range(1, n + 1):
        first = ((m // i) + 1) * i
        mask = 0
        v = first
        end = m + L
        while v <= end:
            mask |= (1 << v)
            v += i
        adj.append(mask)

    pair_u = [-1] * n
    dist = [0] * n

    def bfs():
        from collections import deque
        q = deque()
        for u in range(n):
            if pair_u[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = -1
        found = False
        while q:
            u = q.popleft()
            msk = adj[u]
            while msk:
                lb = msk & (-msk)
                v = lb.bit_length() - 1
                msk ^= lb
                pu = pair_u[v]
                if pu == -1:
                    found = True
                elif dist[pu] == -1:
                    dist[pu] = dist[u] + 1
                    q.append(pu)
        return found

    def dfs(u):
        msk = adj[u]
        while msk:
            lb = msk & (-msk)
            v = lb.bit_length() - 1
            msk ^= lb
            pu = pair_u[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[v] = u
                return True
        dist[u] = -1
        return False

    matching = 0
    while bfs():
        for u in range(n):
            if pair_u[u] == -1 and dfs(u):
                matching += 1
    return matching == n

def _f_val(m, n=N):
    """Least positive L with a valid matching."""
    lo, hi = 1, 144
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid, n):
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return best if best is not None else 144

def construct():
    start = time.time()

    def timed():
        return time.time() - start > DEADLINE

    best_m = 0
    best_f = _f_val(0)
    seen = {0}

    # 1) Guaranteed candidates: m = 12k (i=12 first offset = 12)
    for k in range(0, MAX_M // 12 + 1):
        m = 12 * k
        if m > MAX_M:
            break
        if m in seen:
            continue
        seen.add(m)
        if timed():
            break
        f = _f_val(m)
        if f > best_f:
            best_f = f
            best_m = m

    # 2) Broad sweep (adaptive step) to catch any distinctness-driven outlier
    m = 0
    while m <= MAX_M:
        if m not in seen:
            seen.add(m)
            if not timed():
                f = _f_val(m)
                if f > best_f:
                    best_f = f
                    best_m = m
        if m < 50000:
            m += 1
        elif m < 200000:
            m += 2
        else:
            m += 4
        if timed():
            break

    return int(best_m)
# EVOLVE-BLOCK-END
