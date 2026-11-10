"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

N = 12
MAX_M = 1000000
MAX_L = 144  # proven upper bound: any 144 consecutive ints contain a full matching


def _has_matching(m, L):
    """Return True iff (m, m+L] contains distinct b_i with i | b_i for i=1..N.
    Uses Hopcroft-Karp bipartite matching. i in 1..N (left) -> multiples in (m, m+L] (right)."""
    # Build right side: distinct integers in (m, m+L], grouped by which i they are divisible by.
    # For each i, the candidate right nodes are multiples of i in (m, m+L].
    # Represent right nodes by their actual value v; node id = v - (m+1) in 0..L-1.
    # adj[i] = list of right-node indices v-m-1 where v is a multiple of i.
    adj = [[] for _ in range(N)]
    # For each i, iterate multiples of i in (m, m+L]
    for i in range(1, N + 1):
        start = m + 1
        # first multiple of i >= start
        q = (start + i - 1) // i
        v = q * i
        while v <= m + L:
            adj[i - 1].append(v - m - 1)
            v += i
    # Hopcroft-Karp
    INF = 10 ** 9
    pair_u = [-1] * N          # for left node i (0..N-1), matched right node or -1
    pair_v = [-1] * L          # for right node, matched left node or -1
    dist = [0] * N

    def bfs():
        # returns True if there is an augmenting path
        from collections import deque
        dq = deque()
        found = False
        for i in range(N):
            if pair_u[i] == -1:
                dist[i] = 0
                dq.append(i)
            else:
                dist[i] = INF
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1:
                    found = True
                elif dist[pu] == INF:
                    dist[pu] = dist[u] + 1
                    dq.append(pu)
        return found

    def dfs(u):
        for v in adj[u]:
            pu = pair_v[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = INF
        return False

    matching = 0
    while bfs():
        for i in range(N):
            if pair_u[i] == -1:
                if dfs(i):
                    matching += 1
    return matching == N


def _compute_f(m):
    """Compute f(N, m): least positive L with a matching in (m, m+L]."""
    # Binary search L in [1, MAX_L]
    lo, hi = 1, MAX_L
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_matching(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def _find_best(deadline):
    """Search for m maximizing f(N,m). Returns (best_m, best_f)."""
    best_m = 0
    best_f = 0

    # First, determine the maximum achievable L via binary search over L.
    # For a given L, does there exist m in [0, MAX_M] with f(N,m) >= L?
    # f(N,m) >= L  <=>  _has_matching(m, L-1) is False (no matching in (m, m+L-1])
    #               AND we still need f to be exactly... actually f>=L means no matching for any L' < L.
    # Simpler: f(N,m) >= L  iff  NOT _has_matching(m, L-1).
    # So existence of m with f>=L  iff  exists m with NOT _has_matching(m, L-1).

    def exists_with_f_at_least(L):
        # need m in [0, MAX_M] such that NOT _has_matching(m, L-1)
        # scan a sample; but we want correctness. We'll scan all m? Too slow.
        # Instead, we only need to know the max L and find one m. We'll just scan m.
        return None  # placeholder

    # Practical approach: scan m over a coarse grid, compute f, track best.
    # Then locally refine around the best m.
    t0 = time.time()

    # Phase 1: coarse scan
    step = 37  # coprime-ish step to hit varied residues
    m = 0
    while m <= MAX_M and (time.time() - t0) < deadline:
        f = _compute_f(m)
        if f > best_f:
            best_f = f
            best_m = m
        m += step

    # Phase 2: local refinement around best_m
    if best_f > 0 and (time.time() - t0) < deadline:
        center = best_m
        radius = 200
        for dm in range(0, radius + 1):
            for m2 in (center - dm, center + dm):
                if m2 < 0 or m2 > MAX_M:
                    continue
                if (time.time() - t0) > deadline:
                    break
                f = _compute_f(m2)
                if f > best_f:
                    best_f = f
                    best_m = m2

    # Phase 3: if we found a high f, try to push further with targeted search.
    # The structure: f is large when m is just below a "gap" in the matching structure.
    # Specifically, f(N,m) is the distance from m to the next m' where a matching completes.
    # We can look for m such that for some i, the multiples are sparse.
    # Try scanning m in a finer grid over a wider range if time permits.
    if (time.time() - t0) < deadline:
        # Finer scan over the whole range with smaller step
        step2 = 7
        m = 0
        while m <= MAX_M and (time.time() - t0) < deadline:
            f = _compute_f(m)
            if f > best_f:
                best_f = f
                best_m = m
            m += step2

    return best_m, best_f


def construct():
    deadline = 75.0  # safety margin before 90s kill
    t0 = time.time()
    best_m, best_f = _find_best(deadline)
    # Ensure type is Python int
    return int(best_m)
# EVOLVE-BLOCK-END
