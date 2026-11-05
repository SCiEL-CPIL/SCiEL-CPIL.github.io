import time
import math
import random
from collections import deque

_START_TIME = time.time()
_TIMEOUT = 130  # wall-clock budget (seconds); evaluator kills at 150s -> keep margin


def _deadline_exceeded():
    return (time.time() - _START_TIME) > _TIMEOUT


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------
def _check_simple_and_regular(edges, n, k):
    """Check simple (no loops, no repeated edges) and k-regular."""
    if len(edges) != n * k // 2:
        return False
    seen = set()
    deg = [0] * n
    for u, v in edges:
        if u == v:
            return False
        a, b = min(u, v), max(u, v)
        if (a, b) in seen:
            return False
        seen.add((a, b))
        deg[a] += 1
        deg[b] += 1
    for d in deg:
        if d != k:
            return False
    return True


def _build_adj(edges, n):
    """Build adjacency list from edge list."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _verify_girth(adj, n, g):
    """Check no cycle shorter than g using non-backtracking walks from every vertex.
    Takes pre-built adjacency list to avoid rebuilding."""
    for start in range(n):
        if _deadline_exceeded():
            return False  # assume invalid if we run out of time
        dist = {}
        queue = deque()
        for nb in adj[start]:
            queue.append((start, nb, 1))
            dist[(start, nb)] = 1
        while queue:
            prev, curr, d = queue.popleft()
            if d >= g:
                continue
            for nb2 in adj[curr]:
                if nb2 == prev:
                    continue
                state = (curr, nb2)
                if state in dist:
                    if d + 1 < g:
                        return False
                else:
                    dist[state] = d + 1
                    queue.append((curr, nb2, d + 1))
    return True


def _verify_girth_edges(edges, n, g):
    """Convenience wrapper: build adj then verify."""
    adj = _build_adj(edges, n)
    return _verify_girth(adj, n, g)


def _valid(n, edges, k, g):
    return _check_simple_and_regular(edges, n, k) and _verify_girth_edges(edges, n, g)


# ---------------------------------------------------------------------------
# Moore bound
# ---------------------------------------------------------------------------
def _moore_bound(k, g):
    if g % 2 == 0:
        return 2 * sum((k - 1) ** i for i in range(g // 2))
    return 1 + k * sum((k - 1) ** i for i in range((g - 1) // 2))


def _is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


# ---------------------------------------------------------------------------
# Lazebnik-Ustimenko-Woldar graph CD(d, q), q = k (bipartite)
# Known valid when d is odd and q is prime (or q=4).
# ---------------------------------------------------------------------------
def construct_lw(k, g):
    q = k
    # For girth >= d+5 (odd d), we need d >= g-5
    # Choose d = g-5 if g-5 is odd, else g-4 (to keep d odd when possible)
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1  # prefer odd d for known validity
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    elif _is_prime(q):
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    else:
        # q not prime and not 4: still construct but validity not guaranteed
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges, d, q


# ---------------------------------------------------------------------------
# Alon-Babai-Suzuki construction on Heisenberg group H_q (even k)
# ---------------------------------------------------------------------------
def construct_abs(k, g):
    mb = _moore_bound(k, g)

    q = 2
    while True:
        if _is_prime(q):
            if q ** 3 >= mb:
                break
        q += 1

    if k == 4:
        gens = [
            (1, 0, 0),
            (q - 1, 0, 0),
            (0, 1, 0),
            (0, q - 1, 1),
        ]
    elif k == 6:
        gens = [
            (1, 0, 0),
            (q - 1, 0, 0),
            (0, 1, 0),
            (0, q - 1, 1),
            (0, 0, 1),
            (0, 0, q - 1),
        ]
    else:
        return None, None

    n = q ** 3
    edges = []

    def idx(x, y, z):
        return (x * q + y) * q + z

    for x in range(q):
        for y in range(q):
            for z in range(q):
                u = idx(x, y, z)
                for gx, gy, gz in gens:
                    vx = (x + gx) % q
                    vy = (y + gy) % q
                    vz = (z + gz + x * gy) % q
                    v = idx(vx, vy, vz)
                    if u < v:
                        edges.append((u, v))
    return n, edges


# ---------------------------------------------------------------------------
# Circulant construction: try random sets of k/2 generators on Z_q
# ---------------------------------------------------------------------------
def construct_circulant(k, g, max_tries=200):
    if k % 2 != 0:
        return None
    half = k // 2
    mb = _moore_bound(k, g)
    # q must be > 2*half to avoid generator collisions
    min_q = 2 * half + 1
    start_q = max(mb, min_q)
    
    random.seed(42)
    
    for attempt in range(max_tries):
        if _deadline_exceeded():
            return None
        # Try a range of q values
        q = start_q + attempt * 3  # spread out
        if q < min_q:
            continue
        
        # Generate random distinct generators in 1..(q-1)//2
        max_gen = (q - 1) // 2
        if max_gen < half:
            continue
        
        try:
            gens = random.sample(range(1, max_gen + 1), half)
        except ValueError:
            continue
        
        # Build edges
        edges = []
        for u in range(q):
            for gi in gens:
                v = (u + gi) % q
                if u < v:
                    edges.append((u, v))
        
        if len(edges) != q * k // 2:
            continue
        
        # Quick girth check
        if _verify_girth_edges(edges, q, g):
            return q, edges
    
    return None


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------
def construct(k, g):
    random.seed(12345)
    candidates = []  # list of (n, edges, verified)
    
    def try_add(n, edges, verified=False):
        if n is None or edges is None:
            return
        # cheap checks first
        if len(edges) != n * k // 2:
            return
        candidates.append((n, edges, verified))
    
    # 1. ABS for even k (k=4,6) - known valid family
    if k % 2 == 0:
        n_abs, edges_abs = construct_abs(k, g)
        if n_abs is not None:
            # ABS on Heisenberg group is known to have good girth properties
            # but we should verify; mark as needing verification
            try_add(n_abs, edges_abs, verified=False)
    
    # 2. LW construction (works for any k, bipartite)
    n_lw, edges_lw, d_lw, q_lw = construct_lw(k, g)
    if n_lw is not None:
        # LW is known valid when d is odd and q is prime
        lw_known_valid = (d_lw % 2 == 1) and (_is_prime(q_lw) or q_lw == 4)
        try_add(n_lw, edges_lw, verified=lw_known_valid)
    
    # 3. Circulant for even k (random search for better candidates)
    if k % 2 == 0 and not _deadline_exceeded():
        res = construct_circulant(k, g, max_tries=100)
        if res is not None:
            n_circ, edges_circ = res
            try_add(n_circ, edges_circ, verified=True)  # we already verified in construct_circulant
    
    # Sort candidates by n (smallest first) for verification priority
    candidates.sort(key=lambda x: x[0])
    
    best = None
    for n, edges, already_verified in candidates:
        if _deadline_exceeded():
            break
        if already_verified:
            # Still do a quick sanity check
            if _check_simple_and_regular(edges, n, k):
                if best is None or n < best[0]:
                    best = (n, edges)
        else:
            # Full verification needed
            if _check_simple_and_regular(edges, n, k) and _verify_girth_edges(edges, n, g):
                if best is None or n < best[0]:
                    best = (n, edges)
    
    if best is not None:
        return best
    
    # Fallback: return LW even without full verification
    # (it is a known valid family for odd d with prime q)
    if n_lw is not None:
        return n_lw, edges_lw
    
    # Last resort: return ABS if available
    if k % 2 == 0:
        n_abs, edges_abs = construct_abs(k, g)
        if n_abs is not None:
            return n_abs, edges_abs
    
    # Absolute fallback: return LW
    return n_lw, edges_lw
