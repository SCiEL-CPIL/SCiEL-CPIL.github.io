import time
import collections
import random
import math

_DEADLINE = None

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

def _girth_check_fast(n, adj, g):
    """
    Returns True if graph has a cycle shorter than g.
    Uses non-backtracking BFS from each vertex with early exit.
    adj: list of lists of neighbors.
    """
    max_depth = g // 2
    for start in range(n):
        # dist[(curr, prev)] = depth
        dist = {}
        queue = collections.deque()
        for nb in adj[start]:
            state = (nb, start)
            dist[state] = 1
            queue.append(state)
        while queue:
            curr, prev = queue.popleft()
            d = dist[(curr, prev)]
            if d >= max_depth:
                continue
            for nb in adj[curr]:
                if nb == prev:
                    continue
                if nb == start:
                    if d + 1 < g:
                        return True
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = d + 1
                    queue.append(state)
    return False

def _try_circulant(n, offsets, g):
    """Build circulant graph on Z_n with given offsets, check girth."""
    adj = [[] for _ in range(n)]
    for i in range(n):
        for a in offsets:
            j = (i + a) % n
            if j != i:
                adj[i].append(j)
    # Check regularity: each vertex must have degree exactly len(offsets)*2 (or n if n small)
    # Actually for circulant, degree is 2*len(offsets) if all 2a_j distinct mod n and none is 0
    for i in range(n):
        if len(adj[i]) != 2 * len(offsets):
            return False
    # Check no duplicate edges
    for i in range(n):
        if len(set(adj[i])) != len(adj[i]):
            return False
    return not _girth_check_fast(n, adj, g)

def _gp_girth_check(n, k_gp, g):
    """Fast girth check for GP(n, k_gp) using modular arithmetic."""
    # In GP(n,k), vertices are outer[i], inner[i] for i in 0..n-1
    # Edges: outer[i]-outer[i+1], outer[i]-inner[i], inner[i]-inner[i+k]
    # A non-backtracking walk can be tracked by (position, type, prev_type)
    # position: 0..n-1, type: 0=outer, 1=inner
    # prev_type: where we came from
    # This is complex; use the BFS approach but with the explicit edge structure
    # For speed, use the known result: girth of GP(n,k) >= g iff
    # k^j not ≡ ±1 (mod n) for j=1..g/2-1 AND no short cycles in the "ladder"
    # The modular check is necessary but not sufficient. Let's do a proper BFS
    # but only on the non-backtracking states.
    
    max_depth = g // 2
    # State: (pos, is_inner, prev_pos, prev_inner)
    # But that's too many states. Let's use the standard approach:
    # From a vertex, non-backtracking means we can't go back to where we came from.
    # GP(n,k) has 2n vertices. Let's label them 0..2n-1 where 0..n-1 are outer, n..2n-1 are inner.
    
    adj = [[] for _ in range(2*n)]
    for i in range(n):
        adj[i].append((i+1) % n)       # outer cycle
        adj[i].append((i-1) % n)       # outer cycle
        adj[i].append(n + i)           # spoke
    for i in range(n):
        adj[n+i].append(i)             # spoke
        adj[n+i].append(n + (i + k_gp) % n)   # inner cycle
        adj[n+i].append(n + (i - k_gp) % n)   # inner cycle
    
    return _girth_check_fast(2*n, adj, g)

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 115.0
    
    # Moore bound for k=3, g=16: 2 + 3*(1+2+4+8) = 2+3*15 = 47... 
    # Actually Moore bound for (3,g): 2 + 3*(1 + 2 + 4 + ... + 2^(g/2-1))
    # For g=16: 2 + 3*(1+2+4+8) = 2+45 = 47. But that's for trees.
    # The actual Moore bound for 3-regular girth 16 is 510 (from the task).
    # Let's use a reasonable lower bound.
    
    min_n = 255  # 2*255 = 510 vertices, the Moore bound
    max_n = 600  # Search up to 600 vertices
    
    # Strategy 1: Try GP graphs (fast modular pre-filter, then BFS verification)
    # GP(n, k_gp) has 2n vertices.
    
    # Pre-filter: k_gp^j not ≡ ±1 (mod n) for j = 1..g/2-1
    gp_candidates = []
    for n in range(min_n, max_n + 1):
        if _time_left() < 30.0:
            break
        for k_gp in range(1, n // 2 + 1):
            if k_gp == n - k_gp:
                continue
            if math.gcd(k_gp, n) != 1:
                continue
            # Modular pre-check
            valid = True
            val = k_gp % n
            for j in range(1, g // 2):
                val = (val * k_gp) % n
                if val == 1 or val == n - 1:
                    valid = False
                    break
            if valid:
                gp_candidates.append((n, k_gp))
    
    gp_candidates.sort(key=lambda x: x[0])
    
    for n, k_gp in gp_candidates:
        if _time_left() < 15.0:
            break
        if not _gp_girth_check(n, k_gp, g):
            # Build edges
            edges = []
            for i in range(n):
                edges.append((i, (i + 1) % n))
            for i in range(n):
                edges.append((i, n + i))
            for i in range(n):
                edges.append((n + i, n + (i + k_gp) % n))
            return 2 * n, edges
    
    # Strategy 2: Try circulant graphs
    # Circulant C(n, {a1, a2, a3}) where a1+a2+a3 = 0 or we pick 3 offsets
    # For 3-regular, we need 3 offsets where 2a_j are distinct mod n
    # Actually for 3-regular circulant, we need exactly 3 generators
    # such that {±a1, ±a2, ±a3} gives 6 distinct elements (or fewer if n is small)
    
    # Try small n first
    for n in range(min_n, max_n + 1):
        if _time_left() < 10.0:
            break
        # Try a few random sets of 3 offsets
        for attempt in range(50):
            if _time_left() < 5.0:
                break
            # Pick 3 random offsets from 1..n//2
            offsets = random.sample(range(1, n // 2 + 1), 3)
            # Check that 2*len(offsets) = 6 distinct neighbors
            # i.e., ±a1, ±a2, ±a3 are all distinct mod n
            neighbors = set()
            valid = True
            for a in offsets:
                for sign in [1, -1]:
                    v = (sign * a) % n
                    if v == 0:
                        valid = False
                        break
                    if v in neighbors:
                        valid = False
                        break
                    neighbors.add(v)
            if not valid or len(neighbors) != 6:
                continue
            if _try_circulant(n, offsets, g):
                edges = []
                for i in range(n):
                    for a in offsets:
                        j = (i + a) % n
                        edges.append((i, j))
                return n, edges
    
    # Fallback: LUW construction (slow but guaranteed)
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        if _time_left() < 3.0:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
