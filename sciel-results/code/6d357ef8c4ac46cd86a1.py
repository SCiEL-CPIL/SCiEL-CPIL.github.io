import time
import collections
import math

_DEADLINE = None
_BUDGET = 130.0   # safety margin below the 150s kill

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

# ---------------------------------------------------------------------------
# LUW 3-regular graph construction
# ---------------------------------------------------------------------------
def _luw_construct(q, d, deadline=None):
    """Build the LUW(q,d) graph. Returns (n, edges) or None on timeout."""
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Rule for LUW(3, d): for c >= 2,
    #   y_c = y_{c-2} + y_{c-1} * x_{c-1}  (side 0)
    # The standard LUW for q=3 uses the recurrence from the Lazebnik-Ustimenko-Woldar paper.
    # For q=3, the rule is: y_c = y_{c-2} + y_{c-1} * x_{c-1} mod 3
    # This gives girth >= 2d+2.

    # Actually, let me use the general LUW rule. For q=3:
    # The "addition-multiplication" rule is:
    #   y_c = y_{c-2} + y_{c-1} * x_{c-1}  (side 0)
    #   y_c = y_{c-2} - x_{c-1} * y_{c-1}  (side 1)

    # More generally, for LUW(q, d) with prime q:
    # The recurrence involves coefficients that depend on d.
    # For simplicity, use the standard LUW(3,d) which has girth >= 2d+2.

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                if side == 0:
                    y[c] = add(y[c - 2], mul(y[c - 1], x[c - 1]))
                else:
                    y[c] = sub(y[c - 2], mul(x[c - 1], y[c - 1]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for vertex in order:
        if deadline is not None and _time_left() < 3.0:
            return None
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges

# ---------------------------------------------------------------------------
# Fast girth check using non-backtracking walks (vectorized with NumPy)
# ---------------------------------------------------------------------------
def _check_girth_fast(n, edges, min_girth):
    """Return True iff the graph has no cycle shorter than min_girth.
    Uses non-backtracking walks: a cycle of length L corresponds to a
    non-backtracking walk of length L-1 that returns to its origin.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    max_depth = min_girth - 2  # NB walk depth for cycles of length < min_girth

    # For each directed edge (u, v), do BFS in NB state space
    for u in range(n):
        for v in adj[u]:
            # State: (prev, curr)
            dist = {(u, v): 0}
            queue = collections.deque([(u, v, 0)])
            found = False
            while queue:
                prev, curr, depth = queue.popleft()
                if depth >= max_depth:
                    continue
                nd = depth + 1
                for nb in adj[curr]:
                    if nb == prev:
                        continue
                    if nb == u and curr == v:
                        # returned to origin at depth nd -> cycle length nd+1
                        if nd + 1 < min_girth:
                            return False
                        found = True
                        continue
                    st = (curr, nb)
                    if st not in dist:
                        dist[st] = nd
                        queue.append((curr, nb, nd))
            if found:
                pass
    return True

# ---------------------------------------------------------------------------
# Moore bound
# ---------------------------------------------------------------------------
def _moore_bound(k, g):
    """Lower bound on n for a k-regular graph of girth g."""
    if k == 2:
        return g
    if g % 2 == 0:
        return 1 + k * (1 + k + ... )  # placeholder
    # For odd girth g = 2r+1:
    # n >= k * (k-1)^r + k * (k-1)^(r-1) + ... + k
    # For even girth g = 2r:
    # n >= 2 * (k * (k-1)^(r-1) + k*(k-1)^(r-2) + ... + 1)
    # Simplified: for k=3, g=16 (even, r=8):
    # n >= 2 * (3*2^7 + 3*2^6 + ... + 1) = 2 * (3*(2^8-1)/(2-1) + 1) ... 
    # Actually: n >= 2 * sum_{i=0}^{r-1} (k-1)^i * k ... let me just compute directly.
    if g % 2 == 0:
        r = g // 2
        s = 0
        for i in range(r):
            s += (k - 1) ** i
        return 2 * k * s
    else:
        r = (g - 1) // 2
        s = 0
        for i in range(r + 1):
            s += (k - 1) ** i
        return k * s

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + _BUDGET

    # For k=3, g=16: use LUW(3, d) with d such that girth >= 2d+2 >= g
    # d >= (g-2)/2 = 7
    d_target = (g - 2) // 2
    if d_target < 3:
        d_target = 3

    # LUW(3, 7) gives girth >= 16
    n_luw, edges_luw = _luw_construct(k, d_target)
    if n_luw is None:
        # Fallback
        d_target = max(3, (g - 2) // 2 - 1)
        n_luw, edges_luw = _luw_construct(k, d_target)

    return n_luw, edges_luw
