import random
import time
from collections import deque

# ---- global deadline (150s wall-clock kill; keep margin) ----
_T0 = time.time()
_DEADLINE = _T0 + 140.0


def _verify_girth(adj, g):
    """Check that graph with adjacency list 'adj' has girth >= g.
    Uses BFS from each vertex to depth (g-1)//2, tracking parent to detect
    cycles. Returns True if girth >= g."""
    n = len(adj)
    max_depth = (g - 1) // 2  # for g=16, depth=7
    for start in range(n):
        # BFS from start, tracking (node, depth, parent)
        visited = {}  # node -> (depth, parent)
        visited[start] = (0, -1)
        queue = deque([(start, 0, -1)])
        while queue:
            node, depth, parent = queue.popleft()
            if depth >= max_depth:
                continue
            for nb in adj[node]:
                if nb == parent:
                    continue
                if nb in visited:
                    # Found a cycle of length depth + 1 + visited[nb][0]
                    cycle_len = depth + 1 + visited[nb][0]
                    if cycle_len < g:
                        return False
                else:
                    visited[nb] = (depth + 1, node)
                    queue.append((nb, depth + 1, node))
    return True


def _generate_random_3regular(n):
    """Generate a random 3-regular graph on n vertices using configuration model.
    Returns adjacency list or None if multigraph/loops detected."""
    if n < 4 or n % 2 != 0:
        return None
    # Each vertex has 3 half-edges
    stubs = []
    for v in range(n):
        for _ in range(3):
            stubs.append(v)
    random.shuffle(stubs)
    # Pair up stubs
    adj = [[] for _ in range(n)]
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            return None  # loop
        if v in adj[u]:
            return None  # multi-edge
        adj[u].append(v)
        adj[v].append(u)
    # Verify all degrees are exactly 3
    for v in range(n):
        if len(adj[v]) != 3:
            return None
    return adj


def _try_find_graph(k, g, start_n, max_n, time_limit):
    """Try to find a k-regular graph of girth >= g with n vertices,
    starting from start_n up to max_n."""
    deadline = time.time() + time_limit
    for n in range(start_n, max_n + 1):
        if time.time() > deadline:
            return None
        if n * k % 2 != 0:
            continue
        # Try multiple random instances
        attempts = 0
        max_attempts = 200
        while attempts < max_attempts and time.time() < deadline:
            adj = _generate_random_3regular(n)
            if adj is None:
                attempts += 1
                continue
            if _verify_girth(adj, g):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
            attempts += 1
    return None


def construct(k, g):
    # For k=3, g=16: try randomized search for small 3-regular graphs
    # with girth >= 16.
    if k == 3 and g == 16:
        # Moore bound for n(3,16): 2*sum_{i=0}^{7} 3^i = 2*(3^8-1)/2 = 6560
        # But best known is 936. Start searching from a reasonable range.
        # The lower bound is 512 per the task description.
        # Try from 936 downward first (known to exist), then try smaller.
        # Actually, we want the SMALLEST, so try from lower bound up.
        # But random search is unlikely to find one at the minimum.
        # Strategy: try a range of n values, preferring smaller ones.
        
        time_remaining = _DEADLINE - time.time()
        if time_remaining < 5:
            # Fallback: LUW construction
            return _luw_construct(k, g)
        
        # Try from the known best (936) first to guarantee a valid answer,
        # then try to find smaller ones.
        # Phase 1: Try to find a valid graph near 936
        result = _try_find_graph(k, g, 936, 940, min(30, time_remaining * 0.3))
        if result is not None:
            best_n, best_edges = result
        else:
            # Fallback to LUW
            return _luw_construct(k, g)
        
        # Phase 2: Try to find smaller graphs
        time_remaining = _DEADLINE - time.time()
        if time_remaining > 10:
            # Try smaller n values
            for start in range(512, 936, 64):
                if time.time() > _DEADLINE:
                    break
                end = min(start + 63, 935)
                time_budget = min(20, (_DEADLINE - time.time()) * 0.2)
                if time_budget < 2:
                    break
                result = _try_find_graph(k, g, start, end, time_budget)
                if result is not None:
                    best_n, best_edges = result
                    # Keep searching for even smaller
                    break  # Found one in this range, stop
        
        return best_n, best_edges
    else:
        # General case: use LUW construction
        return _luw_construct(k, g)


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
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
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
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
    return len(order), edges


