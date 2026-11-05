import time
import sys
import random
from collections import deque

# EVOLVE-BLOCK-START

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _luw_graph(q, d):
    """
    Lazebnik-Ustimenko-Woldar construction.
    For odd d >= 3, produces a q-regular bipartite graph on 2*q^(d+1) vertices
    with girth at least d+5.
    
    Uses the rule: for c in 2..d, (a_c, b_c)
    """
    if q == 4:
        # Special case using GF(4)
        table_add = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        table_mul = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def add(x, y): return table_add[x][y]
        def sub(x, y): return table_add[x][y]
        def mul(x, y): return table_mul[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    # Define the rule (a_c, b_c) for c = 2, ..., d
    # Standard LUW rule:
    # c=2: (1,1), c=3: (2,1), c=4: (1,2)
    # For c >= 5: use periodic pattern
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    if d >= 4:
        rule[4] = (1, 2)
    for c in range(5, d + 1):
        # Use a standard pattern that works
        r = (c - 5) % 4
        c0 = c - r
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1: x[0], x[1], ..., x[d]
        for first in range(q):
            y = [None] * (d + 1)
            y[0] = None  # placeholder
            y[1] = first
            # Compute y[2], ..., y[d]
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Left to right: y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Right to left: y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side), tuple(y)

    # Start from a fixed vertex on the left side
    start = (0, (0,) * (d + 1))
    index = {start: 0}
    order = [start]
    edges = []

    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    n = len(order)
    if n == 0:
        return None, None
    return n, edges


def _verify_girth_nonbacktracking(n, edges, g):
    """
    Verify that no cycle shorter than g exists using non-backtracking walks.
    For each vertex, do a non-backtracking BFS up to depth (g-1)//2.
    If we revisit a vertex at depth < (g-1)//2 via a non-backtracking path,
    we have a cycle shorter than g.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    max_depth = (g - 1) // 2

    for s in range(n):
        # State: (vertex, prev_vertex, depth)
        # Track visited states to detect cycles
        visited = set()
        queue = deque()
        # Start: from s, can go to any neighbor
        for v in adj[s]:
            state = (v, s, 1)
            if state not in visited:
                visited.add(state)
                queue.append(state)

        while queue:
            u, prev, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for v in adj[u]:
                if v == prev:
                    continue  # non-backtracking
                if v == s and depth + 1 < g:
                    # Found a cycle of length depth + 1 < g
                    return False
                state = (v, u, depth + 1)
                if state in visited:
                    # We've seen this state before, which means there's a cycle
                    # through this state. The cycle length is at most depth+1.
                    # But we need to be more careful.
                    # Actually, if we reach a previously visited (v, u) state,
                    # it means there's a non-backtracking cycle.
                    # The cycle length would be related to the depth difference.
                    # For safety, if we revisit any state, check if it implies short cycle.
                    # A simpler check: if v == s, we already handled it.
                    # For other revisits, the cycle through (v,u) has length <= depth+1
                    # if the first visit was at depth 0 (which it isn't since we start at depth 1).
                    # Let's just flag it as potential short cycle.
                    return False
                visited.add(state)
                queue.append(state)

    return True


def _verify_degree(n, edges, k):
    """Check that every vertex has degree exactly k."""
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return all(d == k for d in deg)


def _random_k_regular(n, k):
    """Generate a random k-regular graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        return None
    if k >= n:
        return None

    half_edges = []
    for v in range(n):
        half_edges.extend([v] * k)

    random.shuffle(half_edges)

    edges = set()
    for i in range(0, len(half_edges), 2):
        u, v = half_edges[i], half_edges[i + 1]
        if u == v:
            return None
        edge = (min(u, v), max(u, v))
        if edge in edges:
            return None
        edges.add(edge)

    return list(edges)


def _quick_girth_check(n, edges, g):
    """
    Quick check: for each edge (u,v), compute the shortest path from u to v
    not using that edge. If 1 + shortest_path < g, girth is violated.
    This is O(n * m) which is expensive, so we only check a sample.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # For a quick check, do BFS from each endpoint of a random subset of edges
    import random as _rnd
    sample_size = min(50, len(edges))
    sample_edges = _rnd.sample(edges, sample_size) if len(edges) >= sample_size else edges

    for u, v in sample_edges:
        # BFS from u to v, not using edge (u,v)
        dist = [-1] * n
        dist[u] = 0
        queue = deque([u])
        found = -1
        while queue:
            x = queue.popleft()
            for y in adj[x]:
                if (x == u and y == v) or (x == v and y == u):
                    continue  # skip the edge we're excluding
                if dist[y] == -1:
                    dist[y] = dist[x] + 1
                    if y == v:
                        found = dist[y]
                        break
                    queue.append(y)
            if found != -1:
                break
        if found != -1 and 1 + found < g:
            return False

    return True


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Moore bound
    if g % 2 == 0:
        if k == 2:
            moore = g
        else:
            moore = k * sum((k - 1) ** i for i in range(g // 2))
    else:
        if k == 2:
            moore = g
        else:
            moore = 2 * sum((k - 1) ** i for i in range((g + 1) // 2))

    best_n = None
    best_edges = None

    def try_candidate(n, edges):
        nonlocal best_n, best_edges
        if n is None or n == 0 or edges is None:
            return
        if time.time() > deadline:
            return
        if not _verify_degree(n, edges, k):
            return
        # Quick girth check first
        if not _quick_girth_check(n, edges, g):
            return
        # Full verification
        if not _verify_girth_nonbacktracking(n, edges, g):
            return
        if best_n is None or n < best_n:
            best_n = n
            best_edges = edges

    # Strategy 1: LUW construction
    # For k=5, g=9: need q-regular with girth >= 9
    # LUW with q=5, d=5 gives girth >= 10, 2*5^6 = 31250 vertices
    # LUW with q=5, d=3 gives girth >= 8 (not enough)
    # LUW with q=7, d=5 gives 2*7^6 = 33614 vertices
    # These are large. Let's try smaller q.
    # Actually, for the cage problem, we want minimal n.
    # LUW might not be optimal, but it's a valid construction.

    # Try LUW with various parameters
    luw_params = []
    if k == 5 and g == 9:
        # d must be odd, d >= g-5 = 4, so d >= 5
        for q_try in [5, 7, 11, 13]:
            if q_try == 4 or _is_prime(q_try):
                for d_try in [5, 7]:
                    if d_try + 5 >= g:
                        luw_params.append((q_try, d_try))
    else:
        d_min = max(3, g - 5)
        if d_min % 2 == 0:
            d_min += 1
        for q_try in range(k, k + 30):
            if q_try == 4 or _is_prime(q_try):
                for d_try in [d_min, d_min + 2]:
                    if d_try + 5 >= g:
                        luw_params.append((q_try, d_try))

    for q_try, d_try in luw_params:
        if time.time() > deadline - 5:
            break
        n, edges = _luw_graph(q_try, d_try)
        if n is not None:
            try_candidate(n, edges)

    # Strategy 2: Random k-regular graph search
    # Try to find a random k-regular graph with girth >= g
    if time.time() < deadline - 10:
        # Start from a reasonable size
        n_candidate = max(moore, 2 * k)
        # Try increasing sizes
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            # Try a few random graphs at this size
            for _ in range(5):
                if time.time() > deadline - 5:
                    break
                edges = _random_k_regular(n_candidate, k)
                if edges is not None:
                    if _quick_girth_check(n_candidate, edges, g):
                        if _verify_girth_nonbacktracking(n_candidate, edges, g):
                            if best_n is None or n_candidate < best_n:
                                best_n = n_candidate
                                best_edges = edges
                            break
            if best_n is not None and best_n <= n_candidate:
                break
            n_candidate += k  # increase by k to maintain parity

    if best_n is None:
        # Fallback: return a cycle for k=2
        if k == 2:
            n = g
            edges = [(i, (i + 1) % n) for i in range(n)]
            return n, edges
        else:
            # Last resort: return empty
            return 0, []

    return best_n, best_edges

# EVOLVE-BLOCK-END
