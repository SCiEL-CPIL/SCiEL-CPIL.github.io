"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
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
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _girth_check_fast(n, edges, k, g):
    """Check that the graph is k-regular and has girth >= g using non-backtracking BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check regularity
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    # Check girth via BFS from each vertex
    for start in range(n):
        # dist[v] = shortest distance from start to v
        # parent[v] = parent in BFS tree
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            du = dist[u]
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and parent[v] != u:
                    # Found a cycle through u-v
                    # Cycle length = dist[u] + dist[v] - 2*dist[lca] + 1
                    # But we can compute it as: the path from u to lca + path from v to lca + 1
                    a, b = u, v
                    da, db = dist[u], dist[v]
                    while a != b:
                        if da > db:
                            a = parent[a]
                            da -= 1
                        elif db > da:
                            b = parent[b]
                            db -= 1
                        else:
                            a = parent[a]
                            b = parent[b]
                            da -= 1
                            db -= 1
                    cycle_len = dist[u] + dist[v] - 2 * da + 1
                    if cycle_len < g:
                        return False
    return True


def _generate_random_regular(n, k, seed=None):
    """Generate a random k-regular graph on n vertices using configuration model."""
    if seed is not None:
        random.seed(seed)
    # Each vertex needs k stubs
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    random.shuffle(stubs)
    
    edges = []
    edge_set = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            return None  # self-loop
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return None  # multi-edge
        edge_set.add(key)
        edges.append((u, v))
    
    return edges


def _try_random_regular_search(k, g, deadline, n_start, n_end, max_attempts_per_n=200):
    """Search for random k-regular graphs with girth >= g."""
    for n in range(n_start, n_end + 1):
        if time.time() > deadline:
            break
        for attempt in range(max_attempts_per_n):
            if time.time() > deadline:
                break
            edges = _generate_random_regular(n, k)
            if edges is None:
                continue
            if _girth_check_fast(n, edges, k, g):
                return n, edges
    return None


def _moore_bound(k, g):
    """Compute the Moore bound for (k, g) cage."""
    if g % 2 == 0:
        r = g // 2
        return k * (k - 1) ** (r - 1) + k * (k - 1) ** r
    else:
        r = (g - 1) // 2
        return 1 + k * sum((k - 1) ** i for i in range(r))


def construct(k, g):
    deadline = time.time() + 115  # 150s limit, leave 35s margin
    
    # Start with LUW construction as guaranteed valid baseline
    n_luw, edges_luw = _luw_construct(k, g)
    
    # Verify it's valid
    if not _girth_check_fast(n_luw, edges_luw, k, g):
        return n_luw, edges_luw
    
    best_n = n_luw
    best_edges = edges_luw[:]
    
    # Compute Moore bound
    mb = _moore_bound(k, g)
    
    # Try random regular graph search
    # Search from Moore bound upward, but cap at best_n - 1
    n_start = max(mb, 20)
    n_end = min(best_n - 1, 400)
    
    if n_start <= n_end:
        result = _try_random_regular_search(k, g, deadline, n_start, n_end)
        if result is not None:
            n_rand, edges_rand = result
            if n_rand < best_n:
                best_n = n_rand
                best_edges = edges_rand[:]
    
    return best_n, best_edges
# EVOLVE-BLOCK-END
