"""Rewritten program: LUW q=4 construction + girth-preserving perfect matching."""
import time
import math
import random
from collections import deque

# EVOLVE-BLOCK-START

def _moore_bound(k, g):
    """Lower bound on number of vertices for a k-regular graph of girth g."""
    if g % 2 == 1:
        r = (g - 3) // 2
        if k == 1:
            return 2
        total = 1 + k
        for i in range(1, r + 1):
            total += k * (k - 1) ** i
        return total
    else:
        r = g // 2 - 1
        if k == 1:
            return 2
        total = 0
        for i in range(r + 1):
            total += (k - 1) ** i
        return 2 * total


def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _verify_girth(n, edges, k, g):
    """Verify every vertex has degree k, graph is simple, and girth >= g."""
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        if u >= n or v >= n:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
        adj[u].append(v)
        adj[v].append(u)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        found_short = False
        while queue:
            u = queue.popleft()
            if dist[u] >= g:
                break
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    if dist[v] < g:
                        queue.append(v)
                else:
                    if dist[u] + dist[v] + 1 < g:
                        found_short = True
                        break
            if found_short:
                break
        if found_short:
            return False
    return True


def _hopcroft_karp(left_size, right_size, adj_left):
    """Hopcroft-Karp bipartite matching."""
    INF = float('inf')
    pair_u = [-1] * left_size
    pair_v = [-1] * right_size
    dist = [0] * left_size

    def bfs():
        queue = deque()
        for u in range(left_size):
            if pair_u[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = INF
        found = False
        while queue:
            u = queue.popleft()
            for v in adj_left[u]:
                if pair_v[v] == -1:
                    found = True
                elif dist[pair_v[v]] == INF:
                    dist[pair_v[v]] = dist[u] + 1
                    queue.append(pair_v[v])
        return found

    def dfs(u):
        for v in adj_left[u]:
            if pair_v[v] == -1 or (dist[pair_v[v]] == dist[u] + 1 and dfs(pair_v[v])):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = INF
        return False

    matching = 0
    while bfs():
        for u in range(left_size):
            if pair_u[u] == -1:
                if dfs(u):
                    matching += 1
    return matching, pair_u


def _compute_distances(adj, max_dist):
    """Compute distances up to max_dist from every vertex."""
    n = len(adj)
    dist = [[max_dist + 1] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
        queue = deque([i])
        while queue:
            u = queue.popleft()
            if dist[i][u] >= max_dist:
                continue
            for v in adj[u]:
                if dist[i][v] > dist[i][u] + 1:
                    dist[i][v] = dist[i][u] + 1
                    if dist[i][v] < max_dist:
                        queue.append(v)
    return dist


def _luw_q4(d, q):
    """Lazebnik-Ustimenko-Woldar construction CD(d, q) with q=4 (GF(4))."""
    mul_table = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 2, 3],
        [0, 3, 3, 2],
    ]
    add_table = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0],
    ]

    def add(x, y):
        return add_table[x][y]

    def sub(x, y):
        return add_table[x][y]

    def mul(x, y):
        return mul_table[x][y]

    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        pattern = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = pattern[r]

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
    index = {start: 0}
    order = [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges, index, order


def _luw_q5(d, q):
    """LUW construction over prime field F5."""
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        pattern = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = pattern[r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = (x[c] + y[a] * x[b]) % q
                else:
                    y[c] = (x[c] - x[a] * y[b]) % q
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges, index, order


def _general_luw(d, q):
    """General LUW construction over prime field Fq."""
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        pattern = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = pattern[r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = (x[c] + y[a] * x[b]) % q
                else:
                    y[c] = (x[c] - x[a] * y[b]) % q
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges, index, order


def _next_prime(n):
    if n <= 2:
        return 2
    if n % 2 == 0:
        n += 1
    else:
        n += 2
    while True:
        is_prime = True
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                is_prime = False
                break
        if is_prime:
            return n
        n += 2


def _random_regular_graph(n, k):
    """Generate a random k-regular simple graph on n vertices, or None."""
    if n * k % 2 != 0:
        return None
    if k >= n:
        return None
    if k == 0:
        return []
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    random.shuffle(stubs)
    edges = []
    edge_set = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            return None
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return None
        edge_set.add(key)
        edges.append((u, v))
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(d != k for d in deg):
        return None
    return edges


def _try_luw_with_matching(k, g, d, q, use_gf4=False, deadline=None):
    """Try LUW construction and add a perfect matching to increase degree by 1."""
    if use_gf4:
        n_luw, edges_luw, index, order = _luw_q4(d, q)
    else:
        n_luw, edges_luw, index, order = _general_luw(d, q)
    
    if n_luw < 2 or n_luw > 50000:
        return None
    
    # Check current degree
    adj_luw = _build_adj(n_luw, edges_luw)
    for i in range(n_luw):
        if len(adj_luw[i]) != k - 1:
            return None
    
    # Separate into two sides
    side0 = [i for i, v in enumerate(order) if v[0] == 0]
    side1 = [i for i, v in enumerate(order) if v[0] == 1]
    
    if len(side0) != len(side1):
        return None
    
    left_size = len(side0)
    right_size = len(side1)
    
    # Compute distances
    dist = _compute_distances(adj_luw, g)
    
    # Build bipartite graph
    adj_left = [[] for _ in range(left_size)]
    for i, u in enumerate(side0):
        for j, v in enumerate(side1):
            if dist[u][v] >= g:
                adj_left[i].append(j)
    
    matching_size, pair_u = _hopcroft_karp(left_size, right_size, adj_left)
    
    if matching_size == left_size:
        new_edges = list(edges_luw)
        for i, u in enumerate(side0):
            v = side1[pair_u[i]]
            new_edges.append((u, v))
        if _verify_girth(n_luw, new_edges, k, g):
            return n_luw, new_edges
    
    return None


def _try_luw_direct(k, g, d, q, deadline=None):
    """Try LUW construction directly for degree k."""
    if q == 4:
        n_luw, edges_luw, index, order = _luw_q4(d, q)
    else:
        n_luw, edges_luw, index, order = _general_luw(d, q)
    
    if n_luw < 2 or n_luw > 50000:
        return None
    
    adj_luw = _build_adj(n_luw, edges_luw)
    for i in range(n_luw):
        if len(adj_luw[i]) != k:
            return None
    
    if _verify_girth(n_luw, edges_luw, k, g):
        return n_luw, edges_luw
    
    return None


def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 145

    mb = _moore_bound(k, g)
    
    # Strategy 1: For k=5, g=9, try LUW q=4 d=4 + matching
    if k == 5 and g == 9:
        # LUW q=4, d=4 gives 4-regular graph, girth 10
        # Add matching to get 5-regular
        result = _try_luw_with_matching(5, 9, d=4, q=4, use_gf4=True, deadline=deadline)
        if result:
            return result
        
        # Try LUW q=5, d=4 (should give 5-regular directly if girth works)
        result = _try_luw_direct(5, 9, d=4, q=5, deadline=deadline)
        if result:
            return result
        
        # Try LUW q=5, d=5
        result = _try_luw_direct(5, 9, d=5, q=5, deadline=deadline)
        if result:
            return result
        
        # Try LUW q=7, d=4
        result = _try_luw_direct(5, 9, d=4, q=7, deadline=deadline)
        if result:
            return result
        
        # Try LUW q=7, d=5
        result = _try_luw_direct(5, 9, d=5, q=7, deadline=deadline)
        if result:
            return result
    
    # Strategy 2: For other k values, try various LUW constructions
    if k >= 2 and k <= 15:
        # Try LUW with q=k if k is prime
        if k >= 3:
            is_prime = True
            if k < 2:
                is_prime = False
            else:
                for i in range(2, int(math.sqrt(k)) + 1):
                    if k % i == 0:
                        is_prime = False
                        break
            
            if is_prime:
                # For girth g, we need d such that girth(CD(d,q)) >= g
                # girth is d+5 for odd d, d+6 for even d
                for d in range(3, 10):
                    if time.time() > deadline:
                        break
                    expected_girth = d + 5 if d % 2 == 1 else d + 6
                    if expected_girth >= g:
                        result = _try_luw_direct(k, g, d=d, q=k, deadline=deadline)
                        if result:
                            return result
        
        # Try LUW with q = next prime >= k
        q = _next_prime(k)
        if q <= 50:
            for d in range(3, 10):
                if time.time() > deadline:
                    break
                expected_girth = d + 5 if d % 2 == 1 else d + 6
                if expected_girth >= g:
                    result = _try_luw_direct(k, g, d=d, q=q, deadline=deadline)
                    if result:
                        return result
        
        # Try LUW with q=4 (GF(4)) + matching for k=5
        if k == 5:
            for d in range(3, 8):
                if time.time() > deadline:
                    break
                result = _try_luw_with_matching(5, g, d=d, q=4, use_gf4=True, deadline=deadline)
                if result:
                    return result
    
    # Strategy 3: For small k and g, try random search
    if k <= 4 and g <= 10:
        random.seed(42)
        for trial_n in range(mb, min(mb + 200, 500)):
            if time.time() > deadline:
                break
            for attempt in range(30):
                if time.time() > deadline:
                    break
                try:
                    edges = _random_regular_graph(trial_n, k)
                    if edges is not None:
                        if _verify_girth(trial_n, edges, k, g):
                            return trial_n, edges
                except Exception:
                    pass
    
    # Fallback: return Moore bound with empty edges (will fail verification but is a valid return)
    return mb, []

# EVOLVE-BLOCK-END
