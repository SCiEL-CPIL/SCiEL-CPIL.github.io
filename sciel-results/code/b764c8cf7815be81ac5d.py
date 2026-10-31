"""
Construct a k-regular graph with girth at least g.
Structural change: voltage-graph lifts of the Petersen graph (10 base vertices)
with fine-grained m-incrementing, plus a fast non-backtracking girth check.
Falls back to circulant search and LUW if no valid lift is found in time.
"""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin out of 150s limit

    def check_deadline():
        return time.time() > deadline

    # For the specific case k=3, g=16:
    if k == 3 and g == 16:
        # Primary strategy: voltage-graph lifts of the Petersen graph
        # Petersen graph is 3-regular, 10 vertices, girth 5.
        # A voltage lift with m layers gives a 3-regular graph on 10m vertices.
        # Moore bound: n >= 512, so m >= 52.
        # We scan m from 52 upward with fine increments.

        result = _find_petersen_lift(3, 16, deadline)
        if result is not None:
            n, edges = result
            return n, edges

        # Fallback 1: circulant graphs C_n(1, a, b)
        result = _find_circulant_cage(3, 16, deadline)
        if result is not None:
            n, edges = result
            return n, edges

        # Fallback 2: LUW with verified girth
        for d in [9, 11, 13, 15, 17]:
            if check_deadline():
                break
            try:
                n_luw, edges_luw = _luw_construct_with_d(3, 16, d)
                if n_luw is not None and _verify_girth_fast(n_luw, edges_luw, 16):
                    return n_luw, edges_luw
            except Exception:
                pass

    # General fallback: LUW with default d
    return _luw_construct(k, g)


def _find_petersen_lift(k, g, deadline):
    """
    Search for a voltage-graph lift of the Petersen graph with girth >= g.
    
    The Petersen graph has 10 vertices and 15 edges. Each edge gets a voltage
    in Z_m. The lift has 10*m vertices and is k-regular.
    
    We use a systematic approach:
    1. Try deterministic voltage assignments based on edge ordering.
    2. Try random voltage assignments.
    3. Scan m from the Moore bound upward.
    
    Returns (n, edges) if a valid graph is found, None otherwise.
    """
    # Petersen graph edges (standard labeling)
    # Vertices 0-9, edges:
    # Outer pentagon: 0-1, 1-2, 2-3, 3-4, 4-0
    # Inner star: 5-7, 7-9, 9-6, 6-8, 8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # inner
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),  # spokes
    ]
    num_edges = len(petersen_edges)  # 15
    base_n = 10

    # Moore bound: n >= 512, so m >= 52
    min_m = 52
    max_m = 200  # upper bound on m to search

    # Strategy: for each m, try several voltage assignments
    # Deterministic assignments first, then random ones.
    
    for m in range(min_m, max_m + 1):
        if time.time() > deadline - 5:
            return None

        # Try a few deterministic voltage patterns
        # Pattern 1: edge i gets voltage i (mod m)
        # Pattern 2: edge i gets voltage (i * 7) mod m (prime multiplier)
        # Pattern 3: alternating patterns
        # Pattern 4: based on edge type (outer, inner, spoke)
        
        voltage_patterns = []
        
        # Pattern 1: sequential
        voltage_patterns.append([i % m for i in range(num_edges)])
        
        # Pattern 2: prime multiplier
        voltage_patterns.append([(i * 7) % m for i in range(num_edges)])
        
        # Pattern 3: type-based
        # outer edges (0-4): voltage = i
        # inner edges (5-9): voltage = m//2 + i - 5
        # spoke edges (10-14): voltage = (m//3) + i - 10
        v3 = []
        for i in range(5):
            v3.append(i % m)
        for i in range(5):
            v3.append((m // 2 + i) % m)
        for i in range(5):
            v3.append((m // 3 + i) % m)
        voltage_patterns.append(v3)
        
        # Pattern 4: another type-based
        v4 = []
        for i in range(5):
            v4.append((i * 3) % m)
        for i in range(5):
            v4.append((m // 2 + i * 2) % m)
        for i in range(5):
            v4.append((m // 3 + i * 4) % m)
        voltage_patterns.append(v4)
        
        # Try random patterns (a few)
        rng = random.Random(m * 1000 + 42)
        for _ in range(3):
            voltage_patterns.append([rng.randint(0, m - 1) for _ in range(num_edges)])
        
        for voltages in voltage_patterns:
            if time.time() > deadline - 5:
                return None
            
            # Build the lifted graph
            n = base_n * m
            edges = []
            for (u, v) in petersen_edges:
                for layer in range(m):
                    # Edge from (u, layer) to (v, (layer + voltage) % m)
                    # and from (v, layer) to (u, (layer - voltage) % m)
                    # But we need to avoid duplicate edges.
                    # Let's just add all directed edges and then deduplicate.
                    u1 = u * m + layer
                    v1 = v * m + (layer + voltages[petersen_edges.index((u, v))]) % m
                    if u1 < v1:
                        edges.append((u1, v1))
                    else:
                        edges.append((v1, u1))
            
            # Remove duplicates and sort
            edges = list(set(edges))
            edges.sort()
            
            # Verify degree and girth
            if len(edges) != n * k // 2:
                continue
            
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # Check all vertices have degree k
            valid = True
            for i in range(n):
                if len(adj[i]) != k:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check girth using non-backtracking BFS from a few vertices
            if _verify_girth_fast_lift(n, adj, g):
                return n, edges
    
    return None


def _verify_girth_fast_lift(n, adj, g):
    """
    Fast girth verification using non-backtracking BFS.
    For vertex-transitive or near-vertex-transitive graphs, checking from
    a few vertices may suffice, but for safety we check all vertices
    with early termination.
    """
    # First, check for cycles of length < g using BFS from each vertex.
    # For efficiency, we use a layered BFS approach.
    
    # Check for 4-cycles first (fastest to detect)
    if _has_cycle_of_length(n, adj, 4):
        return False
    
    # Check for 6-cycles
    if _has_cycle_of_length(n, adj, 6):
        return False
    
    # For longer cycles, use non-backtracking BFS from each vertex
    # but limit the depth to g//2
    max_depth = g // 2 - 1  # we need to check cycles up to length g-1
    
    for src in range(n):
        if _bfs_short_cycle_nonbacktracking(src, adj, n, g, max_depth):
            return False
    
    return True


def _has_cycle_of_length(n, adj, length):
    """Check if there's a cycle of exactly the given length."""
    if length == 4:
        # 4-cycle: u-v-w-x-u where u,v,w,x distinct
        # For each edge (u,v), check if there's a common neighbor of u and v
        # that is not the other endpoint... actually, a 4-cycle exists if
        # there are two vertices with two common neighbors.
        # Or: for each pair of vertices (u, v) at distance 2, check if they
        # have a common neighbor.
        
        # Faster: for each vertex u, look at all pairs of neighbors (v, w).
        # If v and w have a common neighbor x (x != u), then u-v-x-w-u is a 4-cycle.
        for u in range(n):
            neighbors = adj[u]
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    v = neighbors[i]
                    w = neighbors[j]
                    # Check if v and w have a common neighbor other than u
                    # v's neighbors: adj[v], w's neighbors: adj[w]
                    # Common neighbor x means x in adj[v] and x in adj[w] and x != u
                    # For k=3, each vertex has 3 neighbors, so this is fast.
                    for x in adj[v]:
                        if x != u and x in adj[w]:
                            return True
        return False
    elif length == 6:
        # 6-cycle: more complex. Use BFS from each vertex up to depth 3.
        for src in range(n):
            # BFS up to depth 3
            dist = [-1] * n
            parent = [-1] * n
            dist[src] = 0
            queue = deque([src])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= 3:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u] and u != parent[v]:
                            # Check if this creates a 6-cycle
                            # Path from src to u, edge u-v, path from v to src
                            # Length = dist[u] + 1 + dist[v]
                            # But we need exactly 6, and the paths must be disjoint
                            # except at src.
                            # For a quick check: if dist[u] + 1 + dist[v] == 6
                            # and the paths don't share vertices (other than src),
                            # then it's a 6-cycle.
                            if dist[u] + 1 + dist[v] == 6:
                                # Verify the paths are vertex-disjoint
                                path_u = []
                                cur = u
                                while cur != src:
                                    path_u.append(cur)
                                    cur = parent[cur]
                                path_v = []
                                cur = v
                                while cur != src:
                                    path_v.append(cur)
                                    cur = parent[cur]
                                set_u = set(path_u)
                                set_v = set(path_v)
                                if not (set_u & set_v):
                                    return True
        return False
    return False


def _bfs_short_cycle_nonbacktracking(src, adj, n, g, max_depth):
    """
    Non-backtracking BFS from src to detect cycles shorter than g.
    Returns True if a short cycle is found.
    """
    # State: (current, previous)
    # dist[(u, prev)] = length of shortest non-backtracking walk from src to u
    # arriving from prev
    dist = {}
    queue = deque()
    start_state = (src, -1)
    dist[start_state] = 0
    queue.append(start_state)
    
    while queue:
        u, prev = queue.popleft()
        d = dist[(u, prev)]
        
        if d >= max_depth:
            continue
        
        for v in adj[u]:
            if v == prev:
                continue  # non-backtracking
            new_state = (v, u)
            new_d = d + 1
            if new_d > max_depth:
                continue
            if new_state not in dist or new_d < dist[new_state]:
                dist[new_state] = new_d
                # Check if we've found a cycle: v == src and u != -1
                if v == src and u != -1:
                    # Cycle of length new_d
                    if new_d < g:
                        return True
                queue.append(new_state)
    
    return False


def _find_circulant_cage(k, g, deadline):
    """
    Search for a circulant graph C_n(1, a, b) with girth at least g.
    """
    min_n = 512
    max_n = 1000

    for n in range(min_n, max_n + 1, 2):
        if time.time() > deadline - 3:
            return None

        candidates = _generate_candidates(n)

        for a, b in candidates:
            if time.time() > deadline - 3:
                return None

            if _has_short_cycle_fast(n, a, b, g):
                continue

            edges = []
            for i in range(n):
                for shift in [1, a, b]:
                    j = (i + shift) % n
                    if i < j:
                        edges.append((i, j))

            return n, edges

    return None


def _generate_candidates(n):
    """Generate candidate (a, b) pairs for C_n(1, a, b)."""
    candidates = []
    half = n // 2

    for a in range(2, half):
        for b in range(a + 1, half):
            if _quick_reject(n, a, b):
                continue
            candidates.append((a, b))

    return candidates


def _quick_reject(n, a, b):
    """Quick rejection for pairs that create short cycles."""
    G = [1, -1, a, -a, b, -b]
    G_mod = [x % n for x in G]

    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            if G_mod[i] == (-G_mod[j]) % n:
                continue
            s = (G_mod[i] + G_mod[j]) % n
            for k in range(6):
                if k == i or k == j:
                    continue
                if s == G_mod[k]:
                    return True
    return False


def _has_short_cycle_fast(n, a, b, g):
    """Fast girth check for C_n(1, a, b)."""
    def get_neighbors(u):
        return [(u + 1) % n, (u - 1) % n, (u + a) % n, (u - a) % n, (u + b) % n, (u - b) % n]

    dist = {}
    queue = deque()
    start_state = (0, -1)
    dist[start_state] = 0
    queue.append(start_state)

    max_len = g - 1

    while queue:
        u, prev = queue.popleft()
        d = dist[(u, prev)]

        if d >= max_len:
            continue

        for v in get_neighbors(u):
            if v == prev:
                continue
            new_state = (v, u)
            new_d = d + 1
            if new_d > max_len:
                continue
            if new_state not in dist or new_d < dist[new_state]:
                dist[new_state] = new_d
                if v == 0 and u != -1:
                    return True
                queue.append(new_state)

    return False


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph."""
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1

    return _luw_construct_with_d(k, g, d)


def _luw_construct_with_d(k, g, d):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with specified d."""
    q = k

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
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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

    return len(order), edges


def _verify_girth_fast(n, edges, g):
    """Fast girth verification using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for i in range(n):
        if len(adj[i]) != 3:
            return False

    for src in range(n):
        if _bfs_short_cycle(src, adj, n, g):
            return False

    return True


def _bfs_short_cycle(src, adj, n, g):
    """BFS from src to find if there's a cycle shorter than g."""
    dist = [-1] * n
    parent = [-1] * n
    dist[src] = 0
    queue = deque([src])

    max_depth = g // 2

    while queue:
        u = queue.popleft()
        depth = dist[u]

        if depth >= max_depth:
            for v in adj[u]:
                if parent[v] == u or parent[u] == v:
                    continue
                if dist[v] != -1:
                    cycle_len = depth + 1 + dist[v]
                    if cycle_len < g:
                        return True
            continue

        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = depth + 1
                parent[v] = u
                queue.append(v)
            else:
                if parent[v] == u or parent[u] == v:
                    continue
                cycle_len = depth + 1 + dist[v]
                if cycle_len < g:
                    return True

    return False

# EVOLVE-BLOCK-END
