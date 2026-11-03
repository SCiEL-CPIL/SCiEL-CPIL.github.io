"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Try Lazebnik-Ustimenko-Woldar (LW) construction first - explicit, fast
    lw_result = _construct_lw(k, g)
    if lw_result is not None:
        n_lw, edges_lw = lw_result
        # Verify quickly
        if _verify_graph(k, g, n_lw, edges_lw):
            return n_lw, edges_lw
    
    # If LW doesn't work or gives large n, try other constructions
    # Moore bound gives lower limit
    moore_bound = _moore_bound(k, g)
    
    # Try circulant-based constructions
    circ_result = _construct_circulant(k, g, moore_bound)
    if circ_result is not None:
        n_circ, edges_circ = circ_result
        if _verify_graph(k, g, n_circ, edges_circ):
            if lw_result is None or n_circ < n_lw:
                return n_circ, edges_circ
            else:
                return n_lw, edges_lw
    
    # Try random regular graph with girth check
    random_result = _construct_random_regular(k, g, moore_bound, deadline)
    if random_result is not None:
        n_rand, edges_rand = random_result
        if lw_result is None or n_rand < n_lw:
            return n_rand, edges_rand
    
    # Fallback: return LW if we have it
    if lw_result is not None:
        return lw_result
    
    # Last resort: complete graph (won't satisfy girth but returns valid structure)
    n = k + 1
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return n, edges


def _moore_bound(k, g):
    """Compute the Moore bound (lower bound on number of vertices)."""
    if g % 2 == 0:
        # Even girth: n >= k * (1 + k * (k-1)^(g/2 - 2)) for g >= 4
        if g == 2:
            return k + 1
        elif g == 4:
            return k * (k - 1) + 1
        else:
            return k * (1 + k * (k - 1) ** (g // 2 - 2))
    else:
        # Odd girth: n >= 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
        h = (g - 1) // 2
        total = 1
        power = 1
        for i in range(h):
            total += k * power
            power *= (k - 1)
        return total


def _construct_lw(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) construction."""
    q = k
    # d is related to girth: girth >= d + 5 for odd d
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Build rule table
    rule = {}
    for c in range(2, d + 1):
        if c == 2:
            rule[c] = (1, 1)
        elif c == 3:
            rule[c] = (2, 1)
        elif c == 4:
            rule[c] = (1, 2)
        else:
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
            rule[c] = options[r]
    
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
            yield (1 - side, tuple(y))
    
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
    
    n = len(order)
    # Check if graph is k-regular
    degree_count = [0] * n
    for u, v in edges:
        degree_count[u] += 1
        degree_count[v] += 1
    
    if all(d == k for d in degree_count):
        return n, edges
    return None


def _construct_circulant(k, g, min_n):
    """Try circulant graph constructions."""
    # For a circulant graph C(n, S), vertices 0..n-1, edges between i and i+s mod n for s in S
    # Girth depends on the structure of S
    # Try small circulant graphs
    
    for n in range(max(min_n, 2 * k), min(min_n + 200, 3000)):
        if time.time() > time.time() + 5:  # simple time check placeholder
            break
        # Try different connection sets
        # For k-regular circulant, we need |S| = k/2 if k even, with symmetric S
        if k % 2 == 0:
            half_k = k // 2
            # Try simple circulant with steps 1, 2, ..., half_k
            S = list(range(1, half_k + 1))
            edges = []
            for i in range(n):
                for s in S:
                    j = (i + s) % n
                    if i < j:
                        edges.append((i, j))
                    elif j < i and s > n // 2:
                        edges.append((j, i))
            # Deduplicate and check
            edge_set = set()
            unique_edges = []
            for u, v in edges:
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if (u, v) not in edge_set:
                    edge_set.add((u, v))
                    unique_edges.append((u, v))
            
            # Check degree
            deg = [0] * n
            for u, v in unique_edges:
                deg[u] += 1
                deg[v] += 1
            
            if all(d == k for d in deg):
                # Check girth using BFS
                if _check_girth_bfs(n, unique_edges, g):
                    return n, unique_edges
    return None


def _construct_random_regular(k, g, min_n, deadline):
    """Try random regular graph construction with girth verification."""
    start_time = time.time()
    
    # Start from Moore bound and try increasing n
    for n in range(max(min_n, 2 * k), min(min_n + 500, 2000)):
        if time.time() > deadline - 10:
            break
        
        # Try a few random regular graphs
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            edges = _generate_random_regular(n, k)
            if edges is None:
                continue
            
            if _check_girth_bfs(n, edges, g):
                return n, edges
    
    return None


def _generate_random_regular(n, k):
    """Generate a random k-regular graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        return None
    
    # Create stubs
    stubs = []
    for i in range(n):
        stubs.extend([i] * k)
    
    random.shuffle(stubs)
    
    # Pair up stubs to form edges
    edges = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            return None  # Self-loop
        if (min(u, v), max(u, v)) in edges:
            return None  # Multi-edge
        edges.add((min(u, v), max(u, v)))
    
    return list(edges)


def _check_girth_bfs(n, edges, min_girth):
    """Check if graph has girth at least min_girth using BFS from each vertex."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # BFS from each vertex to find shortest cycle
    for start in range(n):
        if time.time() > 145:  # global deadline check
            return False
        
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and dist[v] <= dist[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        return False
    
    return True


def _verify_graph(k, g, n, edges):
    """Verify that the graph is k-regular and has girth at least g."""
    # Check degree
    deg = [0] * n
    for u, v in edges:
        if u == v:
            return False
        deg[u] += 1
        deg[v] += 1
    
    if any(d != k for d in deg):
        return False
    
    # Check no duplicate edges
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        if (u, v) in edge_set:
            return False
        edge_set.add((u, v))
    
    # Check girth
    return _check_girth_bfs(n, edges, g)

# EVOLVE-BLOCK-END
