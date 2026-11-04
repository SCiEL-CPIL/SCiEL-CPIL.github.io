"""Program for the suite task: construct a small k-regular graph of girth g."""
import math
import random
import time
from collections import deque

# EVOLVE-BLOCK-START

def _moore_bound(k, g):
    """Return the Moore bound (lower bound on n) for a k-regular graph of girth g."""
    if g % 2 == 0:
        m = g // 2
        s = 0
        term = 1
        for i in range(m):
            s += term
            term *= (k - 1)
        return 1 + k * s
    else:
        m = (g - 1) // 2
        s = 0
        term = 1
        for i in range(m):
            s += term
            term *= (k - 1)
        return 1 + k * s


def _find_all_short_cycles(adj, g, max_cycles=500):
    """Find all cycles of length < g using BFS from each vertex.
    Returns list of (cycle_length, [v0, v1, ..., v_{l-1}]) for short cycles."""
    n = len(adj)
    short_cycles = []
    seen_cycles = set()
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        
        while q:
            u = q.popleft()
            if dist[u] >= g - 1:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and u != parent[v] and dist[u] < dist[v]:
                    # Found a back edge, compute cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        # Reconstruct cycle
                        cycle = [u]
                        cur = u
                        while cur != v:
                            cur = parent[cur]
                            cycle.append(cur)
                        # Add path from v back to start of this branch
                        # Actually, let's just track the cycle nodes
                        # The cycle goes: v -> ... -> u -> (edge u-v)
                        # Let's reconstruct properly
                        path_u = [u]
                        cur = u
                        while cur != start:
                            cur = parent[cur]
                            if cur == -1:
                                break
                            path_u.append(cur)
                        path_v = [v]
                        cur = v
                        while cur != start:
                            cur = parent[cur]
                            if cur == -1:
                                break
                            path_v.append(cur)
                        
                        # The cycle is: start...u - v...start (through the tree)
                        # Simpler: just record the length and the edge (u,v)
                        key = frozenset([u, v])
                        if key not in seen_cycles:
                            seen_cycles.add(key)
                            short_cycles.append((cycle_len, (u, v)))
    
    return short_cycles[:max_cycles]


def _count_short_cycles(adj, g):
    """Count the number of cycles of length < g."""
    n = len(adj)
    count = 0
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        
        while q:
            u = q.popleft()
            if dist[u] >= g - 1:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and u != parent[v] and dist[u] < dist[v]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        count += 1
    return count


def _has_short_cycle(adj, g):
    """Check if graph has any cycle shorter than g. Returns (bool, cycle_length or 0)."""
    n = len(adj)
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        
        while q:
            u = q.popleft()
            if dist[u] >= g - 1:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and u != parent[v] and dist[u] < dist[v]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return True, cycle_len
    return False, 0


def _random_k_regular(n, k, seed=None):
    """Generate a random k-regular graph on n vertices.
    Returns adjacency list or None if impossible."""
    if n * k % 2 != 0 or n < k + 1:
        return None
    
    rng = random.Random(seed)
    
    # Use the configuration model
    # Create stubs: each vertex has k stubs
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    rng.shuffle(stubs)
    
    # Pair up stubs to form edges
    edges = []
    edge_set = set()
    
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            return None  # Self-loop, retry
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return None  # Multi-edge, retry
        edge_set.add(key)
        edges.append((u, v))
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    return adj


def _swap_edges(adj, n, k, edge1, edge2):
    """Swap two edges (a,b) and (c,d) to (a,c) and (b,d) or (a,d) and (b,c).
    Returns new adj if valid, else None."""
    a, b = edge1
    c, d = edge2
    
    # Try all 4 possible reconnections
    for new_edges in [
        [(a, c), (b, d)],
        [(a, d), (b, c)],
    ]:
        # Check validity
        valid = True
        for u, v in new_edges:
            if u == v:
                valid = False
                break
            if v in adj[u]:  # Already an edge
                valid = False
                break
        if not valid:
            continue
        
        # Also check we're not creating multi-edges with the remaining edges
        # Remove old edges, add new ones
        new_adj = [list(lst) for lst in adj]
        
        # Remove old edges
        for u, v in [edge1, edge2]:
            new_adj[u].remove(v)
            new_adj[v].remove(u)
        
        # Add new edges
        for u, v in new_edges:
            new_adj[u].append(v)
            new_adj[v].append(u)
        
        # Verify all degrees are k
        ok = True
        for i in range(n):
            if len(new_adj[i]) != k:
                ok = False
                break
        if not ok:
            continue
        
        # Verify no multi-edges
        for i in range(n):
            if len(new_adj[i]) != len(set(new_adj[i])):
                ok = False
                break
        if not ok:
            continue
        
        return new_adj
    
    return None


def _simulate_annealing(n, k, g, time_limit=100, seed=None):
    """Use simulated annealing to find a k-regular graph of girth >= g on n vertices."""
    rng = random.Random(seed)
    start_time = time.time()
    
    # Initialize with random k-regular graph
    adj = None
    for attempt in range(1000):
        adj = _random_k_regular(n, k, seed=seed + attempt if seed else None)
        if adj is not None:
            break
    if adj is None:
        return None
    
    # Check initial girth
    has_sc, sc_len = _has_short_cycle(adj, g)
    if not has_sc:
        return adj
    
    # Simulated annealing
    T = 1.0
    T_min = 0.001
    cooling = 0.999
    
    current_cost = _count_short_cycles(adj, g)
    best_adj = [list(lst) for lst in adj]
    best_cost = current_cost
    
    # Precompute all edges for quick access
    def get_edges(a):
        edges = []
        seen = set()
        for u in range(n):
            for v in a[u]:
                key = (min(u, v), max(u, v))
                if key not in seen:
                    seen.add(key)
                    edges.append(key)
        return edges
    
    edge_list = get_edges(adj)
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
        
        # Pick two random edges to swap
        if len(edge_list) < 2:
            break
        
        i1 = rng.randint(0, len(edge_list) - 1)
        i2 = rng.randint(0, len(edge_list) - 1)
        if i1 == i2:
            continue
        
        e1 = edge_list[i1]
        e2 = edge_list[i2]
        
        # Check they don't share a vertex (for clean 2-opt)
        if len(set([e1[0], e1[1], e2[0], e2[1]])) < 4:
            continue
        
        new_adj = _swap_edges(adj, n, k, e1, e2)
        if new_adj is None:
            continue
        
        new_cost = _count_short_cycles(new_adj, g)
        
        # Accept or reject
        delta = new_cost - current_cost
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            adj = new_adj
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_adj = [list(lst) for lst in adj]
                if best_cost == 0:
                    break
            # Update edge list
            edge_list = get_edges(adj)
        
        T *= cooling
    
    if best_cost == 0:
        return best_adj
    return None


def _construct_via_annealing(k, g, max_n=None, time_limit=100):
    """Try to construct a k-regular graph of girth >= g using simulated annealing.
    Tries increasing n from the Moore bound upward."""
    start_time = time.time()
    
    mb = _moore_bound(k, g)
    # n must be even for k-regular (sum of degrees = n*k must be even)
    if n % 2 != 0:
        n += 1
    if max_n is None:
        max_n = mb + 500  # reasonable upper bound
    
    # Try various n values
    for n in range(mb, max_n + 1, 2):  # step by 2 to keep n even
        if time.time() - start_time > time_limit:
            break
        
        result = _simulate_annealing(n, k, g, time_limit=min(5, time_limit - (time.time() - start_time)))
        if result is not None:
            # Build edge list
            edges = []
            seen = set()
            for u in range(n):
                for v in result[u]:
                    key = (min(u, v), max(u, v))
                    if key not in seen:
                        seen.add(key)
                        edges.append(key)
            return n, edges
    
    return None


def _construct_lazebnik(k, g):
    """Construct Lazebnik-Ustimenko-Woldar graph as fallback. Returns (n, edges) or None."""
    q = k
    d = g - 5
    if d % 2 == 0:
        d += 1
    if d < 3:
        d = 3
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
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
    return n, edges


def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs."""
    
    if k < 2:
        return 0, []
    
    if g <= 2:
        if k == 2:
            n = max(3, g)
            edges = [(i, (i + 1) % n) for i in range(n)]
            return n, edges
        else:
            n = k + 1
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            return n, edges
    
    # Primary: try simulated annealing
    try:
        result = _construct_via_annealing(k, g, time_limit=90)
        if result is not None:
            n, edges = result
            # Verify
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            has_sc, _ = _has_short_cycle(adj, g)
            if not has_sc:
                return n, edges
    except Exception:
        pass
    
    # Fallback: Lazebnik construction
    try:
        n, edges = _construct_lazebnik(k, g)
        return n, edges
    except Exception:
        return 0, []

# EVOLVE-BLOCK-END
