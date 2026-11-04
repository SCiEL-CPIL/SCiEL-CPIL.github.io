"""New approach: random 3-regular graph generation with short-cycle repair."""
import random
import itertools
from collections import deque

def _random_regular_graph(n, k, rng):
    """Generate a random k-regular graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        return None
    # Build stub list
    stubs = []
    for v in range(n):
        for _ in range(k):
            stubs.append(v)
    rng.shuffle(stubs)
    # Pair up stubs
    edges = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            return None  # self-loop, retry
        if (u, v) in edges or (v, u) in edges:
            return None  # multi-edge, retry
        edges.add((min(u, v), max(u, v)))
    return list(edges)

def _find_short_cycle(adj, max_len):
    """Find a cycle of length <= max_len starting from any vertex."""
    n = len(adj)
    for start in range(n):
        # BFS with path tracking, looking for cycles of length <= max_len
        # Use DFS with depth limit
        stack = [(start, [start])]
        visited = set()
        while stack:
            node, path = stack.pop()
            if len(path) > max_len:
                continue
            for nb in adj[node]:
                if nb == path[0] and len(path) >= 3:
                    # Found a cycle
                    return path + [nb]
                if nb not in path and len(path) < max_len:
                    stack.append((nb, path + [nb]))
    return None

def _edge_swap(adj, e1, e2, n):
    """Try to swap edges to break a cycle. Returns new adj or None."""
    u1, v1 = e1
    u2, v2 = e2
    # Remove e1 and e2, add (u1,u2) and (v1,v2) or (u1,v2) and (v1,u2)
    # Check both options
    for (a, b), (c, d) in [((u1, u2), (v1, v2)), ((u1, v2), (v1, u2))]:
        if a == b or c == d:
            continue
        # Check if new edges already exist
        if (min(a,b), max(a,b)) in adj.get(a, set()) or (min(c,d), max(c,d)) in adj.get(c, set()):
            continue
        # Check degrees stay k
        # Remove old edges
        adj2 = {v: set(adj[v]) for v in adj}
        adj2[u1].discard(v1)
        adj2[v1].discard(u1)
        adj2[u2].discard(v2)
        adj2[v2].discard(u2)
        # Add new edges
        adj2[a].add(b)
        adj2[b].add(a)
        adj2[c].add(d)
        adj2[d].add(c)
        # Verify regularity
        if all(len(adj2[v]) == 3 for v in range(n)):
            return adj2
    return None

def _has_short_cycle(adj, girth):
    """Check if graph has any cycle shorter than girth."""
    n = len(adj)
    for start in range(n):
        # BFS to detect cycles shorter than girth
        # Use a more efficient method: check all cycles up to girth-1
        # For each vertex, do limited DFS
        visited_depth = {}
        def dfs(node, depth, path):
            if depth >= girth - 1:
                return False
            for nb in adj[node]:
                if nb == path[0] and depth >= 2:
                    return True
                if nb not in path:
                    if dfs(nb, depth + 1, path + [nb]):
                        return True
            return False
        if dfs(start, 0, [start]):
            return True
    return False

def construct(k, g):
    """Construct a k-regular graph with girth >= g."""
    if k != 3:
        # Fallback to a simple construction for other k
        # Use the original Lazebnik-Ustimenko-Woldar approach
        return _construct_luw(k, g)
    
    girth = g
    rng = random.Random(42)
    
    # Try various sizes starting from Moore bound
    # Moore bound for 3-regular, girth g=16 (odd, r=8): 1 + 3*(2^7 - 1)/(2-1) = 1 + 3*127 = 382
    # But we need to be realistic. Try sizes from ~400 to 900.
    
    best_n = None
    best_edges = None
    
    for attempt in range(200):
        # Choose a target size
        # Start small, increase if we can't find valid graphs
        base_size = 380 + attempt * 5
        if base_size > 930:
            break
        
        n = base_size
        # n must be even for 3-regular
        if n % 2 != 0:
            n += 1
        
        # Try to generate a valid graph
        for trial in range(50):
            edges = _random_regular_graph(n, 3, rng)
            if edges is None:
                continue
            
            # Build adjacency
            adj = {v: set() for v in range(n)}
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            
            # Check girth
            if _has_short_cycle(adj, girth):
                # Try to repair by edge swaps
                adj = _repair_short_cycles(adj, girth, n, rng, max_swaps=100)
                if adj is None:
                    continue
                # Re-check
                if _has_short_cycle(adj, girth):
                    continue
            
            # Valid!
            if best_n is None or n < best_n:
                best_n = n
                best_edges = [(min(u, v), max(u, v)) for u, v in adj.items() for v in adj[u] if u < v]
            
            # Try to find even smaller
            break
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: use the original construction
    return _construct_luw(k, g)

def _repair_short_cycles(adj, girth, n, rng, max_swaps=100):
    """Try to repair short cycles via edge swaps."""
    for _ in range(max_swaps):
        # Find a short cycle
        cycle = _find_short_cycle(adj, girth - 1)
        if cycle is None:
            return adj
        
        # The cycle is a list of vertices [v0, v1, ..., v_{m-1}, v0]
        # Try to swap one edge in the cycle with a non-cycle edge
        cycle_edges = set()
        for i in range(len(cycle) - 1):
            u, v = cycle[i], cycle[i + 1]
            cycle_edges.add((min(u, v), max(u, v)))
        
        # Pick a random edge in the cycle
        ce = list(cycle_edges)
        rng.shuffle(ce)
        
        repaired = False
        for e1 in ce:
            # Find a non-adjacent edge to swap with
            # e1 = (u1, v1), find e2 = (u2, v2) not sharing vertices with e1
            u1, v1 = e1
            candidates = []
            for u2 in range(n):
                if u2 in (u1, v1):
                    continue
                for v2 in adj[u2]:
                    if v2 in (u1, v1):
                        continue
                    if v2 < u2:
                        continue
                    e2 = (min(u2, v2), max(u2, v2))
                    if e2 in cycle_edges:
                        continue
                    candidates.append(e2)
            
            if not candidates:
                continue
            
            e2 = rng.choice(candidates)
            new_adj = _edge_swap(adj, e1, e2, n)
            if new_adj is not None:
                adj = new_adj
                repaired = True
                break
        
        if not repaired:
            return None
    
    return adj

def _construct_luw(k, g):
    """Original Lazebnik-Ustimenko-Woldar construction as fallback."""
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
