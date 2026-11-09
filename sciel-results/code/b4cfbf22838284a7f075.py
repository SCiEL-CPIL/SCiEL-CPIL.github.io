"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
# EVOLVE-BLOCK-START
def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    # whose girth is at least d + 5 for odd d. Points and lines are vectors over GF(q) (q prime,
    # or q = 4); the connected component of the zero point is returned.
    # Try smaller d values and verify girth explicitly to find the smallest valid graph.
    q = k
    deadline = time.time() + 140  # safety margin before 150s limit
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    def get_rule(d):
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        return rule
    
    def build_luw(d):
        rule = get_rule(d)
        
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
    
    def check_girth(n, edges, min_girth):
        """Check if graph has no cycle shorter than min_girth using BFS from each vertex."""
        if n == 0:
            return True
        
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        
        # For each vertex, do BFS and check that no shorter path exists to itself
        # through a non-trivial cycle
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            for v in queue:
                for u in adj[v]:
                    if dist[u] == -1:
                        dist[u] = dist[v] + 1
                        parent[u] = v
                        queue.append(u)
                    elif u != parent[v]:
                        # Found a cycle: dist[v] + 1 + dist[u]
                        cycle_len = dist[v] + 1 + dist[u]
                        if cycle_len < min_girth:
                            return False
        return True
    
    # Try d values from smallest to largest, verify girth, return first valid
    # LUW bound: girth >= d + 5 for odd d, so for girth g, we need d >= g - 5
    # But actual girth might be higher, so try smaller d first
    min_d = 3  # smallest reasonable d
    max_d = g + 2  # safe upper bound
    
    best_n = float('inf')
    best_edges = None
    
    for d in range(min_d, max_d + 1):
        if time.time() > deadline:
            break
        try:
            n, edges = build_luw(d)
            if n < best_n:
                # Check if girth is sufficient
                if check_girth(n, edges, g):
                    best_n = n
                    best_edges = edges
                    # If we found a valid graph, try smaller d values too
                    # but continue to see if we can do even better
        except Exception:
            continue
    
    if best_edges is None:
        # Fallback: use the guaranteed bound
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        n, edges = build_luw(d)
        return n, edges
    
    return best_n, best_edges
# EVOLVE-BLOCK-END
