import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer
    
    # Moore bound for k-regular, girth g
    # n >= 1 + k * sum_{i=0}^{floor((g-1)/2)} (k-1)^i
    if g % 2 == 1:
        r = (g - 1) // 2
        moore_bound = 1 + k * sum((k - 1) ** i for i in range(r))
    else:
        r = g // 2 - 1
        moore_bound = k * sum((k - 1) ** i for i in range(r))
        
    # Start slightly above Moore bound
    # For k=5, g=9, r=4, moore_bound = 1 + 5*(1+4+16+64) = 426
    current_n = max(moore_bound, 10)
    
    best_result = None
    
    while time.time() < deadline:
        # Try to construct a graph with current_n vertices
        result = try_construct(k, g, current_n)
        if result is not None:
            n, edges = result
            if best_result is None or n < best_result[0]:
                best_result = (n, edges)
                # If we found a graph close to the bound, we can return early?
                # Or try to find a smaller one?
                # Since we are increasing n, the first valid one is the smallest for this search order.
                # But maybe a larger n allows a faster construction? No, we want min n.
                # So if we find one at n, we should check if we can find one at a smaller n?
                # We started at moore_bound and increased. So the first success is the min n found so far.
                # We can return immediately if we are confident?
                # But maybe we can find a smaller one if we started too high?
                # We started at moore_bound. So the first success is the minimal n in this sequence.
                return result
        
        # If failed, increase n
        # Increase by a small step to allow more attempts for smaller n?
        # Or jump?
        # For girth 9, the density of valid edges drops rapidly.
        # Let's increase by 5.
        current_n += 5
        
    # If we didn't find anything, return the seed result (L-U-W) as fallback
    # But we must return *something* valid.
    # The seed program logic:
    return _seed_construct(k, g)

def try_construct(k, g, n):
    # Greedy randomized construction
    # 1. Init degrees
    degrees = [0] * n
    adj = [[] for _ in range(n)]
    edges = []
    
    # Target edges
    total_edges = n * k // 2
    
    # To speed up, we can pre-shuffle potential edges?
    # No, just random sampling.
    
    # We need to check distance < g-1
    # BFS depth limit = g - 2
    
    # Optimization: If n is small, BFS is fast.
    
    attempts = 0
    max_attempts = 50000 # Limit attempts per n to avoid hanging
    
    while len(edges) < total_edges and attempts < max_attempts:
        # Find vertices with degree < k
        # To avoid O(n) scan every time, maintain a list of available vertices?
        # Simple scan is O(n). n ~ 500. 500 * 50000 = 2.5M ops. Fine.
        
        # Pick u
        available_u = [i for i in range(n) if degrees[i] < k]
        if not available_u:
            break
        u = random.choice(available_u)
        
        # Pick v
        available_v = [i for i in range(n) if degrees[i] < k and i != u]
        if not available_v:
            break
        v = random.choice(available_v)
        
        # Check if adding (u,v) creates a cycle of length < g
        # This happens if there is a path of length < g-1 between u and v
        # i.e., dist(u,v) <= g-2
        
        if not has_short_path(adj, u, v, g - 2):
            # Safe to add
            adj[u].append(v)
            adj[v].append(u)
            degrees[u] += 1
            degrees[v] += 1
            edges.append((u, v))
            attempts = 0 # Reset attempts on success? Or keep counting?
        else:
            attempts += 1
            
    if len(edges) == total_edges:
        # Verify girth exactly?
        # The greedy check ensures no cycle < g is *created* by the new edge.
        # Since we start with empty graph (girth infinity), all cycles are formed by the last edge added.
        # So if we never added an edge that closed a short cycle, the graph has girth >= g.
        # However, we must ensure the graph is simple (no multi-edges).
        # The check `i != u` handles self-loops.
        # Multi-edges: We check `has_short_path`. If u and v are already connected by an edge, dist is 1.
        # 1 <= g-2 (for g>=3). So it would be rejected.
        # So the graph is simple.
        
        return n, edges
    else:
        return None

def has_short_path(adj, u, v, max_depth):
    # BFS from u to see if v is reachable within max_depth
    # Returns True if dist(u,v) <= max_depth
    if u == v:
        return True # dist 0
    
    visited = set([u])
    current_level = [u]
    
    for depth in range(1, max_depth + 1):
        next_level = []
        for node in current_level:
            for neighbor in adj[node]:
                if neighbor == v:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_level.append(neighbor)
        current_level = next_level
        if not current_level:
            break
            
    return False

def _seed_construct(k, g):
    # Fallback to seed logic
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
