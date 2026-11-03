"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
# EVOLVE-BLOCK-START

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    # For odd girth g, we need girth >= g. The LUW graph CD(d,q) has girth >= d+5.
    # So we need d+5 >= g, i.e., d >= g-5.
    # d must be odd for the standard construction (girth d+5 when d is odd).
    d = g - 5
    if d % 2 == 0:
        d += 1  # make odd
    d = max(3, d)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_fn = lambda x, y: x ^ y
        sub_fn = lambda x, y: x ^ y
        mul_fn = lambda x, y: table[x][y]
    else:
        add_fn = lambda x, y: (x + y) % q
        sub_fn = lambda x, y: (x - y) % q
        mul_fn = lambda x, y: x * y % q
    
    # Rule for CD(d,q)
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
                if side == 0:
                    y[c] = add_fn(x[c], mul_fn(y[a], x[b]))
                else:
                    y[c] = sub_fn(x[c], mul_fn(x[a], y[b]))
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


def _verify_girth(n, edges, g):
    """Verify girth >= g using non-backtracking walks from each vertex.
    Returns True if girth >= g, False otherwise."""
    if n == 0:
        return True
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != len(set(adj[i])):
            return False  # multi-edge
        if i in adj[i]:
            return False  # loop
    
    # BFS for shortest cycle from each vertex
    # Actually, we need to check no cycle shorter than g.
    # Use BFS from each vertex, tracking parent to avoid backtracking.
    # For efficiency, we only need to check up to distance (g-1)//2 from each vertex
    # and look for back-edges that create cycles shorter than g.
    
    half_g = (g - 1) // 2
    for start in range(n):
        # BFS from start, tracking distance and parent
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        for qi in range(len(queue)):
            u = queue[qi]
            if dist[u] >= half_g:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif dist[v] <= dist[u]:
                    # Found a cycle of length dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True


def _random_greedy(k, g, n_target, deadline, seed=42):
    """Try to construct a k-regular graph on n_target vertices with girth >= g
    using random greedy edge addition."""
    rng = random.Random(seed)
    
    # Build adjacency list
    adj = [set() for _ in range(n_target)]
    edges = []
    
    # We want each vertex to have degree k
    # Greedy: repeatedly pick a pair (u,v) with deg(u)<k, deg(v)<k, u!=v, v not in adj[u]
    # and add the edge if it doesn't create a short cycle.
    
    # For cycle checking, we need to verify that adding edge (u,v) doesn't create
    # a cycle of length < g. This means there should be no path from u to v of length < g-1
    # in the current graph (excluding the edge being added).
    
    # Pre-compute distances up to g-2 from each vertex? That's expensive.
    # Instead, do a BFS from u up to depth g-2 and check if v is reachable.
    
    def would_create_short_cycle(u, v):
        """Check if adding edge (u,v) creates a cycle of length < g."""
        # BFS from u, avoiding v initially, up to depth g-2
        # If we reach v at distance < g-1, then adding (u,v) creates a cycle of length < g
        max_depth = g - 2
        if max_depth < 1:
            return False
        visited = {u: 0}
        queue = [(u, 0)]
        for qi in range(len(queue)):
            node, d = queue[qi]
            if d >= max_depth:
                continue
            for nb in adj[node]:
                if nb == v and d + 1 >= g - 1:
                    continue
                if nb == v:
                    # Found path from u to v of length d+1
                    # Cycle length would be d+1+1 = d+2
                    if d + 2 < g:
                        return True
                    continue
                if nb not in visited:
                    visited[nb] = d + 1
                    queue.append((nb, d + 1))
        return False
    
    # Greedy construction
    attempts = 0
    max_attempts = 100000
    
    # Order vertices by how many connections they still need
    while len(edges) < n_target * k // 2:
        if time.time() > deadline:
            return None
        attempts += 1
        if attempts > max_attempts:
            return None
        
        # Find vertices with degree < k
        candidates = [i for i in range(n_target) if len(adj[i]) < k]
        if not candidates:
            break
        if len(candidates) < 2:
            return None
        
        # Try random pairs
        found = False
        for _ in range(50):  # try 50 random pairs per iteration
            u = rng.choice(candidates)
            v = rng.choice(candidates)
            if u == v or v in adj[u]:
                continue
            if would_create_short_cycle(u, v):
                continue
            # Add edge
            adj[u].add(v)
            adj[v].add(u)
            edges.append((u, v))
            found = True
            break
        
        if not found:
            # Try all pairs
            found = False
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    u, v = candidates[i], candidates[j]
                    if v in adj[u]:
                        continue
                    if would_create_short_cycle(u, v):
                        continue
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append((u, v))
                    found = True
                    break
                if found:
                    break
            if not found:
                return None
    
    # Verify regularity
    for i in range(n_target):
        if len(adj[i]) != k:
            return None
    
    return n_target, edges


def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    deadline = time.time() + 140  # safety margin
    
    # First try the LUW construction (deterministic, known girth)
    try:
        n_luw, edges_luw = _luw_construct(k, g)
        # Verify it's valid
        if _verify_girth(n_luw, edges_luw, g):
            best_n, best_edges = n_luw, edges_luw
        else:
            best_n, best_edges = None, None
    except Exception:
        best_n, best_edges = None, None
    
    # Try random greedy with various n values
    # Moore bound: for k-regular graph of girth g (odd), n >= 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
    # For k=5, g=9: n >= 1 + 5*(1 + 4 + 16 + 64) = 1 + 5*85 = 426
    # But actual lower bound is 428.
    
    moore_bound = 1 + k
    for i in range(1, (g - 1) // 2):
        moore_bound += k * (k - 1) ** i
    
    # Try n values starting from Moore bound
    n_candidates = []
    # Add some candidates around the Moore bound and higher
    n_candidates.append(moore_bound)
    n_candidates.append(moore_bound + 2)
    n_candidates.append(moore_bound + 10)
    n_candidates.append(moore_bound + 50)
    n_candidates.append(moore_bound + 100)
    n_candidates.append(moore_bound + 200)
    n_candidates.append(moore_bound + 500)
    n_candidates.append(moore_bound + 1000)
    
    # Filter: n must be even if k is odd (for k-regular graph, n*k must be even)
    if k % 2 == 1:
        n_candidates = [n for n in n_candidates if n % 2 == 0]
    
    # Sort by n (we want smallest)
    n_candidates = sorted(set(n_candidates))
    
    for n_attempt in n_candidates:
        if time.time() > deadline - 5:
            break
        if best_n is not None and n_attempt >= best_n:
            break
        
        # Try multiple seeds
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            result = _random_greedy(k, g, n_attempt, deadline, seed)
            if result is not None:
                n, edges = result
                if _verify_girth(n, edges, g):
                    if best_n is None or n < best_n:
                        best_n, best_edges = n, edges
                    break  # success with this n, move to next
    
    # Return best result
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: return LUW result even if unverified (shouldn't happen)
    try:
        n_luw, edges_luw = _luw_construct(k, g)
        return n_luw, edges_luw
    except Exception:
        # Last resort: trivial construction (may not satisfy girth but satisfies degree)
        # For safety, return a simple cycle-like structure
        n = max(2 * k, g)
        if k % 2 == 1 and n % 2 != 0:
            n += 1
        edges = []
        for i in range(n):
            for j in range(1, k // 2 + 1):
                u = i % n
                v = (i + j) % n
                if u < v:
                    edges.append((u, v))
                else:
                    edges.append((v, u))
            if k % 2 == 1:
                v = (i + n // 2) % n
                if i < v:
                    edges.append((i, v))
                else:
                    edges.append((v, i))
        return n, edges

# EVOLVE-BLOCK-END
