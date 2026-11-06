import time
import random

def _is_triangle_free(n, edge_set):
    """Check if graph is triangle-free using adjacency sets."""
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[u]:
                if w > v and w in adj[v]:
                    return False
    return True

def _is_3_colorable(n, edge_set):
    """Backtracking 3-coloring check."""
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    
    colors = [-1] * n
    
    def dfs(v):
        if v == n:
            return True
        for c in range(3):
            if all(colors[u] != c for u in adj[v]):
                colors[v] = c
                if dfs(v + 1):
                    return True
                colors[v] = -1
        return False
    
    return dfs(0)

def _greedy_fill(n, base_edges, deadline, rng):
    """Greedy edge insertion to maximize edges while maintaining constraints."""
    edges = set(tuple(sorted(e)) for e in base_edges)
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() > deadline:
            break
        if (u, v) in edges:
            continue
        new_edges = edges | {(u, v)}
        # Quick triangle check: only need to check triangles involving (u,v)
        # A triangle (u,v,w) exists if w is adjacent to both u and v
        adj_u = set()
        adj_v = set()
        for a, b in new_edges:
            if a == u:
                adj_u.add(b)
            elif b == u:
                adj_u.add(a)
            if a == v:
                adj_v.add(b)
            elif b == v:
                adj_v.add(a)
        
        # Triangle exists if there's a common neighbor of u and v
        if adj_u & adj_v:
            continue
        
        # Check 3-colorability (expensive, do less frequently or with early exit)
        # For speed, we'll check after a batch of additions
        edges = new_edges
    
    return list(edges)

def _local_search_1swap(n, edges, deadline, rng, max_iters=500):
    """1-1 edge swap local search: remove one edge, add another."""
    edge_set = set(tuple(sorted(e)) for e in edges)
    current_count = len(edge_set)
    best_edges = list(edge_set)
    best_count = current_count
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try to add a random edge
        added = False
        for _ in range(20):
            u, v = rng.choice(all_edges)
            if (u, v) in edge_set:
                continue
            # Triangle check
            adj_u = set()
            adj_v = set()
            for a, b in edge_set:
                if a == u:
                    adj_u.add(b)
                elif b == u:
                    adj_u.add(a)
                if a == v:
                    adj_v.add(b)
                elif b == v:
                    adj_v.add(a)
            if adj_u & adj_v:
                continue
            
            new_set = edge_set | {(u, v)}
            if len(new_set) > best_count and _is_3_colorable(n, new_set):
                edge_set = new_set
                best_edges = list(edge_set)
                best_count = len(edge_set)
                added = True
                break
        
        if not added:
            # Try swap: remove a random edge, then add a random edge
            if edge_set:
                remove_e = rng.choice(list(edge_set))
                edge_set.discard(remove_e)
                for _ in range(20):
                    u, v = rng.choice(all_edges)
                    if (u, v) in edge_set:
                        continue
                    adj_u = set()
                    adj_v = set()
                    for a, b in edge_set:
                        if a == u:
                            adj_u.add(b)
                        elif b == u:
                            adj_u.add(a)
                        if a == v:
                            adj_v.add(b)
                        elif b == v:
                            adj_v.add(a)
                    if adj_u & adj_v:
                        continue
                    new_set = edge_set | {(u, v)}
                    if len(new_set) > best_count and _is_3_colorable(n, new_set):
                        edge_set = new_set
                        best_edges = list(edge_set)
                        best_count = len(edge_set)
                        break
                else:
                    # Revert if no improvement
                    edge_set.add(remove_e)
    
    return best_edges, best_count

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    rng = random.Random(42)
    
    # Start with Grötzsch graph base (vertices 0-10)
    base_edges = []
    for i in range(5):
        base_edges.append([i, (i + 1) % 5])
    for i in range(5):
        u = i
        v = (i + 1) % 5
        base_edges.append([u, 5 + v])
        base_edges.append([v, 5 + u])
    for i in range(5):
        base_edges.append([10, 5 + i])
    
    best_edges = []
    best_count = 0
    
    # Multiple restarts with different randomizations
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        
        # Greedy fill from base
        filled = _greedy_fill(n, base_edges, deadline, rng)
        count = len(filled)
        
        if count > best_count:
            best_edges = filled
            best_count = count
        
        # Local search
        improved_edges, improved_count = _local_search_1swap(n, filled, deadline, rng, max_iters=200)
        if improved_count > best_count:
            best_edges = improved_edges
            best_count = improved_count
        
        # Verify best is valid
        if best_count > 0:
            edge_set = set(tuple(sorted(e)) for e in best_edges)
            if not _is_triangle_free(n, edge_set):
                best_count = 0
                best_edges = []
            elif _is_3_colorable(n, edge_set):
                # Shouldn't happen, but safety check
                best_count = 0
                best_edges = []
    
    # Convert to list of [u, v] format
    result = [[u, v] for u, v in best_edges]
    return result
