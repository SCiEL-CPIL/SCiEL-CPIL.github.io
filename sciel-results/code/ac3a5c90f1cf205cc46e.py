import time

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Build polarity graph of PG(2,5): 31 vertices, 6-regular, C4-free, alpha=6
    n_base = 31
    points = _pg25_points()
    adj = [set() for _ in range(n_base)]
    
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if _is_on_polar(points[i], points[j]):
                adj[i].add(j)
                adj[j].add(i)
    
    # Verify C4-free
    assert _is_c4_free(adj, n_base), "Base graph is not C4-free"
    
    # Try to add 8 more vertices
    n = n_base
    for _ in range(8):
        if time.time() > deadline:
            break
        n += 1
        adj.append(set())
        new_idx = n - 1
        
        # Try to connect to existing vertices, maintaining C4-free
        # For each candidate u, check if |adj[new_idx] ∩ adj[u]| <= 1
        candidates = list(range(n - 1))
        # Sort by degree ascending to pack more edges
        candidates.sort(key=lambda x: len(adj[x]))
        
        for u in candidates:
            if time.time() > deadline:
                break
            common = adj[new_idx] & adj[u]
            if len(common) <= 1:
                adj[new_idx].add(u)
                adj[u].add(new_idx)
    
    # Now add edges between non-adjacent pairs to reduce alpha
    # Priority: pairs that most reduce alpha
    alpha = _max_independent_set(adj, n, timeout=20)
    
    if alpha > 10:
        # Try to add edges to reduce alpha
        # Find non-adjacent pairs and add edges that don't create C4
        non_adj_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if j not in adj[i]:
                    non_adj_pairs.append((i, j))
        
        # Sort by some heuristic: pairs where both vertices have high degree
        non_adj_pairs.sort(key=lambda p: -(len(adj[p[0]]) + len(adj[p[1]])))
        
        for i, j in non_adj_pairs:
            if time.time() > deadline:
                break
            # Check if adding edge (i,j) maintains C4-free
            common = adj[i] & adj[j]
            if len(common) <= 1:
                adj[i].add(j)
                adj[j].add(i)
        
        alpha = _max_independent_set(adj, n, timeout=20)
    
    if alpha <= 10:
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
    else:
        # Fallback to 31-vertex base
        edges = []
        for i in range(n_base):
            for j in adj[i]:
                if i < j:
                    edges.append([i, j])
        return {'n': n_base, 'edges': edges}

def _pg25_points():
    points = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    for a in range(1, 5):
        points.append((1, a, 0))
    for a in range(1, 5):
        points.append((1, 0, a))
    for a in range(1, 5):
        for b in range(5):
            points.append((1, a, b))
    return points

def _is_on_polar(p, q):
    x, y, z = p
    xp, yp, zp = q
    return (x * yp + y * (xp + zp) + z * yp) % 5 == 0

def _is_c4_free(adj, n):
    for u in range(n):
        for v in adj[u]:
            if u < v:
                if len(adj[u] & adj[v]) > 1:
                    return False
    return True

def _max_independent_set(adj, n, timeout=20):
    start = time.time()
    best = [0]
    
    # Use a simple branch and bound
    order = sorted(range(n), key=lambda x: len(adj[x]))
    in_set = [False] * n
    
    def _search(pos, size):
        nonlocal best
        if time.time() - start > timeout:
            return
        if size > best[0]:
            best[0] = size
        if pos == n:
            return
        if size + (n - pos) <= best[0]:
            return
        
        v = order[pos]
        # Try not including v
        _search(pos + 1, size)
        # Try including v
        if not any(in_set[u] for u in adj[v]):
            in_set[v] = True
            _search(pos + 1, size + 1)
            in_set[v] = False
    
    _search(0, 0)
    return best[0]
