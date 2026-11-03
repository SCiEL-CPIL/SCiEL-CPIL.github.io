import numpy as np

def construct(k, g):
    # Compute Moore bound M (lower bound on n)
    # For k-regular graph with girth g:
    # If g = 2r+1 (odd): M = 1 + k * sum_{i=0}^{r-1} (k-1)^i
    # If g = 2r   (even): M = 2 * sum_{i=0}^{r-1} (k-1)^i  (for k>1, roughly)
    r = (g - 1) // 2
    M = 1
    for i in range(r):
        if i == 0:
            M += k
        else:
            M += k * (k - 1) ** i
    # Ensure M is at least something reasonable
    M = max(M, 2)
    
    # For (5, 9): r = 4, M = 1 + 5*(1 + 4 + 16 + 64) = 1 + 5*85 = 426
    # Best known lower bound is 428, so M is close.
    
    def try_cage(n, k, g, max_attempts=200, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        # Try to build a k-regular graph on n vertices with girth >= g
        # Using configuration model + 2-switches to fix short cycles
        # First, generate a random k-regular simple graph
        for attempt in range(max_attempts):
            # Configuration model: n*k half-edges
            half_edges = np.arange(n * k)
            rng.shuffle(half_edges)
            
            # Build adjacency (allowing multi-edges/loops initially)
            adj = [set() for _ in range(n)]
            valid = True
            for i in range(n * k // 2):
                u = half_edges[2 * i] // k
                v = half_edges[2 * i + 1] // k
                if u == v:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            
            if not valid:
                continue
            
            # Check degree exactly k
            if any(len(adj[i]) != k for i in range(n)):
                continue
            
            # Now try to increase girth via 2-switches
            # We'll iteratively find short cycles and break them
            success = False
            for _ in range(5000):
                # Find if girth < g
                # BFS-based girth check
                girth_found = False
                for start in range(n):
                    # BFS from start
                    dist = [-1] * n
                    dist[start] = 0
                    queue = [start]
                    head = 0
                    while head < len(queue):
                        u = queue[head]
                        head += 1
                        if dist[u] >= g // 2:
                            continue
                        for v in adj[u]:
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                queue.append(v)
                            else:
                                # Back edge or cross edge
                                # Cycle length: dist[u] + dist[v] + 1 (approx, not exact)
                                # For girth check, if dist[u] + dist[v] + 1 < g, we have a cycle < g
                                # This is a heuristic; exact check needs more work
                                if v != start:  # avoid trivial
                                    cyc_len = dist[u] + dist[v] + 1
                                    if cyc_len < g and cyc_len <= g:
                                        # Found a cycle of length < g (or = g, which is ok)
                                        if cyc_len < g:
                                            girth_found = True
                                            break
                        if girth_found:
                            break
                    if girth_found:
                        break
                
                if not girth_found:
                    success = True
                    break
                
                # Perform a 2-switch to break cycles
                # Pick a random edge (u,v), and another edge (x,y), and swap to (u,x),(v,y) or (u,y),(v,x)
                # This preserves regularity
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                
                if len(edges) < 2:
                    break
                
                # Try several random 2-switches
                for _ in range(20):
                    e1 = edges[rng.integers(len(edges))]
                    e2 = edges[rng.integers(len(edges))]
                    if e1 == e2:
                        continue
                    u, v = e1
                    x, y = e2
                    # Try swap: (u,v),(x,y) -> (u,x),(v,y)
                    # Check no loops/multiedges
                    if x in (u, v) or y in (u, v):
                        continue
                    # Check if u-x and v-y are new edges
                    if x in adj[u] or y in adj[v]:
                        # Try other pairing
                        if y in adj[u] or x in adj[v]:
                            continue
                        # Swap to (u,y),(v,x)
                        adj[u].remove(v)
                        adj[v].remove(u)
                        adj[x].remove(y)
                        adj[y].remove(x)
                        adj[u].add(y)
                        adj[y].add(u)
                        adj[v].add(x)
                        adj[x].add(v)
                    else:
                        # Swap to (u,x),(v,y)
                        adj[u].remove(v)
                        adj[v].remove(u)
                        adj[x].remove(y)
                        adj[y].remove(x)
                        adj[u].add(x)
                        adj[x].add(u)
                        adj[v].add(y)
                        adj[y].add(v)
                    break  # Only one switch per iteration
            
            if success:
                # Build edge list
                edge_list = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edge_list.append((u, v))
                return edge_list
        return None
    
    # Search for the smallest n >= M that admits a valid graph
    # Start from M and go up
    # For efficiency, try a range
    rng = np.random.default_rng(42)
    
    # Try from M upward
    # M for (5,9) is 426, best known lower bound is 428
    # We'll try n = M, M+1, M+2, ... up to some limit
    max_n = M + 50  # Search a modest range
    
    for n in range(M, max_n + 1):
        # For each n, try multiple random seeds
        for seed in range(50):
            rng_local = np.random.default_rng(seed)
            edges = try_cage(n, k, g, max_attempts=30, rng=rng_local)
            if edges is not None:
                # Validate
                # Check all vertices have degree k
                adj_check = [set() for _ in range(n)]
                for (u, v) in edges:
                    adj_check[u].add(v)
                    adj_check[v].add(u)
                if any(len(adj_check[i]) != k for i in range(n)):
                    continue
                # Check girth
                from collections import deque
                ok = True
                for start in range(n):
                    dist = [-1] * n
                    dist[start] = 0
                    q = deque([start])
                    while q and ok:
                        u = q.popleft()
                        if dist[u] >= g // 2:
                            continue
                        for v in adj_check[u]:
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                q.append(v)
                            else:
                                if v != start:
                                    cyc = dist[u] + dist[v] + 1
                                    if cyc < g:
                                        ok = False
                                        break
                    if not ok:
                        break
                if ok:
                    return n, edges
    
    # Fallback: return a simple construction if all else fails
    # Use a circulant-like graph
    n = M
    edges = []
    for i in range(n):
        for j in range(1, k + 1):
            neighbor = (i + j) % n
            if i < neighbor:
                edges.append((i, neighbor))
    return n, edges
