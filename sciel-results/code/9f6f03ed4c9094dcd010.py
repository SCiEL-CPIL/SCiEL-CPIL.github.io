import time
import numpy as np

def _count_c4(adj):
    """Count 4-cycles in a graph given its adjacency matrix."""
    n = adj.shape[0]
    # common[u, v] = number of common neighbors of u and v
    common = adj @ adj
    total = 0
    # Iterate over upper triangle
    for u in range(n):
        for v in range(u + 1, n):
            k = common[u, v]
            if k >= 2:
                total += k * (k - 1) // 2
    return total

def _build_polarity_graph():
    """
    Construct the polarity graph of PG(2,5).
    Returns adj matrix (31x31) and list of edges.
    """
    q = 5
    points = []
    seen = set()
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0 and b == 0 and c == 0:
                    continue
                vec = [a, b, c]
                for i in range(3):
                    if vec[i] != 0:
                        inv = pow(vec[i], q - 2, q)
                        vec = [(v * inv) % q for v in vec]
                        break
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(vec)
    
    n = len(points)
    assert n == 31
    
    # Polarity matrix
    # Using M = [[1, 0, 0], [0, 1, 0], [0, 0, 4]]
    M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 4]], dtype=int)
    
    adj = np.zeros((n, n), dtype=int)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            x = np.array(points[i], dtype=int)
            y = np.array(points[j], dtype=int)
            val = (x @ M @ y) % q
            if val == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                edges.append([i, j])
    
    return edges, adj

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Build base graph
    edges, adj = _build_polarity_graph()
    
    # 2. Add best edge
    n = 31
    best_edge = None
    best_c4 = float('inf')
    
    # Precompute neighbors for speed?
    # Just loop
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0:
                # Count C4s created by adding (i,j)
                # This is number of edges (a,b) with a in N(i), b in N(j)
                ni = np.where(adj[i] == 1)[0]
                nj = np.where(adj[j] == 1)[0]
                
                c4s_created = 0
                # Use matrix multiplication or loop
                # adj[ni, nj] gives submatrix
                # Count 1s
                if len(ni) > 0 and len(nj) > 0:
                    sub = adj[np.ix_(ni, nj)]
                    c4s_created = np.sum(sub)
                
                if c4s_created < best_c4:
                    best_c4 = c4s_created
                    best_edge = [i, j]
                    
                if time.time() > deadline - 10.0:
                    break
        if time.time() > deadline - 10.0:
            break
            
    if best_edge:
        adj[best_edge[0], best_edge[1]] = 1
        adj[best_edge[1], best_edge[0]] = 1
        edges.append(best_edge)
        
    current_c4 = best_c4 # Since base was 0
    
    # 3. Local Search
    # Try to swap edges to reduce C4 count
    # Max iterations
    max_iter = 500
    
    for _ in range(max_iter):
        if time.time() > deadline - 20.0:
            break
            
        best_swap = None
        best_c4_new = current_c4
        
        # Iterate over all edges to remove
        # and all non-edges to add
        # To speed up, we can sample or limit
        # But n=31 is small.
        # 91 edges * ~370 non-edges = 33k swaps
        
        # List of edges
        edge_list = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == 1]
        
        for e1 in edge_list:
            u, v = e1
            # Remove e1
            adj[u, v] = 0
            adj[v, u] = 0
            
            # Try adding non-edges
            # We can skip non-edges that are "too far" or just try all
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i, j] == 0 and (i,j) != (u,v) and (i,j) != (v,u):
                         # Add (i,j)
                         adj[i, j] = 1
                         adj[j, i] = 1
                         
                         new_c4 = _count_c4(adj)
                         
                         # Rollback add
                         adj[i, j] = 0
                         adj[j, i] = 0
                         
                         if new_c4 < best_c4_new:
                             best_c4_new = new_c4
                             best_swap = (e1, [i, j])
                             
                         if time.time() > deadline - 20.0:
                             break
                if time.time() > deadline - 20.0:
                    break
            if time.time() > deadline - 20.0:
                break
                
            # Rollback remove
            adj[u, v] = 1
            adj[v, u] = 1
            
            if time.time() > deadline - 20.0:
                break
                
        if best_swap:
            # Apply best swap
            e1, e2 = best_swap
            u, v = e1
            x, y = e2
            adj[u, v] = 0
            adj[v, u] = 0
            adj[x, y] = 1
            adj[y, x] = 1
            
            # Update edges list
            edges.remove([u, v])
            edges.append([x, y])
            
            current_c4 = best_c4_new
        else:
            # No improvement found
            break
            
    # Return edges
    return edges
