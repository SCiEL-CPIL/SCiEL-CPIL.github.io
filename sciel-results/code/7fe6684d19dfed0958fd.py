import scipy.sparse as sp

def _verify_girth_fast(n, edges, g):
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Map directed edges
    dir_edge_index = {}
    idx = 0
    for u in range(n):
        for v in adj[u]:
            dir_edge_index[(u, v)] = idx
            idx += 1
            
    size = idx
    rows = []
    cols = []
    data = []
    
    for u in range(n):
        for v in adj[u]:
            i = dir_edge_index[(u, v)]
            for w in adj[v]:
                if w != u:
                    j = dir_edge_index[(v, w)]
                    rows.append(i)
                    cols.append(j)
                    data.append(1)
                    
    B = sp.csr_matrix((data, (rows, cols)), shape=(size, size))
    
    current_power = B.copy()
    
    for k in range(3, g):
        if k > 3:
            current_power = current_power @ B
        # Check diagonal
        # Extract diagonal
        diag = current_power.diagonal()
        if np.any(diag > 0):
            return False
            
    return True
