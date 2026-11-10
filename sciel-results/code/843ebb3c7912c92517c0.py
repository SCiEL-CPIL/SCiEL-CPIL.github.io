import numpy as np

def construct():
    n = 20
    np.random.seed(42)
    
    dim = 2
    
    def compute_adj(pos, threshold):
        dists = np.zeros((n, n))
        for u in range(n):
            for v in range(u+1, n):
                d = np.sqrt(np.sum((pos[u] - pos[v])**2))
                dists[u, v] = dists[v, u] = d
        adj = (dists < threshold).astype(np.int8)
        np.fill_diagonal(adj, 0)
        return adj
    
    def objective_fast(pos, threshold):
        adj = compute_adj(pos, threshold)
        num_edges = int(adj.sum() // 2)
        common = adj @ adj
        penalty = 0
        for u in range(n):
            for v in range(u+1, n):
                c = common[u, v]
                if c >= 3:
                    penalty += c - 2
        return num_edges - 5 * penalty
    
    best_obj = -np.inf
    best_adj = None
    
    for threshold in [1.0, 1.3, 1.6, 1.9]:
        np.random.seed(42)
        pos = np.random.randn(n, dim) * 0.8
        
        obj = objective_fast(pos, threshold)
        
        for it in range(100):
            improved = False
            for i in range(n):
                for d in range(dim):
                    for delta in [-0.08, 0.08]:
                        new_pos = pos.copy()
                        new_pos[i, d] += delta
                        new_obj = objective_fast(new_pos, threshold)
                        if new_obj > obj:
                            pos = new_pos
                            obj = new_obj
                            improved = True
            if not improved:
                break
        
        if obj > best_obj:
            best_obj = obj
            best_adj = compute_adj(pos, threshold)
    
    if best_adj is None:
        best_adj = np.zeros((n, n), dtype=np.int8)
    
    # Greedy edge addition with fast checking
    adj = best_adj.copy()
    improved = True
    while improved:
        improved = False
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v] == 0:
                    ok = True
                    common_uv = int(adj[u] @ adj[v])
                    if common_uv >= 3:
                        continue
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        if adj[v, w]:
                            if int(adj[u] @ adj[w]) + 1 >= 3:
                                ok = False
                                break
                        if ok and adj[u, w]:
                            if int(adj[v] @ adj[w]) + 1 >= 3:
                                ok = False
                                break
                    if ok:
                        adj[u, v] = 1
                        adj[v, u] = 1
                        improved = True
    
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u, v]:
                edges.append([u, v])
    return edges
