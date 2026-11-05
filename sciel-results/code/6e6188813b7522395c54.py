import numpy as np
from scipy.optimize import differential_evolution

def construct():
    n = 16
    target_edges = 65
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges = len(all_edges)  # 120 possible edges
    
    # Precompute triangles
    triangles = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triangles.append((i, j, k))
    num_triangles = len(triangles)
    
    # Map edge to index
    edge_to_idx = {}
    for idx, (u, v) in enumerate(all_edges):
        edge_to_idx[(u, v)] = idx
    
    # For each triangle, the three edges
    triangle_edges = []
    for (i, j, k) in triangles:
        e1 = edge_to_idx[(i, j)]
        e2 = edge_to_idx[(j, k)]
        e3 = edge_to_idx[(i, k)]
        triangle_edges.append((e1, e2, e3))
    
    # For each vertex, list of edge indices
    vertex_edges = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(all_edges):
        vertex_edges[u].append(idx)
        vertex_edges[v].append(idx)
    
    # Precompute triangle-vertex incidence
    triangle_vertices = [(i, j, k) for (i, j, k) in triangles]
    
    def objective(x):
        x = np.round(x).astype(int)
        
        # Edge count penalty
        edge_count = np.sum(x)
        edge_penalty = (edge_count - target_edges) ** 2 * 1000
        
        # Compute degrees
        degs = np.zeros(n, dtype=int)
        for v in range(n):
            degs[v] = np.sum(x[vertex_edges[v]])
        
        # Compute max degree sum over triangles
        max_sum = 0
        for t_idx in range(num_triangles):
            e1, e2, e3 = triangle_edges[t_idx]
            if x[e1] and x[e2] and x[e3]:
                i, j, k = triangle_vertices[t_idx]
                s = degs[i] + degs[j] + degs[k]
                if s > max_sum:
                    max_sum = s
        
        # Penalty for high degree sums
        threshold = 22
        if max_sum > threshold:
            tri_penalty = (max_sum - threshold) ** 2 * 100
        else:
            tri_penalty = 0
        
        return edge_penalty + tri_penalty + max_sum * 0.5
    
    # Use differential evolution for global QUBO search
    bounds = [(0, 1)] * num_edges
    
    result = differential_evolution(
        objective,
        bounds,
        maxiter=50,
        tol=1e-3,
        seed=42,
        polish=False,
        init='latinhypercube',
        popsize=10
    )
    
    x_best = np.round(result.x).astype(int)
    
    # Ensure exactly 65 edges
    current_count = np.sum(x_best)
    rng = np.random.default_rng(42)
    
    if current_count < target_edges:
        missing = target_edges - current_count
        non_edges = [i for i in range(num_edges) if x_best[i] == 0]
        for _ in range(missing):
            idx = rng.choice(non_edges)
            x_best[idx] = 1
            non_edges.remove(idx)
    elif current_count > target_edges:
        extra = current_count - target_edges
        edges_list = [i for i in range(num_edges) if x_best[i] == 1]
        for _ in range(extra):
            idx = rng.choice(edges_list)
            x_best[idx] = 0
            edges_list.remove(idx)
    
    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=int)
    for idx, (u, v) in enumerate(all_edges):
        if x_best[idx]:
            adj[u, v] = 1
            adj[v, u] = 1
    
    # Fast score calculation
    def calc_score_fast(adj):
        degs = np.sum(adj, axis=1)
        max_sum = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    common = np.where(adj[u] & adj[v] > 0)[0]
                    for w in common:
                        s = degs[u] + degs[v] + degs[w]
                        if s > max_sum:
                            max_sum = s
        return max_sum
    
    current_score = calc_score_fast(adj)
    best_score = current_score
    best_adj = adj.copy()
    
    # Simulated annealing refinement
    rng = np.random.default_rng(123)
    T = 2.0
    cooling = 0.9995
    iterations = 50000
    
    for i in range(iterations):
        u = rng.integers(0, n)
        v = rng.integers(0, n)
        while u == v or adj[u, v] == 0:
            u = rng.integers(0, n)
            v = rng.integers(0, n)
        
        x = rng.integers(0, n)
        y = rng.integers(0, n)
        while x == y or adj[x, y] == 1:
            x = rng.integers(0, n)
            y = rng.integers(0, n)
        
        if (u == x and v == y) or (u == y and v == x):
            continue
        
        adj[u, v] = 0
        adj[v, u] = 0
        adj[x, y] = 1
        adj[y, x] = 1
        
        new_score = calc_score_fast(adj)
        delta = new_score - current_score
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = adj.copy()
        else:
            adj[u, v] = 1
            adj[v, u] = 1
            adj[x, y] = 0
            adj[y, x] = 0
        
        T *= cooling
    
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
