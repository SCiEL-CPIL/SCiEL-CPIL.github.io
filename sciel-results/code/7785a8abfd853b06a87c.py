import numpy as np
import time

def construct():
    n = 40
    # Safety margin: evaluator kills at 150s, we aim for 110s
    deadline = time.time() + 110.0
    
    # Initialize adjacency matrix with 8 disjoint K5s
    A = np.zeros((n, n), dtype=np.int64)
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                A[u, v] = 1
                A[v, u] = 1
                
    m = 80
    best_m = m
    best_A = A.copy()
    
    # Precompute indices for upper triangle
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    def compute_P5(mat):
        # A^5 = A^4 * A = (A^2)^2 * A
        A2 = mat @ mat
        A4 = A2 @ A2
        A5 = A4 @ mat
        return A5
        
    P5 = compute_P5(A)
    
    def get_valid_non_edges(adj, p5):
        # Check non-edges in upper triangle
        # adj[i_idx, j_idx] == 0 AND p5[i_idx, j_idx] == 0
        valid_mask = (adj[i_idx, j_idx] == 0) & (p5[i_idx, j_idx] == 0)
        valid_indices = np.where(valid_mask)[0]
        # Convert to list of tuples
        return [(int(i_idx[k]), int(j_idx[k])) for k in valid_indices]
        
    def get_all_edges(adj):
        edge_mask = adj[i_idx, j_idx] == 1
        edge_indices = np.where(edge_mask)[0]
        return [(int(i_idx[k]), int(j_idx[k])) for k in edge_indices]
        
    steps_since_improvement = 0
    # Revert to best if we've made many moves without improving the best score
    # This prevents wandering too far into a bad region
    max_steps_without_improvement = 5000
    
    while time.time() < deadline:
        valid_edges = get_valid_non_edges(A, P5)
        
        if not valid_edges:
            # Stuck: No edges can be added
            # Perform a kick: remove a random edge
            current_edges = get_all_edges(A)
            if not current_edges:
                break
            
            u, v = current_edges[np.random.randint(len(current_edges))]
            A[u, v] = 0
            A[v, u] = 0
            m -= 1
            P5 = compute_P5(A)
            
            # If we have degraded significantly, revert to best
            if m < best_m - 10:
                A = best_A.copy()
                m = best_m
                P5 = compute_P5(A)
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
                if steps_since_improvement > max_steps_without_improvement:
                    A = best_A.copy()
                    m = best_m
                    P5 = compute_P5(A)
                    steps_since_improvement = 0
            continue
            
        # Add a random valid edge
        u, v = valid_edges[np.random.randint(len(valid_edges))]
        A[u, v] = 1
        A[v, u] = 1
        m += 1
        P5 = compute_P5(A)
        
        if m > best_m:
            best_m = m
            best_A = A.copy()
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1
            if steps_since_improvement > max_steps_without_improvement:
                A = best_A.copy()
                m = best_m
                P5 = compute_P5(A)
                steps_since_improvement = 0
                
    # Convert best_A to list of edges
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_A[u, v] == 1:
                result.append([u, v])
                
    return result
