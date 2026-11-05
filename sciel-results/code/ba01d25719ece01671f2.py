import numpy as np
import time

def construct():
    n = 40
    # Safety margin: evaluator kills at 150s, we aim for 105s
    deadline = time.time() + 105.0
    
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
    
    def count_blocked_edges(adj, p5, u, v):
        """Count how many potential edges are blocked by edge (u,v)"""
        # An edge (a,b) is blocked by (u,v) if adding (a,b) would create a C6
        # This happens when there's a path of length 5 between a and b that uses u-v
        # Simpler: just count non-edges (a,b) where p5[a,b] > 0
        # But we want to know which edges to remove to unblock the most
        # For now, use a heuristic: prefer edges with high degree
        return 0
    
    def weighted_valid_edge_selection(adj, p5, valid_edges, temperature=1.0):
        """Select a valid edge with some heuristic weighting"""
        if not valid_edges:
            return None
        
        # Weight by degree sum (prefer connecting well-connected vertices)
        weights = []
        for u, v in valid_edges:
            deg_u = int(adj[u].sum())
            deg_v = int(adj[v].sum())
            # Prefer edges that connect vertices with moderate degree
            # (not too isolated, not already over-connected)
            weight = 1.0 + 0.1 * (deg_u + deg_v)
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Sample with temperature
        if temperature > 0:
            weights = weights ** (1.0 / temperature)
            weights = weights / weights.sum()
        
        idx = np.random.choice(len(valid_edges), p=weights)
        return valid_edges[idx]
    
    steps_since_improvement = 0
    max_steps_without_improvement = 3000
    kick_count = 0
    
    while time.time() < deadline:
        valid_edges = get_valid_non_edges(A, P5)
        
        if not valid_edges:
            # Stuck: No edges can be added
            # Perform a kick: remove an edge that blocks the most potential additions
            current_edges = get_all_edges(A)
            if not current_edges:
                break
            
            # Try to find an edge whose removal unblocks the most edges
            # Score each edge by how many valid edges would become available
            best_edge_to_remove = None
            best_score = -1
            
            # Sample a subset of edges to evaluate (for speed)
            num_to_eval = min(50, len(current_edges))
            sample_indices = np.random.choice(len(current_edges), size=num_to_eval, replace=False)
            
            for idx in sample_indices:
                u, v = current_edges[idx]
                # Temporarily remove and count new valid edges
                A[u, v] = 0
                A[v, u] = 0
                P5_temp = compute_P5(A)
                new_valid = get_valid_non_edges(A, P5_temp)
                score = len(new_valid)
                
                # Restore
                A[u, v] = 1
                A[v, u] = 1
                
                if score > best_score:
                    best_score = score
                    best_edge_to_remove = (u, v)
            
            # Restore P5
            P5 = compute_P5(A)
            
            if best_edge_to_remove is None:
                # Fallback to random
                u, v = current_edges[np.random.randint(len(current_edges))]
            else:
                u, v = best_edge_to_remove
            
            A[u, v] = 0
            A[v, u] = 0
            m -= 1
            P5 = compute_P5(A)
            kick_count += 1
            
            # If we have degraded significantly, revert to best
            if m < best_m - 8:
                A = best_A.copy()
                m = best_m
                P5 = compute_P5(A)
                steps_since_improvement = 0
                kick_count = 0
            else:
                steps_since_improvement += 1
                if steps_since_improvement > max_steps_without_improvement:
                    A = best_A.copy()
                    m = best_m
                    P5 = compute_P5(A)
                    steps_since_improvement = 0
                    kick_count = 0
            continue
            
        # Add a valid edge with heuristic weighting
        temperature = 1.0 + 0.5 * (kick_count / 10.0)  # increase randomness after kicks
        selected = weighted_valid_edge_selection(A, P5, valid_edges, temperature)
        
        if selected is None:
            break
            
        u, v = selected
        A[u, v] = 1
        A[v, u] = 1
        m += 1
        P5 = compute_P5(A)
        kick_count = 0
        
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
                kick_count = 0
                
    # Convert best_A to list of edges
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_A[u, v] == 1:
                result.append([u, v])
                
    return result
