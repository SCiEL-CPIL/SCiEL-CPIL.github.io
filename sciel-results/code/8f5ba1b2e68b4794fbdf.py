import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n = 40
    # Initialize adjacency matrix
    A = np.zeros((n, n), dtype=np.int64)
    
    # Start with 8 disjoint K5s
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
    
    # Precompute indices for upper triangle to iterate edges/non-edges
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    def compute_P5(mat):
        # A^5 = A^4 * A = (A^2)^2 * A
        # Using integer arithmetic to avoid floating point issues, 
        # though bool would work for existence. Int is safer for counts if needed, 
        # but we only care about > 0.
        A2 = mat @ mat
        A4 = A2 @ A2
        A5 = A4 @ mat
        return A5
        
    P5 = compute_P5(A)
    
    def has_simple_path_len5(u, v):
        """
        Returns True if there is a simple path of length 5 between u and v in the graph defined by A.
        Path: u - a - b - c - d - v
        a, b, c, d must be distinct and not equal to u or v.
        """
        # Get neighbors of u
        N_u = np.where(A[u] == 1)[0]
        
        for a in N_u:
            if a == v: # Path length 1, not 5
                continue
            # Get neighbors of a, excluding u
            N_a = np.where(A[a] == 1)[0]
            for b in N_a:
                if b == u:
                    continue
                # Get neighbors of b, excluding a
                N_b = np.where(A[b] == 1)[0]
                for c in N_b:
                    if c == a:
                        continue
                    # Get neighbors of c, excluding b
                    N_c = np.where(A[c] == 1)[0]
                    for d in N_c:
                        if d == b:
                            continue
                        # Check if d is connected to v and d != v (to ensure simple path ending at v via d)
                        # Actually, we just need d connected to v. 
                        # And d must not be u, a, b, c.
                        if d == u or d == a or d == c:
                            continue
                        if A[d, v] == 1:
                            return True
        return False

    steps_since_improvement = 0
    max_steps_without_improvement = 5000
    
    while time.time() < deadline:
        # 1. Try to add a definitely safe edge (P5[u,v] == 0)
        # Get non-edges
        non_edge_mask = (A[i_idx, j_idx] == 0)
        # Get those with P5 == 0
        safe_mask = non_edge_mask & (P5[i_idx, j_idx] == 0)
        safe_indices = np.where(safe_mask)[0]
        
        if len(safe_indices) > 0:
            idx = safe_indices[np.random.randint(len(safe_indices))]
            u = i_idx[idx]
            v = j_idx[idx]
            
            # Add edge
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
                    # Revert to best
                    A = best_A.copy()
                    m = best_m
                    P5 = compute_P5(A)
                    steps_since_improvement = 0
            continue
            
        # 2. Try to add a potentially risky edge (P5[u,v] > 0)
        risky_mask = non_edge_mask & (P5[i_idx, j_idx] > 0)
        risky_indices = np.where(risky_mask)[0]
        
        if len(risky_indices) > 0:
            # Try a random subset to avoid spending too much time on exact checks
            # If there are too many, just pick one
            if len(risky_indices) > 10:
                sample_size = 10
            else:
                sample_size = len(risky_indices)
            
            sampled = np.random.choice(risky_indices, size=sample_size, replace=False)
            
            added_risky = False
            for idx in sampled:
                u = i_idx[idx]
                v = j_idx[idx]
                
                if not has_simple_path_len5(u, v):
                    # It's actually safe
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
                    added_risky = True
                    break
            
            if not added_risky:
                steps_since_improvement += 1
                if steps_since_improvement > max_steps_without_improvement:
                    A = best_A.copy()
                    m = best_m
                    P5 = compute_P5(A)
                    steps_since_improvement = 0
            continue
            
        # 3. No edges can be added (all non-edges are risky and likely unsafe, or we are stuck)
        # Perform a kick: remove a random edge
        edge_mask = (A[i_idx, j_idx] == 1)
        edge_indices = np.where(edge_mask)[0]
        
        if len(edge_indices) == 0:
            break
            
        idx = edge_indices[np.random.randint(len(edge_indices))]
        u = i_idx[idx]
        v = j_idx[idx]
        
        # Remove edge
        A[u, v] = 0
        A[v, u] = 0
        m -= 1
        P5 = compute_P5(A)
        
        # If we've degraded significantly, revert
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

    # Convert best_A to list of edges
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_A[i, j] == 1:
                result.append([i, j])
                
    return result
