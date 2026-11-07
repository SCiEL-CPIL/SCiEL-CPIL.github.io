import numpy as np

def heuristics(prizes, distance_matrix, max_len):
        n = len(prizes)
        prizes = np.asarray(prizes, dtype=np.float64)
        dist = np.asarray(distance_matrix, dtype=np.float64)
        
        # Heuristic matrix
        eta = np.zeros((n, n), dtype=np.float64)
        
        eps = 1e-9
        alpha = 1.5
        beta = 1.5
        
        # Precompute node-level penalties
        # d_j0 is distance from j to start
        d_j0 = dist[:, 0].copy()
        # d_0j is distance from start to j (same if symmetric, but let's use dist[0, :] just in case)
        d_0j = dist[0, :].copy()
        
        # Feasibility check: Can we visit j and return?
        # 0 -> j -> 0 cost is d_0j + d_j0
        feasible = (d_0j + d_j0) <= max_len
        
        # For each j, calculate the "static" attractiveness
        # We want eta[i, j] to be high if j is good.
        # The "cost" of j depends on i (d_ij) and static (d_j0).
        
        # Vectorized calculation
        # d_ij is the full distance matrix
        # We want eta[i, j] = p[j] / (d[i, j] * d[j, 0])^alpha ... wait, product of terms or term of product?
        # Let's use: eta[i, j] = p[j] / ((d[i, j] + eps) * (d[j, 0] + eps))
        
        # Create matrix for d_j0 term
        # d_j0_term[i, j] = d[j, 0] + eps
        d_j0_mat = np.tile(d_j0[None, :], (n, 1))
        
        # d_ij_mat = dist + eps
        d_ij_mat = dist + eps
        
        # Denominator
        denom = (d_ij_mat ** alpha) * (d_j0_mat ** beta)
        
        # Numerator
        numer = np.tile(prizes[None, :], (n, 1))
        
        # Calculate eta
        with np.errstate(divide='ignore', invalid='ignore'):
            eta = numer / denom
        
        # Apply feasibility mask
        # If node j is not feasible, set eta[:, j] to 1e-9
        for j in range(n):
            if not feasible[j]:
                eta[:, j] = 1e-9
                
        # Handle node 0
        # We don't move TO node 0 in the loop (it's the start/end, but ants stop when they can't move to unvisited).
        # Actually, the prompt says "moves to an unvisited node j". Node 0 is visited.
        # So column 0 should be ignored/low.
        eta[:, 0] = 1e-9
        
        # Diagonal
        np.fill_diagonal(eta, 1e-9)
        
        # Ensure positive and finite
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        
        return eta
