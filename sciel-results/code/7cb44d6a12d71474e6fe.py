import numpy as np

def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    
    # 1. Base heuristic: inverse distance
    # Avoid division by zero for diagonal
    base_heuristic = 1.0 / (distance_matrix + 1e-9)
    
    # 2. Construct Weighted Graph for Laplacian
    # Weights are inverse distances
    W = base_heuristic.copy()
    np.fill_diagonal(W, 0.0)
    
    # 3. Compute Laplacian L = D - W
    # D is diagonal matrix of row sums
    degrees = np.sum(W, axis=1)
    D = np.diag(degrees)
    L = D - W
    
    # 4. Compute Eigenvalues and Eigenvectors
    # eigh is for symmetric matrices
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # eigvals are sorted in ascending order
        # Fiedler vector is the eigenvector corresponding to the 2nd smallest eigenvalue
        # eigvecs[:, 0] is for lambda_0 (usually 0, constant vector)
        # eigvecs[:, 1] is for lambda_1 (Fiedler)
        fiedler = eigvecs[:, 1]
        
        # 5. Compute Penalty based on Fiedler coordinates
        # Difference in Fiedler coordinates
        diff = np.abs(fiedler[:, None] - fiedler[None, :])
        
        # Normalize differences to [0, 1]
        max_diff = np.max(diff)
        if max_diff > 1e-9:
            norm_diff = diff / max_diff
        else:
            # Degenerate case, no penalty
            return base_heuristic
        
        # Apply exponential penalty
        # Beta controls the strength. 
        # Higher beta -> stronger penalty for long-range Fiedler edges.
        beta = 2.0
        penalty = np.exp(-beta * (norm_diff ** 2))
        
        # 6. Combine
        heuristic = base_heuristic * penalty
        
        # Ensure non-positive values are handled (though exp is always positive)
        # The task says "Heuristic entries that are not positive become 1e-9"
        # We should probably ensure the diagonal is 0 or very small, but the evaluator
        # handles the 1e-9 floor. However, setting diagonal to 0 is standard.
        np.fill_diagonal(heuristic, 0.0)
        
        return heuristic

    except Exception:
        # Fallback to base heuristic if spectral decomposition fails
        return base_heuristic
