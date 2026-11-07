import numpy as np

def heuristics(distance_matrix):
    dist = np.array(distance_matrix)
    n = dist.shape[0]
    
    # 1. Create weight matrix W where W_ij = 1/d_ij
    # Add small epsilon to avoid division by zero
    eps = 1e-9
    W = 1.0 / (dist + eps)
    np.fill_diagonal(W, 0.0)
    
    # 2. Compute Laplacian L = D - W
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    
    # 3. Compute eigenvalues and eigenvectors
    # For symmetric matrix, use eigh
    eigvals, eigvecs = np.linalg.eigh(L)
    
    # 4. Get the first two non-trivial eigenvectors (indices 1 and 2)
    # The first eigenvector (index 0) is usually constant (connected graph)
    if n > 2:
        u = eigvecs[:, 1]
        v = eigvecs[:, 2]
    else:
        # Fallback for very small N, though task says N=50
        u = np.zeros(n)
        v = np.zeros(n)
        
    # 5. Compute pairwise distances in the spectral embedding space
    # diff_u_ij = |u_i - u_j|
    diff_u = np.abs(u[:, None] - u[None, :])
    diff_v = np.abs(v[:, None] - v[None, :])
    emb_dist = np.sqrt(diff_u**2 + diff_v**2)
    
    # 6. Construct Heuristic
    # Base heuristic is 1/d
    # Modulate by spectral proximity: 1 / (1 + emb_dist)
    # This makes edges that are close in spectral space more attractive
    heuristic = W / (1.0 + emb_dist)
    
    # 7. Ensure all values are positive and finite
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
