import numpy as np
import time

def construct():
    """
    Construct an exact tensor decomposition for multiplying a 3x4 matrix by a 4x8 matrix.
    Uses alternating least squares to try to find a lower-rank decomposition.
    """
    deadline = time.time() + 105
    n, m, p = 3, 4, 8
    dim_u, dim_v, dim_w = n*m, m*p, p*n  # 12, 32, 24
    num_eq = n * m * p  # 96
    
    # Precompute index arrays
    flat_idx = np.arange(num_eq)
    i_idx = flat_idx // (m * p)
    j_idx = (flat_idx // p) % m
    k_idx = flat_idx % p
    
    # Schoolbook solution (rank 96)
    best_rank = 96
    best_U = np.zeros((dim_u, best_rank))
    best_V = np.zeros((dim_v, best_rank))
    best_W = np.zeros((dim_w, best_rank))
    for c in range(num_eq):
        best_U[i_idx[c]*m + j_idx[c], c] = 1
        best_V[j_idx[c]*p + k_idx[c], c] = 1
        best_W[k_idx[c]*n + i_idx[c], c] = 1
    
    # Try to find lower rank via alternating least squares
    for target_rank in [88, 84, 80]:
        if time.time() > deadline - 40:
            break
        
        # Initialize from schoolbook
        U = best_U[:, :target_rank].copy()
        V = best_V[:, :target_rank].copy()
        W = best_W[:, :target_rank].copy()
        
        for iteration in range(200):
            if time.time() > deadline - 40:
                break
            
            # Solve for W given U, V
            A_w = np.zeros((num_eq, dim_w * target_rank))
            for t in range(target_rank):
                cols = (k_idx * n + i_idx) * target_rank + t
                A_w[np.arange(num_eq), cols] = U[i_idx*m + j_idx, t] * V[j_idx*p + k_idx, t]
            W = np.linalg.lstsq(A_w, np.ones(num_eq), rcond=1e-12)[0].reshape(dim_w, target_rank)
            
            # Solve for U given V, W
            A_u = np.zeros((num_eq, dim_u * target_rank))
            for t in range(target_rank):
                cols = (i_idx * m + j_idx) * target_rank + t
                A_u[np.arange(num_eq), cols] = V[j_idx*p + k_idx, t] * W[k_idx*n + i_idx, t]
            U = np.linalg.lstsq(A_u, np.ones(num_eq), rcond=1e-12)[0].reshape(dim_u, target_rank)
            
            # Solve for V given U, W
            A_v = np.zeros((num_eq, dim_v * target_rank))
            for t in range(target_rank):
                cols = (j_idx * p + k_idx) * target_rank + t
                A_v[np.arange(num_eq), cols] = U[i_idx*m + j_idx, t] * W[k_idx*n + i_idx, t]
            V = np.linalg.lstsq(A_v, np.ones(num_eq), rcond=1e-12)[0].reshape(dim_v, target_rank)
        
        # Check if we found an exact solution
        max_err = 0.0
        for t in range(target_rank):
            contrib = U[i_idx*m + j_idx, t] * V[j_idx*p + k_idx, t] * W[k_idx*n + i_idx, t]
            # Sum over t for each equation
            # Need to sum properly
        # Proper check
        tensor_vals = np.zeros(num_eq)
        for t in range(target_rank):
            tensor_vals += U[i_idx*m + j_idx, t] * V[j_idx*p + k_idx, t] * W[k_idx*n + i_idx, t]
        max_err = np.max(np.abs(tensor_vals - 1))
        
        if max_err < 1e-8:
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
