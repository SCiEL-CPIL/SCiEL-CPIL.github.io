"""Improved matrix multiplication tensor decomposition for 4x5 x 5x6."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    best_rank = n * m * p  # Schoolbook rank = 120
    
    # Helper: verify decomposition
    def verify_fast(U, V, W, rank, tol=1e-9):
        if time.time() > deadline:
            return False
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # The tensor T[i,j,k] = 1 for all i,j,k
        # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        T = np.zeros((n, m, p))
        for c in range(rank):
            uc_3d = U[:, c].reshape(n, m)
            vc_3d = V[:, c].reshape(m, p)
            wc_3d = W[:, c].reshape(p, n)
            # For each i,j,k: uc_3d[i,j] * vc_3d[j,k] * wc_3d[k,i]
            # Use einsum for efficiency
            T += np.einsum('ij,jk,ki->ijk', uc_3d, vc_3d, wc_3d)
        
        if not np.allclose(T, np.ones((n, m, p)), atol=tol):
            return False
        return True
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U, V, W, rank = schoolbook()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try ALS-based optimization to find lower-rank decompositions
    def try_rank_als(r, max_iter=50, tol=1e-10, seed=42):
        nonlocal best_U, best_V, best_W, best_rank
        if time.time() > deadline:
            return
        if r >= best_rank:
            return
        
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, r)
        V = rng.randn(m * p, r)
        W = rng.randn(p * n, r)
        
        # Normalize columns to prevent divergence
        for c in range(r):
            norm_u = np.linalg.norm(U[:, c])
            norm_v = np.linalg.norm(V[:, c])
            norm_w = np.linalg.norm(W[:, c])
            if norm_u > 0:
                U[:, c] /= norm_u
            if norm_v > 0:
                V[:, c] /= norm_v
            if norm_w > 0:
                W[:, c] /= norm_w
        
        for it in range(max_iter):
            if time.time() > deadline:
                return
            
            # Update U
            for c in range(r):
                # For each c, we want to minimize ||T - sum_d U_d ⊗ V_d ⊗ W_d||_F
                # Fix V and W, update U[:, c]
                # The gradient w.r.t. U[:, c] is:
                # For each (i,j): sum_k V[j*p+k, c] * W[k*n+i, c] * (1 - T_recon[i,j,k])
                # This is complex. Let's use a simpler approach.
                pass
            
            # Simpler: use alternating least squares on the tensor
            # Reshape T to a matrix for each mode
            # Mode-1 unfolding: T_1[i, (j,k)] where index = i + n*(j + m*k)
            # But our T is defined with the special structure...
            
            # Let's use a different approach: minimize the Frobenius norm of (T_recon - T_target)
            # where T_target[i,j,k] = 1
            # This is equivalent to minimizing ||sum_c u_c ⊗ v_c ⊗ w_c - T_target||_F
            
            # For mode-1 (U): fix V, W
            # T_recon[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # For fixed i,j: T_recon[i,j,k] as a function of U[i*m+j, c]
            # This is a linear system in U[:, c] for each c
            
            # Actually, let's use a simpler heuristic: random restarts with greedy column selection
            
            break  # Skip complex ALS for now, use random restarts
        
        # Instead, use random restarts with verification
        return
    
    # Use random restarts to find lower-rank exact decompositions
    def random_search(r, n_trials=10, seed=42):
        nonlocal best_U, best_V, best_W, best_rank
        if time.time() > deadline:
            return
        if r >= best_rank:
            return
        
        rng = np.random.RandomState(seed)
        
        for trial in range(n_trials):
            if time.time() > deadline:
                return
            
            # Initialize with random values
            U = rng.randn(n * m, r)
            V = rng.randn(m * p, r)
            W = rng.randn(p * n, r)
            
            # Try to refine using gradient descent on the error
            lr = 0.01
            for it in range(100):
                if time.time() > deadline:
                    return
                
                # Compute T_recon
                T_recon = np.zeros((n, m, p))
                for c in range(r):
                    uc_3d = U[:, c].reshape(n, m)
                    vc_3d = V[:, c].reshape(m, p)
                    wc_3d = W[:, c].reshape(p, n)
                    T_recon += np.einsum('ij,jk,ki->ijk', uc_3d, vc_3d, wc_3d)
                
                error = T_recon - np.ones((n, m, p))
                err_norm = np.linalg.norm(error)
                
                if err_norm < 1e-8:
                    # Check if we can round to exact values
                    if verify_fast(U, V, W, r, tol=1e-8):
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_rank = r
                        return
                
                # Compute gradients
                # dU[i*m+j, c] = -sum_k error[i,j,k] * V[j*p+k, c] * W[k*n+i, c]
                # dV[j*p+k, c] = -sum_i error[i,j,k] * U[i*m+j, c] * W[k*n+i, c]
                # dW[k*n+i, c] = -sum_j error[i,j,k] * U[i*m+j, c] * V[j*p+k, c]
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for c in range(r):
                    uc_3d = U[:, c].reshape(n, m)
                    vc_3d = V[:, c].reshape(m, p)
                    wc_3d = W[:, c].reshape(p, n)
                    
                    # dU
                    for i in range(n):
                        for j in range(m):
                            dU[i*m+j, c] = -np.sum(error[i, j, :] * vc_3d[j, :] * np.array([wc_3d[k, i] for k in range(p)]))
                    
                    # dV
                    for j in range(m):
                        for k in range(p):
                            dV[j*p+k, c] = -np.sum(error[:, j, k] * np.array([uc_3d[i, j] for i in range(n)]) * wc_3d[k, i])
                    
                    # dW
                    for k in range(p):
                        for i in range(n):
                            dW[k*n+i, c] = -np.sum(error[i, :, k] * np.array([uc_3d[i, j] for j in range(m)]) * vc_3d[j, k])
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            # Check final result
            if verify_fast(U, V, W, r, tol=1e-8):
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = r
                return
    
    # Try ranks from best_rank-1 down to a lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(20, 30, 24) = 30
    # But actually the tensor rank is at least max(n*m, m*p, p*n) for the flattening
    lower_bound = max(n*m, m*p, p*n)
    
    # Try a few promising ranks
    ranks_to_try = [best_rank - 1, best_rank - 2, best_rank - 5, best_rank - 10, best_rank - 20]
    ranks_to_try = [r for r in ranks_to_try if r >= lower_bound]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        random_search(r, n_trials=5, seed=42 + r)
    
    # Return the best result found
    result = {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
    
    return result
# EVOLVE-BLOCK-END
