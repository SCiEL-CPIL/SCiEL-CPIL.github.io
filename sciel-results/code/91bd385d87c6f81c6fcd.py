"""Program for finding a rank-minimal exact tensor decomposition for 2x4*4x5 matrix multiplication."""
import numpy as np
import random
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a low-rank tensor decomposition for 2x4 * 4x5 matrix multiplication.
    
    The tensor T has indices (i, j, k) where i in [0,1], j in [0,3], k in [0,4].
    The standard multiplication tensor is T[i,j,k] = 1 for all i,j,k.
    
    We use alternating minimization to find a rank-r decomposition:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    The output factor follows the official trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p  # 40
    
    def build_schoolbook():
        U = np.zeros((n * m, schoolbook_rank))
        V = np.zeros((m * p, schoolbook_rank))
        W = np.zeros((p * n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    def compute_tensor(U, V, W):
        """Compute the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]"""
        r = U.shape[1]
        T = np.zeros((n, m, p))
        for c in range(r):
            uc = U[:, c]  # shape (n*m,)
            vc = V[:, c]  # shape (m*p,)
            wc = W[:, c]  # shape (p*n,)
            # Reshape for outer product
            uc_r = uc.reshape(n, m)
            vc_r = vc.reshape(m, p)
            wc_r = wc.reshape(p, n)
            # T[i,j,k] += uc_r[i,j] * vc_r[j,k] * wc_r[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += uc_r[i, j] * vc_r[j, k] * wc_r[k, i]
        return T
    
    def tensor_residual(U, V, W):
        """Compute residual ||T_computed - T_target|| where T_target is all ones"""
        T = compute_tensor(U, V, W)
        return np.linalg.norm(T - 1.0)
    
    def alternating_minimization(r, max_iters=5000, tol=1e-12, n_restarts=20, seed=42):
        """
        Find a rank-r decomposition via alternating minimization.
        T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        """
        rng = np.random.RandomState(seed)
        best_res = np.inf
        best_U, best_V, best_W = None, None, None
        
        for restart in range(n_restarts):
            # Random initialization
            scale = 1.0 / np.sqrt(r)
            U = rng.randn(n * m, r) * scale
            V = rng.randn(m * p, r) * scale
            W = rng.randn(p * n, r) * scale
            
            for it in range(max_iters):
                prev_res = tensor_residual(U, V, W)
                
                # Update U: for each (i,j), minimize sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c])
                # Fix V and W, solve for U
                # For fixed (i,j): T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # This is a linear system in U[i*m+j, :]
                # Let A[i,j,k,c] = V[j*p+k,c] * W[k*n+i,c]
                # Then T[i,j,k] = sum_c A[i,j,k,c] * U[i*m+j,c]
                # We want T[i,j,k] = 1 for all k
                # So for each (i,j), we solve: A_{ij} @ u_{ij} = ones(p)
                # where A_{ij} is p x r matrix with A_{ij}[k,c] = V[j*p+k,c] * W[k*n+i,c]
                
                for i in range(n):
                    for j in range(m):
                        # Build p x r matrix
                        A_mat = np.zeros((p, r))
                        for k in range(p):
                            for c in range(r):
                                A_mat[k, c] = V[j*p + k, c] * W[k*n + i, c]
                        # Solve A_mat @ u = ones(p)
                        # Use least squares
                        try:
                            u_sol, _, _, _ = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                            U[i*m + j, :] = u_sol
                        except:
                            pass
                
                # Update V: for each (j,k), minimize
                # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # Fix U and W, solve for V
                # For fixed (j,k): T[i,j,k] = sum_c V[j*p+k,c] * (U[i*m+j,c] * W[k*n+i,c])
                # A_{jk}[i,c] = U[i*m+j,c] * W[k*n+i,c]
                # A_{jk} @ v_{jk} = ones(n)
                
                for j in range(m):
                    for k in range(p):
                        A_mat = np.zeros((n, r))
                        for i in range(n):
                            for c in range(r):
                                A_mat[i, c] = U[i*m + j, c] * W[k*n + i, c]
                        try:
                            v_sol, _, _, _ = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)
                            V[j*p + k, :] = v_sol
                        except:
                            pass
                
                # Update W: for each (k,i), minimize
                # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # Fix U and V, solve for W
                # For fixed (k,i): T[i,j,k] = sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c])
                # A_{ki}[j,c] = U[i*m+j,c] * V[j*p+k,c]
                # A_{ki} @ w_{ki} = ones(m)
                
                for k in range(p):
                    for i in range(n):
                        A_mat = np.zeros((m, r))
                        for j in range(m):
                            for c in range(r):
                                A_mat[j, c] = U[i*m + j, c] * V[j*p + k, c]
                        try:
                            w_sol, _, _, _ = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                            W[k*n + i, :] = w_sol
                        except:
                            pass
                
                # Check convergence
                cur_res = tensor_residual(U, V, W)
                if cur_res < best_res:
                    best_res = cur_res
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if cur_res < tol:
                    break
                if abs(prev_res - cur_res) < tol:
                    break
        
        return best_U, best_V, best_W, best_res
    
    # Try to find lower rank decompositions
    # The all-ones tensor of shape (2,4,5) should have low rank
    # Let's try ranks from low to high
    
    best_rank = schoolbook_rank
    best_result = build_schoolbook()
    
    # Try candidate ranks
    candidate_ranks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 24, 30, 35, 36]
    
    for r in candidate_ranks:
        if r >= best_rank:
            break
        
        # Use multiple seeds for robustness
        for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]:
            U, V, W, res = alternating_minimization(
                r, max_iters=3000, tol=1e-10, n_restarts=3, seed=seed
            )
            if res < 1e-8:
                best_rank = r
                best_result = (U, V, W)
                break
        if best_rank == r:
            break
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
