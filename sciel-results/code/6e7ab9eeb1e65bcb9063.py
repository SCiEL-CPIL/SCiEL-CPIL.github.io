import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115  # 115s out of 120s
    
    # Construct the target tensor T
    # T has shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Fallback to schoolbook
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, column] = 1
                V_school[j * p + k, column] = 1
                W_school[k * n + i, column] = 1
                column += 1
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_schoolbook
    
    def try_rank(rank, U_init, V_init, W_init, max_iters=500, tol=1e-10):
        """Try to refine a decomposition using ALS. Returns (success, U, V, W)."""
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                return False, U, V, W
            
            # Update U: fix V, W
            # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # For each i: T[i,:,:] = U[i,:] @ (V * W^T) where (V*W^T)[r, (j,k)] = V[j,r]*W[k,r]
            # Reshape V and W to form the matrix M_U of shape (rank, m*p*p*n)
            M_U = np.einsum('jr,kr->rjk', V, W).reshape(rank, m * p * p * n)
            T_U = T.reshape(n * m, m * p * p * n)
            # Solve: T_U = U @ M_U^T => U = T_U @ (M_U^T)^+ 
            # Use least squares
            try:
                U_new, _, _, _ = np.linalg.lstsq(M_U.T, T_U.T, rcond=None)
                U = U_new.T
            except:
                break
            
            # Update V: fix U, W
            M_V = np.einsum('ir,kr->rik', U, W).reshape(rank, n * m * p * n)
            T_V = T.reshape(m * p, n * m * p * n)
            # Need to permute T_V properly
            # T[i,j,k] with i from U, j from V, k from W
            # For V update: T[j,:,:] = V[j,:] @ (U * W^T) where (U*W^T)[r,(i,k)] = U[i,r]*W[k,r]
            # So M_V[r, (i,k)] = U[i,r]*W[k,r], shape (rank, n*m*p*n)
            # T_V[j, (i,k)] = T[i,j,k], shape (m*p, n*m*p*n)
            T_V = np.zeros((m * p, n * m * p * n))
            for j in range(m * p):
                for i in range(n * m):
                    for k in range(p * n):
                        T_V[j, i * p * n + k] = T[i, j, k]
            M_V = np.einsum('ir,kr->rik', U, W).reshape(rank, n * m * p * n)
            try:
                V_new, _, _, _ = np.linalg.lstsq(M_V.T, T_V.T, rcond=None)
                V = V_new.T
            except:
                break
            
            # Update W: fix U, V
            # T[k,:,:] = W[k,:] @ (U * V^T) where (U*V^T)[r,(i,j)] = U[i,r]*V[j,r]
            T_W = np.zeros((p * n, n * m * m * p))
            for k in range(p * n):
                for i in range(n * m):
                    for j in range(m * p):
                        T_W[k, i * m * p + j] = T[i, j, k]
            M_W = np.einsum('ir,jr->rij', U, V).reshape(rank, n * m * m * p)
            try:
                W_new, _, _, _ = np.linalg.lstsq(M_W.T, T_W.T, rcond=None)
                W = W_new.T
            except:
                break
            
            # Check convergence
            S = np.einsum('ir,jr,kr->ijk', U, V, W)
            if np.linalg.norm(T - S) < tol:
                return True, U, V, W
        
        S = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.linalg.norm(T - S) < 1e-8:
            return True, U, V, W
        return False, U, V, W
    
    # Try various ranks and initializations
    target_ranks = [64, 56, 48, 40, 32]
    
    for rank in target_ranks:
        if time.time() > deadline:
            break
        
        # Try multiple initializations
        for init_seed in range(30):
            if time.time() > deadline:
                break
            
            np.random.seed(init_seed)
            
            # Random initialization
            U_init = np.random.randn(n * m, rank) * 0.5
            V_init = np.random.randn(m * p, rank) * 0.5
            W_init = np.random.randn(p * n, rank) * 0.5
            
            success, U, V, W = try_rank(rank, U_init, V_init, W_init, max_iters=300)
            
            if success:
                # Round to nearest half-integer
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify
                S = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
                if np.allclose(S, T, atol=1e-10):
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                else:
                    # Return unrounded if close enough
                    if np.allclose(S, T, atol=1e-6):
                        return {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
        
        # Also try starting from schoolbook and merging
        if rank < rank_schoolbook:
            # Try to find a subset of schoolbook columns that spans the same space
            # This is a combinatorial problem, so try random subsets
            for trial in range(10):
                if time.time() > deadline:
                    break
                
                np.random.seed(1000 + trial)
                # Select random columns from schoolbook
                cols = np.random.choice(rank_schoolbook, rank, replace=False)
                U_init = U_school[:, cols]
                V_init = V_school[:, cols]
                W_init = W_school[:, cols]
                
                success, U, V, W = try_rank(rank, U_init, V_init, W_init, max_iters=200)
                
                if success:
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    S = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
                    if np.allclose(S, T, atol=1e-10):
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
    
    return best_result
# EVOLVE-BLOCK-END
