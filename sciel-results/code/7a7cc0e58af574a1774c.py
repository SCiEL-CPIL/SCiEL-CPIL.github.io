import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 96
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115
    
    # Build target tensor (sparse, only 96 non-zeros)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook fallback
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    
    # Precompute reshaped T for fast ALS
    # T shape: (n*m, m*p, p*n)
    # For U update: fix V,W. T[i,:,:] = U[i,:] @ (V*W^T)
    # M_U[r, (j,k)] = V[j,r]*W[k,r], shape (rank, m*p*p*n)
    # T_U[i, (j,k)] = T[i,j,k], shape (n*m, m*p*p*n)
    
    def fast_als(rank, U_init, V_init, W_init, max_iters=200, tol=1e-10):
        """Fast ALS using precomputed structures."""
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        # Precompute T reshapes
        # T_U: (n*m, m*p*p*n) where T_U[i, j*p*n+k] = T[i,j,k]
        T_U = np.zeros((n * m, m * p * p * n))
        for i in range(n * m):
            for j in range(m * p):
                for k in range(p * n):
                    T_U[i, j * p * n + k] = T[i, j, k]
        
        # T_V: (m*p, n*m*p*n) where T_V[j, i*p*n+k] = T[i,j,k]
        T_V = np.zeros((m * p, n * m * p * n))
        for i in range(n * m):
            for j in range(m * p):
                for k in range(p * n):
                    T_V[j, i * p * n + k] = T[i, j, k]
        
        # T_W: (p*n, n*m*m*p) where T_W[k, i*m*p+j] = T[i,j,k]
        T_W = np.zeros((p * n, n * m * m * p))
        for i in range(n * m):
            for j in range(m * p):
                for k in range(p * n):
                    T_W[k, i * m * p + j] = T[i, j, k]
        
        for it in range(max_iters):
            if time.time() > deadline:
                return False, U, V, W
            
            # Update U: T_U = U @ M_U^T, where M_U[r, (j,k)] = V[j,r]*W[k,r]
            M_U = np.einsum('jr,kr->rjk', V, W).reshape(rank, m * p * p * n)
            try:
                U_new = np.linalg.lstsq(M_U.T, T_U.T, rcond=None)[0].T
                U = U_new
            except:
                break
            
            # Update V: T_V = V @ M_V^T, where M_V[r, (i,k)] = U[i,r]*W[k,r]
            M_V = np.einsum('ir,kr->rik', U, W).reshape(rank, n * m * p * n)
            try:
                V_new = np.linalg.lstsq(M_V.T, T_V.T, rcond=None)[0].T
                V = V_new
            except:
                break
            
            # Update W: T_W = W @ M_W^T, where M_W[r, (i,j)] = U[i,r]*V[j,r]
            M_W = np.einsum('ir,jr->rij', U, V).reshape(rank, n * m * m * p)
            try:
                W_new = np.linalg.lstsq(M_W.T, T_W.T, rcond=None)[0].T
                W = W_new
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
    
    # Try structured initializations based on Kronecker products
    # The matrix multiplication tensor can be written using Kronecker structure
    # Try ranks: 88, 80, 72, 64
    
    target_ranks = [88, 84, 80, 76, 72, 68, 64]
    
    for rank in target_ranks:
        if time.time() > deadline:
            break
        
        # Multiple random initializations
        for seed in range(40):
            if time.time() > deadline:
                break
            
            np.random.seed(seed)
            U_init = np.random.randn(n * m, rank) * 0.3
            V_init = np.random.randn(m * p, rank) * 0.3
            W_init = np.random.randn(p * n, rank) * 0.3
            
            success, U, V, W = fast_als(rank, U_init, V_init, W_init, max_iters=150)
            
            if success:
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                S = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                if np.allclose(S, T, atol=1e-10):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                if np.allclose(S, T, atol=1e-6):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Try starting from schoolbook with column merging
        if rank < rank_schoolbook:
            for trial in range(15):
                if time.time() > deadline:
                    break
                np.random.seed(5000 + trial)
                # Randomly select columns from schoolbook
                cols = np.random.choice(rank_schoolbook, rank, replace=False)
                U_init = U_school[:, cols]
                V_init = V_school[:, cols]
                W_init = W_school[:, cols]
                
                success, U, V, W = fast_als(rank, U_init, V_init, W_init, max_iters=100)
                if success:
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    S = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                    if np.allclose(S, T, atol=1e-10):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    return best_result
# EVOLVE-BLOCK-END
