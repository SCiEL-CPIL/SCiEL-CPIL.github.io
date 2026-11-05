import numpy as np
import scipy.optimize
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Target: 3x3 * 3x6
    # We decompose B into B1 (3x3) and B2 (3x3)
    # C = A B1 + A B2 (blockwise)
    # Rank 3x3*3x3 is 7.
    # So we aim for Rank 14.
    
    def verify(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Step 1: Find rank-7 decomposition for 3x3 * 3x3
    # U1 (9x7), V1 (9x7), W1 (9x7)
    # Target T1[i,j,k] = 1 for all i,j,k in 0..2
    
    def find_rank7():
        if time.time() > deadline - 60:
            return None
            
        n3, m3, p3 = 3, 3, 3
        rank = 7
        dim_U = n3 * m3
        dim_V = m3 * p3
        dim_W = p3 * n3
        
        T_target = np.ones((dim_U, dim_V, dim_W)) # All 1s for 3x3x3x3
        
        def objective(params):
            U = params[:dim_U*rank].reshape(dim_U, rank)
            V = params[dim_U*rank : dim_U*rank + dim_V*rank].reshape(dim_V, rank)
            W = params[dim_U*rank + dim_V*rank :].reshape(dim_W, rank)
            R = np.einsum('ir,jr,kr->ijk', U, V, W)
            return np.sum((R - T_target)**2)
        
        # Initialize with random values
        best_err = float('inf')
        best_params = None
        
        for seed in [42, 1, 2, 3, 4, 5, 10, 100]:
            if time.time() > deadline - 30:
                break
            np.random.seed(seed)
            U = np.random.randn(dim_U, rank)
            V = np.random.randn(dim_V, rank)
            W = np.random.randn(dim_W, rank)
            
            # Normalize
            for r in range(rank):
                U[:, r] /= (np.linalg.norm(U[:, r]) + 1e-10)
                V[:, r] /= (np.linalg.norm(V[:, r]) + 1e-10)
                W[:, r] /= (np.linalg.norm(W[:, r]) + 1e-10)
                
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = scipy.optimize.minimize(
                    objective, params, method='L-BFGS-B',
                    options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.fun < best_err:
                    best_err = res.fun
                    best_params = res.x
                    if best_err < 1e-8:
                        break
            except Exception:
                continue
        
        if best_err < 1e-6:
            U1 = best_params[:dim_U*rank].reshape(dim_U, rank)
            V1 = best_params[dim_U*rank : dim_U*rank + dim_V*rank].reshape(dim_V, rank)
            W1 = best_params[dim_U*rank + dim_V*rank :].reshape(dim_W, rank)
            
            # Snap to half-integers
            U1_s = np.round(U1 * 2) / 2
            V1_s = np.round(V1 * 2) / 2
            W1_s = np.round(W1 * 2) / 2
            
            R = np.einsum('ir,jr,kr->ijk', U1_s, V1_s, W1_s)
            if np.allclose(R, T_target, atol=1e-6):
                return U1_s, V1_s, W1_s
        
        return None

    rank7_sol = find_rank7()
    
    if rank7_sol is not None:
        U1, V1, W1 = rank7_sol
        # Construct Rank 14 for 3x3 * 3x6
        # U shape (9, 14)
        # V shape (18, 14)
        # W shape (18, 14)
        
        rank_total = 14
        U = np.zeros((n*m, rank_total))
        V = np.zeros((m*p, rank_total))
        W = np.zeros((p*n, rank_total))
        
        # First 7 columns: for B1 (k=0,1,2)
        # U[i*3+j, 0:7] = U1[i*3+j, 0:7]
        U[:, 0:7] = U1
        
        # V[j*6+k, 0:7] for k=0,1,2
        # V1 index is j*3+k.
        # We need to map V1 (9x7) to V (18x7)
        # V1[j*3+k] goes to V[j*6+k]
        for j in range(3):
            for k in range(3):
                V[j*6+k, 0:7] = V1[j*3+k, 0:7]
        
        # W[k*3+i, 0:7] for k=0,1,2
        # W1 index is k*3+i.
        # W target index is k*3+i.
        # So W[:, 0:7] = W1
        W[:, 0:7] = W1
        
        # Last 7 columns: for B2 (k=3,4,5)
        # U is same
        U[:, 7:14] = U1
        
        # V[j*6+k, 7:14] for k=3,4,5
        # Map V1[j*3+(k-3)] to V[j*6+k]
        for j in range(3):
            for k in range(3, 6):
                V[j*6+k, 7:14] = V1[j*3+(k-3), 7:14]
        
        # W[k*3+i, 7:14] for k=3,4,5
        # Map W1[(k-3)*3+i] to W[k*3+i]
        for k in range(3, 6):
            for i in range(3):
                W[k*3+i, 7:14] = W1[(k-3)*3+i, 7:14]
                
        # Verify
        if verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Try to optimize Rank 14 directly if rank7 failed
    # Or just return schoolbook
    # Given time, schoolbook is safe.
    
    return make_schoolbook()
# EVOLVE-BLOCK-END
