# EVOLVE-BLOCK-START
import time
import numpy as np
import scipy

_start_time = time.time()
_DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    U_t = U_np.reshape(n, m, rank)
    V_t = V_np.reshape(m, p, rank)
    W_t = W_np.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    
    # Check if T is all ones
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _svd_init(n, m, p, rank, seed=0):
    np.random.seed(seed)
    # Create a random tensor
    T = np.random.randn(n, m, p)
    # Reshape to matrix for SVD
    # We want to approximate T as sum of rank-1 tensors
    # Unfold along the 3rd mode (p)
    # T_p[i, j, k] -> Matrix of size (n*m) x p
    T_mat = T.reshape(n * m, p)
    U_mat, S, Vt = np.linalg.svd(T_mat, full_matrices=False)
    # Take top 'rank' components
    U_mat = U_mat[:, :rank]
    S = S[:rank]
    Vt = Vt[:rank, :]
    
    # Reconstruct U, V, W
    # U[i, j, r] = U_mat[i*m+j, r] * S[r]^{1/3}
    # V[j, k, r] = ?
    # This SVD only gives a decomposition of the unfolded matrix, not the full tensor.
    # It's a rough initialization.
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # This initialization is weak. Let's use random instead if SVD doesn't help.
    return None

def _try_rank(n, m, p, rank, max_iter=500, tol=1e-10, seed=0):
    if time.time() > _DEADLINE - 5.0:
        return None
    
    np.random.seed(seed)
    
    # Random Initialization
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5.0:
            return None
        
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    for r in range(rank):
                        A[k, r] = V[j * p + k, r] * W[k * n + i, r]
                try:
                    x = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    U[i * m + j, :] = x
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    for r in range(rank):
                        B[i, r] = U[i * m + j, r] * W[k * n + i, r]
                try:
                    x = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                    V[j * p + k, :] = x
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, rank))
                for j in range(m):
                    for r in range(rank):
                        C[j, r] = U[i * m + j, r] * V[j * p + k, r]
                try:
                    x = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                    W[k * n + i, :] = x
                except:
                    pass
        
        if it % 10 == 0:
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return U.tolist(), V.tolist(), W.tolist()
            
    return None

def construct():
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = n * m * p
    
    # Try ranks from 18 to 40
    for rank in range(18, 41):
        if time.time() > _DEADLINE - 10.0:
            break
        for seed in range(20):
            if time.time() > _DEADLINE - 5.0:
                break
            result = _try_rank(n, m, p, rank, max_iter=100, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {"U": U, "V": V, "W": W}
                    break
        
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
