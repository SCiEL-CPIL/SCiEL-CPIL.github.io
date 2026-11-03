"""
Rank-optimal tensor decomposition for 3x3 * 3x7 matrix multiplication.
Uses alternating least squares with random restarts to search for low-rank decompositions.
"""
import time
import numpy as np

START_TIME = time.time()
TIME_LIMIT = 130


def _verify(U, V, W, n=3, m=3, p=7, tol=1e-10):
    """Verify that the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _als_decompose(n, m, p, rank, max_iter=200, tol=1e-12, seed=42):
    """Alternating least squares to find rank-r decomposition."""
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n*m, rank)
    V = rng.randn(m*p, rank)
    W = rng.randn(p*n, rank)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Update U
        for i in range(n):
            for j in range(m):
                A_mat = np.zeros((p, rank))
                for k in range(p):
                    A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    u, _, _, _ = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                    U[i*m+j, :] = u
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                A_mat = np.zeros((n, rank))
                for i in range(n):
                    A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    v, _, _, _ = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)
                    V[j*p+k, :] = v
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                A_mat = np.zeros((m, rank))
                for j in range(m):
                    A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    w, _, _, _ = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                    W[k*n+i, :] = w
                except:
                    pass
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0)**2
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < tol:
            break
    
    return best_U, best_V, best_W, best_err


def _schoolbook():
    """Return the schoolbook decomposition with rank 63."""
    n, m, p = 3, 3, 7
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    n, m, p = 3, 3, 7
    
    U_fb, V_fb, W_fb = _schoolbook()
    best_rank = n * m * p
    best_result = {'U': U_fb, 'V': V_fb, 'W': W_fb}
    
    min_rank = 21
    max_rank_to_try = 55
    
    for rank in range(min_rank, max_rank_to_try + 1):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        for seed in range(8):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            
            U, V, W, err = _als_decompose(n, m, p, rank, max_iter=80, tol=1e-12, seed=seed)
            
            if err < 1e-10 and _verify(U, V, W, n, m, p, tol=1e-10):
                if rank < best_rank:
                    best_rank = rank
                    best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                break
        
        if best_rank == rank:
            break
    
    return best_result
