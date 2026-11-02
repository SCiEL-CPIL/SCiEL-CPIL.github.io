# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 140  # 10s safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Check if U,V,W form a valid decomposition."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Vectorized check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _polish(U, V, W, n, m, p, max_iters=200, tol=1e-12):
    """Final high-precision polish pass with tighter tolerance."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 2:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k,:] = V[j*p+k,:] * W[k*n+i,:]
                u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=1e-14)
                U[i*m+j,:] = u
        
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i,:] = U[i*m+j,:] * W[k*n+i,:]
                v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=1e-14)
                V[j*p+k,:] = v
        
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j,:] = U[i*m+j,:] * V[j*p+k,:]
                w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=1e-14)
                W[k*n+i,:] = w
        
        # Check convergence
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=tol):
            return (U.tolist(), V.tolist(), W.tolist())
    
    return None

def _als_step_vectorized(U, V, W, n, m, p, rank):
    """
    Perform one ALS iteration using vectorized operations where possible.
    Returns updated U, V, W.
    """
    U_new = U.copy()
    V_new = V.copy()
    W_new = W.copy()
    
    # Update U: U[i, j, :]
    # For each i, j:
    # A[k, c] = V[j, k, c] * W[k, i, c]
    # Solve A @ u = 1
    # V is (m, p, rank), W is (p, n, rank)
    # We can iterate i, j but vectorize over k? No, lstsq is per (i,j).
    # But we can precompute products?
    # V[j, :, :] is (p, rank). W[:, i, :] is (p, rank).
    # A = V[j, :, :] * W[:, i, :] (element-wise)
    
    for i in range(n):
        for j in range(m):
            # A shape (p, rank)
            A = V[j, :, :] * W[:, i, :]
            # Solve A @ u = 1
            # Use lstsq
            try:
                u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=1e-12)
                U_new[i, j, :] = u
            except:
                pass

    for j in range(m):
        for k in range(p):
            # A shape (n, rank)
            # A[i, c] = U[i, j, c] * W[k, i, c]
            # U[:, j, :] is (n, rank). W[k, :, :] is (n, rank).
            A = U[:, j, :] * W[k, :, :]
            try:
                v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=1e-12)
                V_new[j, k, :] = v
            except:
                pass

    for k in range(p):
        for i in range(n):
            # A shape (m, rank)
            # A[j, c] = U[i, j, c] * V[j, k, c]
            # U[i, :, :] is (m, rank). V[:, k, :] is (m, rank).
            A = U[i, :, :] * V[:, k, :]
            try:
                w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=1e-12)
                W_new[k, i, :] = w
            except:
                pass

    return U_new, V_new, W_new

def _try_rank_structured(rank, n, m, p, max_iters=2000, seed=42):
    """Try to find a rank-rank decomposition using structured initialization."""
    rng = np.random.RandomState(seed)
    
    # Initialize with a structured pattern
    U = np.zeros((n, m, rank))
    V = np.zeros((m, p, rank))
    W = np.zeros((p, n, rank))
    
    # Strategy 0: Random
    U = rng.randn(n, m, rank) * 0.1
    V = rng.randn(m, p, rank) * 0.1
    W = rng.randn(p, n, rank) * 0.1
    
    # Strategy 1: Structured Basis
    # This is often a better starting point for tensor decompositions
    U_struct = np.zeros((n, m, rank))
    V_struct = np.zeros((m, p, rank))
    W_struct = np.zeros((p, n, rank))
    
    for c in range(rank):
        i_idx = c % n
        j_idx = (c // n) % m
        k_idx = (c // (n*m)) % p
        
        U_struct[i_idx, j_idx, c] = 1.0
        V_struct[j_idx, k_idx, c] = 1.0
        W_struct[k_idx, i_idx, c] = 1.0
        
    # Mix random and structured
    U = 0.5 * U + 0.5 * U_struct
    V = 0.5 * V + 0.5 * V_struct
    W = 0.5 * W + 0.5 * W_struct
    
    # Run ALS
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            return None
        
        U, V, W = _als_step_vectorized(U, V, W, n, m, p, rank)
        
        # Check convergence
        if _verify(U.reshape(n*m, rank).tolist(), V.reshape(m*p, rank).tolist(), W.reshape(p*n, rank).tolist(), n, m, p, tol=1e-8):
            # Polish
            polished = _polish(U.reshape(n*m, rank).tolist(), V.reshape(m*p, rank).tolist(), W.reshape(p*n, rank).tolist(), n, m, p)
            if polished is not None:
                return polished
            return (U.reshape(n*m, rank).tolist(), V.reshape(m*p, rank).tolist(), W.reshape(p*n, rank).tolist())
            
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    lower_bound = max(n*m, m*p, n*p)
    
    best_U, best_V, best_W = None, None, None
    best_rank = schoolbook_rank
    
    # Try ranks from lower bound up to schoolbook
    # We prioritize lower ranks
    for rank in range(lower_bound, best_rank + 1):
        if time.time() > _DEADLINE - 5:
            break
        
        # Try multiple seeds
        # Limit seeds per rank to avoid spending too much time on impossible ranks
        max_seeds = 10 if rank < 20 else 5
        
        for seed in range(max_seeds):
            if time.time() > _DEADLINE - 5:
                break
            result = _try_rank_structured(rank, n, m, p, max_iters=1000, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p):
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    break
    
    if best_U is None:
        # Fall back to schoolbook
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
        return {"U": U, "V": V, "W": W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
