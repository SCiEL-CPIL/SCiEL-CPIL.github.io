"""Matrix multiplication 3x4x6: find a low-rank exact tensor decomposition.

We start from the schoolbook rank (n*m*p = 72) and attempt to find a lower-rank
decomposition. The multiplication tensor T has T[i*m+j, j*p+k, k*n+i] = 1
for all i in [0,n), j in [0,m), k in [0,p).

A rank-r decomposition writes T as sum_s U[:,s] * V[:,s] * W[:,s] (outer product).
"""

import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 6]

def construct():
    """Return the lowest-rank exact decomposition found within the time budget."""
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    n, m, p = DIMENSIONS
    n_m = n * m      # 12
    m_p = m * p      # 24
    n_p = p * n      # 18
    schoolbook_rank = n * m * p  # 72

    # Build the schoolbook decomposition (guaranteed correct)
    U_sb = np.zeros((n_m, schoolbook_rank))
    V_sb = np.zeros((m_p, schoolbook_rank))
    W_sb = np.zeros((n_p, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1

    # Build the full tensor T as a flat array
    T = np.zeros((n_m, m_p, n_p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    T_flat = T.flatten()

    best_rank = schoolbook_rank
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    def verify_exact(U, V, W, tol=1e-9):
        """Check if U, V, W form an exact decomposition of T."""
        r = U.shape[1]
        T_rec = np.zeros((n_m, m_p, n_p))
        for s in range(r):
            T_rec += np.einsum('a,b,c->abc', U[:, s], V[:, s], W[:, s])
        return np.allclose(T_rec, T, atol=tol)

    def try_rank_decomposition(r, max_iters=1000):
        """Attempt to find a rank-r decomposition using iterative optimization."""
        if r >= best_rank:
            return None
        
        # Initialize with random values
        rng = np.random.RandomState(42)
        U = rng.randn(n_m, r) * 0.1
        V = rng.randn(m_p, r) * 0.1
        W = rng.randn(n_p, r) * 0.1
        
        # Reshape for vectorized computation
        U_flat = U.flatten()
        V_flat = V.flatten()
        W_flat = W.flatten()
        
        def residual(params):
            U = params[:n_m * r].reshape(n_m, r)
            V = params[n_m * r:n_m * r + m_p * r].reshape(m_p, r)
            W = params[n_m * r + m_p * r:].reshape(n_p, r)
            
            # Compute reconstructed tensor
            T_rec = np.zeros((n_m, m_p, n_p))
            for s in range(r):
                T_rec += np.einsum('a,b,c->abc', U[:, s], V[:, s], W[:, s])
            
            return (T_rec - T).flatten()
        
        params = np.concatenate([U_flat, V_flat, W_flat])
        
        try:
            result = least_squares(residual, params, method='lm', max_nfev=max_iters, 
                                  xtol=1e-12, ftol=1e-12, gtol=1e-12)
            
            if result.success or np.linalg.norm(result.fun) < 1e-6:
                U_new = result.x[:n_m * r].reshape(n_m, r)
                V_new = result.x[n_m * r:n_m * r + m_p * r].reshape(m_p, r)
                W_new = result.x[n_m * r + m_p * r:].reshape(n_p, r)
                
                if verify_exact(U_new, V_new, W_new, tol=1e-8):
                    return (U_new, V_new, W_new)
        except Exception:
            pass
        
        return None

    # Try to find lower rank decompositions
    # Start from rank 71 and go down
    for r in range(schoolbook_rank - 1, 50, -1):
        if time.time() > deadline:
            break
        
        result = try_rank_decomposition(r, max_iters=500)
        if result is not None:
            U_new, V_new, W_new = result
            best_rank = r
            best_U = U_new
            best_V = V_new
            best_W = W_new
            break  # Found a lower rank, stop searching
    
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # If we didn't find a better rank, return the schoolbook solution
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
