"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]
import time
import numpy as np
from scipy.optimize import least_squares

# EVOLVE-BLOCK-START

def _verify(U, V, W, tol=1e-6):
    """Check if U, V, W exactly decompose the 3x3x3 matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i,j,k] = 1 if (i,j,k) corresponds to a valid multiplication term, else 0
    # Actually, the matrix multiplication tensor is: C[i,k] = sum_j A[i,j]*B[j,k]
    # The decomposition says: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal
    # the matrix multiplication tensor M[i,j,k] where M[i,j,k] = 1 if this is a valid term
    # Wait, let me think more carefully.
    # The tensor for 3x3 matrix multiplication has entries T[i,j,k] where i is row of A (and C),
    # j is col of A (and row of B), k is col of B (and C).
    # T[i,j,k] = 1 for all valid (i,j,k) in [0,n)x[0,m)x[0,p) since each product A[i,j]*B[j,k]
    # contributes to C[i,k].
    # The decomposition: for each c, we have vectors u_c in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
    # The tensor reconstructed is: sum_c outer(u_c, v_c, w_c)
    # where (outer(u,v,w))[i,j,k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
    # We need this to equal the matrix multiplication tensor which has all 1s for valid indices.
    # Actually no - the matrix multiplication tensor has T[i,j,k] = 1 for all (i,j,k) since
    # C[i,k] = sum_j A[i,j]*B[j,k], so each (i,j,k) triple contributes with coefficient 1.
    
    # Let's build the target tensor
    target = np.ones((n, m, p))
    
    # Build reconstructed tensor
    rec = np.zeros((n, m, p))
    for c in range(rank):
        u = U[:, c]
        v = V[:, c]
        w = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    rec[i, j, k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
    
    diff = np.max(np.abs(rec - target))
    return diff < tol

def _try_rank(rank, max_iter=5000, n_restarts=5):
    """Try to find a rank decomposition using numerical optimization."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    total_vars = rank * (nm + mp + pn)
    
    best_diff = float('inf')
    best_result = None
    
    for restart in range(n_restarts):
        # Initialize with random values
        np.random.seed(42 + restart)
        U_init = np.random.randn(nm, rank) * 0.1
        V_init = np.random.randn(mp, rank) * 0.1
        W_init = np.random.randn(pn, rank) * 0.1
        
        # Scale to make initial residual reasonable
        U_init *= (1.0 / np.sqrt(rank))
        V_init *= (1.0 / np.sqrt(rank))
        W_init *= (1.0 / np.sqrt(rank))
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def residuals(x):
            U = x[:nm*rank].reshape(nm, rank)
            V = x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
            W = x[nm*rank+mp*rank:].reshape(pn, rank)
            # Build reconstructed tensor
            rec = np.zeros((n, m, p))
            for c in range(rank):
                u = U[:, c]
                v = V[:, c]
                w = W[:, c]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            rec[i, j, k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
            target = np.ones((n, m, p))
            return (rec - target).flatten()
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=max_iter, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            final_diff = np.max(np.abs(result.fun))
            if final_diff < best_diff:
                best_diff = final_diff
                U = result.x[:nm*rank].reshape(nm, rank)
                V = result.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                W = result.x[nm*rank+mp*rank:].reshape(pn, rank)
                best_result = (U, V, W)
        except Exception:
            pass
    
    if best_result is not None and best_diff < 1e-6:
        U, V, W = best_result
        # Try to round to nice fractions
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        if _verify(U_round, V_round, W_round, tol=1e-6):
            return U_round.tolist(), V_round.tolist(), W_round.tolist()
        # Try rounding to integers
        U_round = np.round(U)
        V_round = np.round(V)
        W_round = np.round(W)
        if _verify(U_round, V_round, W_round, tol=1e-6):
            return U_round.tolist(), V_round.tolist(), W_round.tolist()
        # Return as-is if close enough
        if best_diff < 1e-6:
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _schoolbook_decomposition():
    """Return the schoolbook rank-27 decomposition."""
    n, m, p = DIMENSIONS
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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # First, try to find a rank-23 decomposition (Strassen-like for 3x3)
    # Actually, the known minimal rank for 3x3 matrix multiplication is 23
    # Let's try rank 23 first, then 24, etc.
    
    for rank in [23, 24, 25, 26]:
        if time.time() > deadline:
            break
        result = _try_rank(rank, max_iter=3000, n_restarts=3)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, tol=1e-5):
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook rank-27
    return _schoolbook_decomposition()

# EVOLVE-BLOCK-END
