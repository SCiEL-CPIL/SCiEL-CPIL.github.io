import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 6]

def _solve_3x3x3_rank23():
    """Find a rank-23 decomposition of 3x3x3 matrix multiplication with half-integer entries."""
    n, m, p = 3, 3, 3
    rank = 23
    
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for seed in range(50):
        np.random.seed(seed)
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        def residuals(params):
            U_ = params[0:n*m*rank].reshape(n*m, rank)
            V_ = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_ = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
            res = np.zeros(n*m*p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U_[i*m+j, r] * V_[j*p+k, r] * W_[k*n+i, r]
                        res[idx] = val - 1.0
                        idx += 1
            return res
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=5000)
            U_ = result.x[0:n*m*rank].reshape(n*m, rank)
            V_ = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_ = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Round to half-integers
            U_r = np.round(U_ * 2) / 2
            V_r = np.round(V_ * 2) / 2
            W_r = np.round(W_ * 2) / 2
            
            # Verify exact decomposition
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U_r[i*m+j, r] * V_r[j*p+k, r] * W_r[k*n+i, r]
                        err += abs(val - 1.0)
            
            if err < 1e-6:
                return U_r, V_r, W_r
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_r, V_r, W_r
        except Exception:
            continue
    
    # Fallback: use trivial rank-27 if no good solution found
    if best_err > 1e-6:
        rank = n*m*p
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
        return U, V, W
    
    return best_U, best_V, best_W

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Get rank-23 decomposition for 3x3x3
    U3, V3, W3 = _solve_3x3x3_rank23()
    rank3 = U3.shape[1]
    
    # If we got a trivial 27-rank solution, just use it for the full problem
    if rank3 == 27:
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build full 3x3x6 decomposition from two 3x3x3 decompositions
    rank = 2 * rank3
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # First half: columns k=0,1,2
    for r in range(rank3):
        U[:, r] = U3[:, r]
        for j in range(m):
            for k in range(3):
                V[j*p+k, r] = V3[j*3+k, r]
        for i in range(n):
            for k in range(3):
                W[k*n+i, r] = W3[k*n+i, r]
    
    # Second half: columns k=3,4,5
    for r in range(rank3):
        U[:, r + rank3] = U3[:, r]
        for j in range(m):
            for k in range(3):
                V[j*p+(k+3), r + rank3] = V3[j*3+k, r]
        for i in range(n):
            for k in range(3):
                W[(k+3)*n+i, r + rank3] = W3[k*n+i, r]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
