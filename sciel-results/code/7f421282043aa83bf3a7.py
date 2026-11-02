import numpy as np
import itertools

DIMENSIONS = [3, 4, 7]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact for matrix multiplication."""
    r = U.shape[1]
    # Check all (i, j, k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(r, n, m, p, max_trials=200):
    """Try to find a rank-r decomposition."""
    U_size = n * m
    V_size = m * p
    W_size = p * n
    
    for trial in range(max_trials):
        np.random.seed(42 + trial)
        U = np.random.randn(U_size, r)
        V = np.random.randn(V_size, r)
        
        # Solve for W: for each (k, i), we have m equations
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for j=0..m-1
        W = np.zeros((W_size, r))
        valid = True
        
        for k in range(p):
            for i in range(n):
                # Build m x r matrix A where A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                
                b = np.ones(m)
                
                # Solve A * w = b
                try:
                    # Check if A has full row rank (m <= r)
                    if m <= r:
                        # Use least squares
                        w, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                        if len(residuals) > 0 and residuals[0] > 1e-10:
                            valid = False
                            break
                    else:
                        # More equations than unknowns, use least squares
                        w, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                        if len(residuals) > 0 and residuals[0] > 1e-10:
                            valid = False
                            break
                    W[k * n + i, :] = w
                except:
                    valid = False
                    break
            
            if not valid:
                break
        
        if not valid:
            continue
        
        # Check if the solution is exact
        if _verify_decomposition(U, V, W, n, m, p):
            # Try to round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
            
            # Try integer rounding
            U_rounded = np.round(U)
            V_rounded = np.round(V)
            W_rounded = np.round(W)
            
            if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
    
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 84
    
    # Try ranks from schoolbook-1 downward
    # For 3x4x7, known bounds: rank is between 3*4*7 - min(3,4,7)^2 + ... 
    # Actually, let's just try a range
    # Theoretical lower bound for matrix multiplication M_{n,m,p} is at least max(n*m, m*p, p*n)
    # For 3,4,7: max(12, 28, 21) = 28
    # But we need to be practical with the 120s limit
    
    # Try from rank_school-1 down to a reasonable lower bound
    # We'll try a few ranks to see if we can find something better
    min_rank_to_try = max(1, rank_school - 30)  # Try up to 30 ranks below schoolbook
    
    for r in range(rank_school - 1, min_rank_to_try - 1, -1):
        result = _try_rank(r, n, m, p, max_trials=50)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    rank = rank_school
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
# EVOLVE-BLOCK-END
