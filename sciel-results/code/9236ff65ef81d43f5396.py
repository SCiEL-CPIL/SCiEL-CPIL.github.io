"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def _verify(U, V, W, n, m, p):
    """Check exact decomposition:
    sum_r W[k*n+i, r] * U[i*m+j1, r] * V[j2*p+k, r] == delta(j1, j2)
    for all i, k, j1, j2
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    s = 0
                    for r in range(rank):
                        s += W[k*n+i][r] * U[i*m+j1][r] * V[j2*p+k][r]
                    target = 1 if j1 == j2 else 0
                    if abs(s - target) > 1e-9:
                        return False
    return True

def _try_rank(r, n, m, p, max_iters=5000, tol=1e-8):
    """Try to find a rank-r decomposition. Returns (U, V, W) if successful, else None."""
    nU = n * m
    nV = m * p
    nW = p * n
    n_vars = r * (nU + nV + nW)
    
    # Number of constraints: n * p * m * m
    n_constraints = n * p * m * m
    
    def residuals(x):
        U = x[:r*nU].reshape(r, nU).T  # shape (nU, r)
        V = x[r*nU:r*(nU+nV)].reshape(r, nV).T  # shape (nV, r)
        W = x[r*(nU+nV):].reshape(r, nW).T  # shape (nW, r)
        
        res = np.zeros(n_constraints)
        idx = 0
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = 0.0
                        for r_idx in range(r):
                            s += W[k*n+i, r_idx] * U[i*m+j1, r_idx] * V[j2*p+k, r_idx]
                        target = 1.0 if j1 == j2 else 0.0
                        res[idx] = s - target
                        idx += 1
        return res
    
    # Try multiple random initializations
    np.random.seed(42)
    best_cost = np.inf
    best_x = None
    
    for attempt in range(20):
        x0 = np.random.randn(n_vars) * 0.5
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=max_iters)
            cost = result.cost
            if cost < best_cost:
                best_cost = cost
                best_x = result.x
            if cost < tol:
                break
        except Exception:
            continue
    
    if best_cost > tol:
        return None
    
    # Extract solution
    x = best_x
    U = x[:r*nU].reshape(r, nU).T
    V = x[r*nU:r*(nU+nV)].reshape(r, nV).T
    W = x[r*(nU+nV):].reshape(r, nW).T
    
    # Round to nearest half-integer if close
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    # Verify the rounded version
    if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
        return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    # Try original (non-rounded)
    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s deadline for safety
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6, 9, 6) = 9
    # But also need at least the "tensor rank" which for matrix mult is at least 
    # the maximum of the three dimensions products' geometric considerations.
    # For 2x3 * 3x3, known bounds suggest rank between ~9 and 18.
    # Let's try from 9 upward.
    
    min_rank = 9
    max_rank = rank_schoolbook  # 18
    
    for r in range(min_rank, max_rank + 1):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank(r, n, m, p)
        if result is not None:
            U, V, W = result
            # Final verification
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Return schoolbook as fallback
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
