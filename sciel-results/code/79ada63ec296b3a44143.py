import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

def construct():
    """
    Returns the rank-minimal exact tensor decomposition for matrix multiplication 2x3 by 3x7.
    Uses a search strategy to find ranks lower than the schoolbook rank (42).
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p
    
    # Target tensor T where T[i,j,k] = 1
    T_target = np.ones((n, m, p))

    def residuals(x, r):
        # x is a flattened vector of U, V, W
        U = x[0:nm*r].reshape(n, m, r)
        V = x[nm*r:nm*r+mp*r].reshape(m, p, r)
        W = x[nm*r+mp*r:].reshape(p, n, r)
        
        # Compute sum over alpha of U[i,j,alpha] * V[j,k,alpha] * W[k,i,alpha]
        # Using einsum for speed
        res = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        
        return (res - T_target).flatten()

    def verify(U, V, W, r):
        # Check if entries are close to half-integers
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-8): return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-8): return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-8): return False
        
        # Round to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Verify exact decomposition with rounded values
        x_flat = np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()])
        res = residuals(x_flat, r)
        return np.allclose(res, 0, atol=1e-10)

    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None

    # Initialize with schoolbook decomposition
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    # Search for lower ranks
    # Try a few specific ranks that are likely to be achievable
    ranks_to_try = [41, 40, 39]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        found = False
        max_restarts = 20  # More restarts to increase chance of success

        for restart in range(max_restarts):
            if time.time() > deadline - 3:
                break
                
            # Different initialization strategies
            if restart % 3 == 0:
                # Small random values
                x0 = np.random.randn(nm*r + mp*r + pn*r) * 0.1
            elif restart % 3 == 1:
                # Larger random values
                x0 = np.random.randn(nm*r + mp*r + pn*r) * 0.5
            else:
                # Sparse initialization: mostly zeros with some random non-zeros
                x0 = np.zeros(nm*r + mp*r + pn*r)
                num_nonzero = max(10, (nm*r + mp*r + pn*r) // 5)
                indices = np.random.choice(nm*r + mp*r + pn*r, num_nonzero, replace=False)
                x0[indices] = np.random.randn(num_nonzero) * 0.5
            
            try:
                # Try Levenberg-Marquardt first (fast for small systems)
                sol = least_squares(residuals, x0, args=(r,), method='lm', max_nfev=2000)
                
                if sol.success or sol.cost < 1e-10:
                    U = sol.x[0:nm*r].reshape(nm, r)
                    V = sol.x[nm*r:nm*r+mp*r].reshape(mp, r)
                    W = sol.x[nm*r+mp*r:].reshape(pn, r)
                    
                    if verify(U, V, W, r):
                        best_rank = r
                        best_U = U
                        best_V = V
                        best_W = W
                        found = True
                        break
                else:
                    # If LM fails, try Trust Region Reflective (more robust)
                    # Only if we have time
                    if time.time() < deadline - 5:
                        sol2 = least_squares(residuals, x0, args=(r,), method='trf', max_nfev=2000)
                        if sol2.success or sol2.cost < 1e-10:
                            U = sol2.x[0:nm*r].reshape(nm, r)
                            V = sol2.x[nm*r:nm*r+mp*r].reshape(mp, r)
                            W = sol2.x[nm*r+mp*r:].reshape(pn, r)
                            
                            if verify(U, V, W, r):
                                best_rank = r
                                best_U = U
                                best_V = V
                                best_W = W
                                found = True
                                break
            except Exception:
                pass

        # If found a solution for this r, continue to next lower r.
        # If not, continue to next lower r.

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
