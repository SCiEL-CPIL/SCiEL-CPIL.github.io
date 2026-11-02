"""
Optimal rank decomposition for 2x3 by 3x10 matrix multiplication.
The minimum rank for this tensor is 60 (schoolbook).
"""

import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Try to find a decomposition with rank < 60
    for target_rank in [59, 58, 57]:
        sol = _attempt_decomposition(n, m, p, target_rank)
        if sol is not None:
            U, V, W = sol
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook rank 60
    rank = n * m * p
    U = [[0.0] * rank for _ in range(nm)]
    V = [[0.0] * rank for _ in range(mp)]
    W = [[0.0] * rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _attempt_decomposition(n, m, p, target_rank, max_restarts=5):
    """
    Attempt to find a rank-target_rank decomposition.
    Returns (U, V, W) as lists of lists, or None if not found.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    # Build nonzero constraints: (a, b, c) where T[a,b,c] = 1
    nonzero = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nonzero.append((i*m + j, j*p + k, k*n + i))
    
    # Build a sample of zero constraints
    zero_set = set()
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                # Determine if (a,b,c) is a valid (i,j,k) triple
                valid = False
                for i in range(n):
                    if a == i*m:  # j must be 0 for this i? No...
                        pass
                # More careful check:
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if i*m+j == a and j*p+k == b and k*n+i == c:
                                valid = True
                if not valid:
                    zero_set.add((a, b, c))
    
    zero_list = list(zero_set)
    np.random.seed(12345)
    if len(zero_list) > 300:
        idx = np.random.choice(len(zero_list), 300, replace=False)
        zero_sample = [zero_list[i] for i in idx]
    else:
        zero_sample = zero_list
    
    all_constraints = [(a, b, c, 1.0) for (a, b, c) in nonzero] + \
                      [(a, b, c, 0.0) for (a, b, c) in zero_sample]
    
    num_params = (nm + mp + pn) * target_rank
    
    for restart in range(max_restarts):
        np.random.seed(restart * 7919 + 1)
        params = np.random.randn(num_params) * 0.5
        
        def residuals(x):
            U_flat = x[:nm * target_rank].reshape(nm, target_rank)
            V_flat = x[nm*target_rank : nm*target_rank + mp*target_rank].reshape(mp, target_rank)
            W_flat = x[nm*target_rank + mp*target_rank :].reshape(pn, target_rank)
            
            res = np.empty(len(all_constraints))
            for idx, (a, b, c, t) in enumerate(all_constraints):
                res[idx] = np.dot(np.multiply(U_flat[a], V_flat[b]), W_flat[c]) - t
            return res
        
        try:
            from scipy.optimize import least_squares
            res = least_squares(residuals, params, method='lm', max_nfev=200)
            
            if res.cost < 1e-10:
                # Full verification
                U_flat = res.x[:nm * target_rank].reshape(nm, target_rank)
                V_flat = res.x[nm*target_rank : nm*target_rank + mp*target_rank].reshape(mp, target_rank)
                W_flat = res.x[nm*target_rank + mp*target_rank :].reshape(pn, target_rank)
                
                # Check all nonzero
                ok = True
                for (a, b, c) in nonzero:
                    if abs(np.dot(np.multiply(U_flat[a], V_flat[b]), W_flat[c]) - 1.0) > 1e-8:
                        ok = False
                        break
                
                # Check all zero entries (full check)
                if ok:
                    for (a, b, c) in zero_set:
                        if abs(np.dot(np.multiply(U_flat[a], V_flat[b]), W_flat[c])) > 1e-8:
                            ok = False
                            break
                
                if ok:
                    U_list = U_flat.tolist()
                    V_list = V_flat.tolist()
                    W_list = W_flat.tolist()
                    return (U_list, V_list, W_list)
        except Exception:
            continue
    
    return None
# EVOLVE-BLOCK-END
