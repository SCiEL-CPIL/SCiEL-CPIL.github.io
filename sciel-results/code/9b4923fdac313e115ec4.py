import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target tensor T
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    T_flat = T.flatten()
    
    def residuals(x):
        R = len(x) // 3
        U = x[:nm*R].reshape(nm, R)
        V = x[nm*R:2*nm*R].reshape(mp, R)
        W = x[2*nm*R:3*nm*R].reshape(pn, R)
        res = np.einsum('ar,br,cr->abc', U, V, W)
        return res.flatten() - T_flat

    def try_rank(R, num_restarts=10, sparsity=0.1):
        for restart in range(num_restarts):
            # Sparse random initialization
            x0 = np.random.randn(3 * nm * R)
            # Set some to zero to encourage sparsity?
            # Actually, just scaling is often enough.
            # Let's try a mix of 0 and small values.
            mask = np.random.rand(3 * nm * R) < sparsity
            x0 = np.where(mask, x0 * 0.1, 0.0)
            
            try:
                sol = least_squares(residuals, x0, method='trf', max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            except Exception:
                continue
                
            if sol.success and sol.cost < 1e-8:
                U = sol.x[:nm*R].reshape(nm, R)
                V = sol.x[nm*R:2*nm*R].reshape(mp, R)
                W = sol.x[2*nm*R:3*nm*R].reshape(pn, R)
                
                # Round to nearest half-integer
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify
                res_check = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                if np.allclose(res_check, T, atol=1e-9):
                    return U_r, V_r, W_r
        return None

    # Try ranks from 21 to 26
    # Rank 21 is a known upper bound for 3x3? 
    # If we find 21, score is 1.28.
    for R in range(21, 27):
        result = try_rank(R, num_restarts=5)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
