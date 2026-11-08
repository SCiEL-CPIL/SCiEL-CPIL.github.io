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
    # T[a, b, c] = 1 if a, b, c correspond to i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
    # Note: c index in prompt is k*n+i.
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
        
        # Compute tensor product sum
        # Reshape for broadcasting
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # We want sum_r U[a,r] V[b,r] W[c,r]
        # This is like a batched dot product?
        # We can use einsum
        res = np.einsum('ar,br,cr->abc', U, V, W)
        return res.flatten() - T_flat

    def try_rank(R, num_restarts=5):
        for restart in range(num_restarts):
            # Random initialization
            # Scale small to help convergence
            x0 = np.random.randn(3 * nm * R) * 0.1
            
            try:
                sol = least_squares(residuals, x0, method='trf', max_nfev=2000)
            except Exception:
                continue
                
            if sol.success and sol.cost < 1e-8:
                # Extract U, V, W
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

    # Try ranks from 19 to 26
    for R in range(19, 27):
        result = try_rank(R, num_restarts=3 if R > 19 else 5)
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
