import numpy as np
from fractions import Fraction
from scipy.optimize import least_squares

def construct():
    n, m, p = DIMENSIONS
    
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    # Precompute indices for the tensor equation
    # We want to minimize || T - T_recon ||_F
    # T_recon[u, v, w] = sum_l U[u,l] V[v,l] W[w,l]
    
    def residuals(params, r):
        U = params[0:dim_u*r].reshape(dim_u, r)
        V = params[dim_u*r:dim_u*r+dim_v*r].reshape(dim_v, r)
        W = params[dim_u*r+dim_v*r:].reshape(dim_w, r)
        
        # Compute T_recon
        # Efficiently: T_recon = np.einsum('ul,vl,wl->uvw', U, V, W)
        T_recon = np.einsum('ul,vl,wl->uvw', U, V, W)
        return (T_recon - T).flatten()

    best_rank = n * m * p
    best_factors = None

    # Try ranks 25 to 29
    for r in range(25, 30):
        # Random initialization
        params = np.random.rand(dim_u*r + dim_v*r + dim_w*r)
        
        try:
            res = least_squares(residuals, params, args=(r,), max_nfev=1000)
            if res.success or res.cost < 1e-10:
                U = res.x[0:dim_u*r].reshape(dim_u, r)
                V = res.x[dim_u*r:dim_u*r+dim_v*r].reshape(dim_v, r)
                W = res.x[dim_u*r+dim_v*r:].reshape(dim_w, r)
                
                # Snap to half-integers
                U_snap = np.zeros_like(U)
                V_snap = np.zeros_like(V)
                W_snap = np.zeros_like(W)
                
                valid = True
                for i in range(U.size):
                    val = U.flat[i]
                    frac = Fraction(val).limit_denominator(2)
                    if abs(frac - val) > 1e-6:
                        valid = False
                        break
                    U.flat[i] = float(frac)
                
                if not valid: continue
                for i in range(V.size):
                    val = V.flat[i]
                    frac = Fraction(val).limit_denominator(2)
                    if abs(frac - val) > 1e-6:
                        valid = False
                        break
                    V.flat[i] = float(frac)
                
                if not valid: continue
                for i in range(W.size):
                    val = W.flat[i]
                    frac = Fraction(val).limit_denominator(2)
                    if abs(frac - val) > 1e-6:
                        valid = False
                        break
                    W.flat[i] = float(frac)
                
                if not valid: continue
                
                # Verify exactness
                T_recon = np.einsum('ul,vl,wl->uvw', U, V, W)
                if np.allclose(T_recon, T, atol=1e-9):
                    best_rank = r
                    best_factors = (U, V, W)
                    break
        except Exception:
            pass

    if best_factors:
        U, V, W = best_factors
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback
        total_rank = n * m * p
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
                    
        return {"U": U, "V": V, "W": W}
