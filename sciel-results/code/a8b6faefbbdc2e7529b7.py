import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Build the target tensor T
    T = np.zeros((size_u, size_v, size_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def check_decomposition(U, V, W, tol=1e-9):
        R = np.einsum('ar,br,cr->abc', U, V, W)
        if not np.allclose(R, T, atol=tol):
            return False
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        R_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(R_r, T, atol=1e-9):
            return True
        return False

    def polish_with_scipy(U, V, W, rank):
        """Use scipy to minimize error on flattened params, then round."""
        def objective(params):
            U_flat = params[:size_u * rank].reshape(size_u, rank)
            V_flat = params[size_u * rank : size_u * rank + size_v * rank].reshape(size_v, rank)
            W_flat = params[size_u * rank + size_v * rank :].reshape(size_w, rank)
            R = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
            return np.sum((R - T) ** 2)
        
        def jac(params):
            # Numerical gradient or analytical? Use finite differences for simplicity in this scope
            # Or just let L-BFGS-B use finite differences
            pass

        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            # Use L-BFGS-B
            res = opt.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12})
            x_opt = res.x
            U_opt = x_opt[:size_u * rank].reshape(size_u, rank)
            V_opt = x_opt[size_u * rank : size_u * rank + size_v * rank].reshape(size_v, rank)
            W_opt = x_opt[size_u * rank + size_v * rank :].reshape(size_w, rank)
            
            # Round to half-integers
            U_r = np.round(U_opt * 2) / 2
            V_r = np.round(V_opt * 2) / 2
            W_r = np.round(W_opt * 2) / 2
            
            if check_decomposition(U_r, V_r, W_r):
                return U_r, V_r, W_r
        except Exception:
            pass
        return None

    def find_decomposition(rank, max_attempts=30, max_iter=50):
        if time.time() > deadline:
            return None
            
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return None
                
            # Random initialization
            U = np.random.randn(size_u, rank)
            V = np.random.randn(size_v, rank)
            W = np.random.randn(size_w, rank)
            
            # ALS iterations
            prev_err = np.inf
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    return None
                    
                # Update U
                M_vw = np.einsum('br,cr->bcr', V, W).reshape(size_v * size_w, rank)
                T_1 = T.reshape(size_u, size_v * size_w)
                U, _, _, _ = np.linalg.lstsq(M_vw.T, T_1.T, rcond=None)
                U = U.T
                
                # Update V
                M_ua = np.einsum('ar,cr->acr', U, W).reshape(size_u * size_w, rank)
                T_2 = T.transpose(1, 0, 2).reshape(size_v, size_u * size_w)
                V, _, _, _ = np.linalg.lstsq(M_ua.T, T_2.T, rcond=None)
                V = V.T
                
                # Update W
                M_uv = np.einsum('ar,br->abr', U, V).reshape(size_u * size_v, rank)
                T_3 = T.transpose(0, 2, 1).reshape(size_w, size_u * size_v)
                W, _, _, _ = np.linalg.lstsq(M_uv.T, T_3.T, rcond=None)
                W = W.T
                
                R = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(R - T)
                
                if abs(prev_err - err) < 1e-12:
                    break
                prev_err = err
            
            # Try to polish with scipy
            polished = polish_with_scipy(U, V, W, rank)
            if polished is not None:
                U_r, V_r, W_r = polished
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
                
            # Check direct rounding
            if check_decomposition(U, V, W):
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
        return None

    best_result = None
    # Try ranks from 23 upward
    for rank in [23, 24, 25, 26, 27]:
        if time.time() > deadline - 20:
            break
        res = find_decomposition(rank, max_attempts=50, max_iter=40)
        if res is not None:
            best_result = res
            break

    if best_result is not None:
        return best_result
    
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
