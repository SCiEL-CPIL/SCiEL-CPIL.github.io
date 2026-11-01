import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    max_time = 145.0  # Safety margin for 150s limit
    
    nm, mp, pn = n * m, m * p, p * n
    rank_sb = n * m * p  # 24 for schoolbook
    
    # Build schoolbook decomposition (rank 24)
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    def verify(U, V, W, r):
        """Verify that U, V, W (each of shape (rows, r)) form a valid rank-r
        decomposition of the matrix multiplication tensor with half-integer entries."""
        # Check half-integer constraint
        for c in range(r):
            for val in U[:, c]:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
            for val in V[:, c]:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
            for val in W[:, c]:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
        # Check exact decomposition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    for c in range(r):
                        s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def objective_and_grad(x, r):
        """Objective: sum of squared errors. Returns (f, g) where g is gradient."""
        U_flat = x[:r * nm]
        V_flat = x[r * nm : r * nm + r * mp]
        W_flat = x[r * nm + r * mp : r * nm + r * mp + r * pn]
        
        U = U_flat.reshape(r, nm)
        V = V_flat.reshape(r, mp)
        W = W_flat.reshape(r, pn)
        
        f = 0.0
        grad = np.zeros_like(x)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    for c in range(r):
                        s += U[c, u_idx] * V[c, v_idx] * W[c, w_idx]
                    e = s - 1.0
                    f += e * e
                    # d(f)/d(U[c, u_idx]) = 2*e * V[c, v_idx] * W[c, w_idx]
                    for c in range(r):
                        uv = V[c, v_idx] * W[c, w_idx]
                        vw = U[c, u_idx] * W[c, w_idx]
                        uu = U[c, u_idx] * V[c, v_idx]
                        grad[c * nm + u_idx] += 2 * e * uv
                        grad[r * nm + c * mp + v_idx] += 2 * e * vw
                        grad[r * nm + r * mp + c * pn + w_idx] += 2 * e * uu
        
        return f, grad
    
    def try_rank(r, n_restarts, timeout):
        """Try to find a valid rank-r decomposition using L-BFGS-B."""
        if time.time() - start_time > timeout:
            return None
        
        total_vars = r * (nm + mp + pn)
        
        for restart in range(n_restarts):
            if time.time() - start_time > timeout:
                return None
            
            # Random initialization with small values
            x0 = np.random.uniform(-0.5, 0.5, size=total_vars)
            
            from scipy.optimize import minimize
            try:
                res = minimize(objective_and_grad, x0, args=(r,),
                              method='L-BFGS-B', jac=True,
                              options={'maxiter': 2000, 'ftol': 1e-14,
                                       'gtol': 1e-10, 'maxls': 50})
            except Exception:
                continue
            
            if res.fun < 1e-10:
                U_flat = res.x[:r * nm]
                V_flat = res.x[r * nm : r * nm + r * mp]
                W_flat = res.x[r * nm + r * mp : r * nm + r * mp + r * pn]
                
                U = U_flat.reshape(r, nm)
                V = V_flat.reshape(r, mp)
                W = W_flat.reshape(r, pn)
                
                # Transpose to get (rows, rank) shape for verification
                U_t = U.T.copy()
                V_t = V.T.copy()
                W_t = W.T.copy()
                
                # Round to half-integers
                U_rounded = np.round(U_t * 2) / 2.0
                V_rounded = np.round(V_t * 2) / 2.0
                W_rounded = np.round(W_t * 2) / 2.0
                
                if verify(U_rounded, V_rounded, W_rounded, r):
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
        
        return None
    
    # Try ranks from 23 down to 15, with fewer restarts for lower ranks
    for r in range(rank_sb - 1, 14, -1):
        if time.time() - start_time > max_time - 10:
            break
        
        n_restarts = max(3, 20 - (rank_sb - 1 - r) * 2)
        timeout = max_time - 5
        result = try_rank(r, n_restarts, timeout)
        if result is not None:
            return result
    
    # Return schoolbook decomposition
    return {
        "U": U_sb.T.tolist(),
        "V": V_sb.T.tolist(),
        "W": W_sb.T.tolist()
    }
