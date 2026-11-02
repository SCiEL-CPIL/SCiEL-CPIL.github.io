import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def tensor_product(U, V, W):
        return np.einsum('ar,br,cr->abc', U, V, W)
    
    def verify(U, V, W, tol=1e-9):
        R = tensor_product(U, V, W)
        return np.allclose(R, T, atol=tol, rtol=0)
    
    def snap_to_half_integer(x, tol=1e-6):
        x2 = 2 * x
        rounded = np.round(x2)
        if np.max(np.abs(x2 - rounded)) < tol:
            return rounded / 2.0
        return None
    
    def snap_and_verify(U, V, W, tol=1e-5):
        U_snap = snap_to_half_integer(U, tol)
        V_snap = snap_to_half_integer(V, tol)
        W_snap = snap_to_half_integer(W, tol)
        if U_snap is not None and V_snap is not None and W_snap is not None:
            if verify(U_snap, V_snap, W_snap):
                return U_snap, V_snap, W_snap
        return None
    
    # Schoolbook solution (rank 30)
    U_s = np.zeros((n*m, 30))
    V_s = np.zeros((m*p, 30))
    W_s = np.zeros((p*n, 30))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1
                V_s[j*p+k, col] = 1
                W_s[k*n+i, col] = 1
                col += 1
    best_rank = 30
    best_decomp = (U_s, V_s, W_s)
    
    # Try ranks from 20 up to 29
    for r in range(20, best_rank):
        if time.time() > deadline:
            break
        
        found = False
        num_restarts = 30
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Structured initialization
            if restart == 0:
                # Block-diagonal-like structure
                U = np.random.randn(n*m, r) * 0.3
                V = np.random.randn(m*p, r) * 0.3
                W = np.random.randn(p*n, r) * 0.3
            elif restart == 1:
                # Sparse initialization
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                W = np.zeros((p*n, r))
                for c in range(r):
                    row_u = np.random.randint(0, n*m)
                    row_v = np.random.randint(0, m*p)
                    row_w = np.random.randint(0, p*n)
                    U[row_u, c] = np.random.choice([-1, 1])
                    V[row_v, c] = np.random.choice([-1, 1])
                    W[row_w, c] = np.random.choice([-1, 1])
            elif restart == 2:
                # Symmetric-ish initialization
                U = np.random.randn(n*m, r) * 0.5
                V = np.random.randn(m*p, r) * 0.5
                W = np.random.randn(p*n, r) * 0.5
                # Add some structure
                for c in range(r):
                    U[np.random.randint(0, n*m), c] *= 2
                    V[np.random.randint(0, m*p), c] *= 2
                    W[np.random.randint(0, p*n), c] *= 2
            else:
                # Random with varying scales
                scale = 0.2 + 0.3 * (restart % 5) / 5.0
                U = np.random.randn(n*m, r) * scale
                V = np.random.randn(m*p, r) * scale
                W = np.random.randn(p*n, r) * scale
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def obj(p_arr):
                U_ = p_arr[0:n*m*r].reshape(n*m, r)
                V_ = p_arr[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_ = p_arr[n*m*r + m*p*r:].reshape(p*n, r)
                R = tensor_product(U_, V_, W_)
                return np.sum((R - T)**2)
            
            def grad(p_arr):
                U_ = p_arr[0:n*m*r].reshape(n*m, r)
                V_ = p_arr[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_ = p_arr[n*m*r + m*p*r:].reshape(p*n, r)
                R = tensor_product(U_, V_, W_)
                diff = R - T
                gU = 2 * np.einsum('abc,br,cr->ar', diff, V_, W_)
                gV = 2 * np.einsum('abc,ar,cr->br', diff, U_, W_)
                gW = 2 * np.einsum('abc,ar,br->cr', diff, U_, V_)
                return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
            
            # Try L-BFGS-B
            try:
                res = minimize(obj, params, jac=grad, method='L-BFGS-B', 
                               options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10,
                                        'maxls': 50, 'maxfun': 5000})
                
                if res.fun < 1e-8:
                    U_opt = res.x[0:n*m*r].reshape(n*m, r)
                    V_opt = res.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                    W_opt = res.x[n*m*r + m*p*r:].reshape(p*n, r)
                    
                    result = snap_and_verify(U_opt, V_opt, W_opt)
                    if result is not None:
                        best_rank = r
                        best_decomp = result
                        found = True
                        break
                    
                    # Also try with raw floats if they're very close
                    if verify(U_opt, V_opt, W_opt, 1e-7):
                        result = snap_and_verify(U_opt, V_opt, W_opt, 1e-4)
                        if result is not None:
                            best_rank = r
                            best_decomp = result
                            found = True
                            break
            except Exception:
                pass
            
            # If L-BFGS didn't work, try Nelder-Mead with reduced params
            if time.time() > deadline:
                break
            if obj(params) < 1e-4:
                try:
                    res2 = minimize(obj, params, method='Nelder-Mead',
                                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
                    if res2.fun < 1e-8:
                        U_opt = res2.x[0:n*m*r].reshape(n*m, r)
                        V_opt = res2.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                        W_opt = res2.x[n*m*r + m*p*r:].reshape(p*n, r)
                        result = snap_and_verify(U_opt, V_opt, W_opt)
                        if result is not None:
                            best_rank = r
                            best_decomp = result
                            found = True
                            break
                except Exception:
                    pass
        
        if found:
            break
    
    # Final polish: if we have a near-solution, try to improve
    if best_decomp[0].shape[1] < 30:
        U, V, W = best_decomp
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        try:
            res = minimize(obj, params, jac=grad, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12})
            if res.fun < 1e-10:
                U_opt = res.x[0:n*m*r].reshape(n*m, r)
                V_opt = res.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_opt = res.x[n*m*r + m*p*r:].reshape(p*n, r)
                result = snap_and_verify(U_opt, V_opt, W_opt, 1e-5)
                if result is not None:
                    best_decomp = result
        except Exception:
            pass
    
    U_list = best_decomp[0].tolist()
    V_list = best_decomp[1].tolist()
    W_list = best_decomp[2].tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
