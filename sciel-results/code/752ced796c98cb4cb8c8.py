import numpy as np
import time
import scipy.optimize as opt

_START = time.time()
_DEADLINE = _START + 110

def _verify(U, V, W, tol=1e-10):
    T = np.einsum('ar,br,cr->abc', U, V, W)
    valid_mask = np.zeros((9, 9, 9), dtype=bool)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                valid_mask[a, b, c] = True
                if abs(T[a, b, c] - 1.0) > tol:
                    return False
    invalid = T[~valid_mask]
    if np.max(np.abs(invalid)) > tol:
        return False
    return True

def _get_rank27():
    U = np.zeros((9, 27), dtype=float)
    V = np.zeros((9, 27), dtype=float)
    W = np.zeros((9, 27), dtype=float)
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, col] = 1.0
                V[j*3+k, col] = 1.0
                W[k*3+i, col] = 1.0
                col += 1
    return U, V, W

def _optimize_rank(rank, max_iters=100, tol=1e-8):
    # Target tensor M
    M = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                M[a, b, c] = 1.0

    # Flatten M for objective
    M_flat = M.flatten()
    
    def objective(params):
        U = params[:9*rank].reshape(9, rank)
        V = params[9*rank:18*rank].reshape(9, rank)
        W = params[18*rank:].reshape(9, rank)
        T = np.einsum('ar,br,cr->abc', U, V, W)
        T_flat = T.flatten()
        # We want T to match M
        # Error is sum of squares
        return np.sum((T_flat - M_flat)**2)

    def jac(params):
        U = params[:9*rank].reshape(9, rank)
        V = params[9*rank:18*rank].reshape(9, rank)
        W = params[18*rank:].reshape(9, rank)
        T = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T - M
        # Gradient of sum(diff^2) wrt U, V, W
        # d/du (diff^2) = 2 * diff * dT/du
        # T_abc = sum_r U_ar V_br W_cr
        # dT_abc / dU_ar = V_br W_cr
        # grad_U_ar = sum_bc 2 * diff_abc * V_br W_cr
        # This can be computed efficiently
        # grad_U = 2 * np.einsum('abc,br,cr->ar', diff, V, W)
        # grad_V = 2 * np.einsum('abc,ar,cr->br', diff, U, W)
        # grad_W = 2 * np.einsum('abc,ar,br->cr', diff, U, V)
        
        grad_U = 2 * np.einsum('abc,br,cr->ar', diff, V, W)
        grad_V = 2 * np.einsum('abc,ar,cr->br', diff, U, W)
        grad_W = 2 * np.einsum('abc,ar,br->cr', diff, U, V)
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for attempt in range(5):
        if time.time() > _DEADLINE - 10:
            break
            
        # Random init
        U_init = np.random.randn(9, rank)
        V_init = np.random.randn(9, rank)
        W_init = np.random.randn(9, rank)
        
        # Scale init to be closer to 1?
        U_init /= np.linalg.norm(U_init)
        V_init /= np.linalg.norm(V_init)
        W_init /= np.linalg.norm(W_init)
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            res = opt.minimize(objective, x0, method='L-BFGS-B', jac=jac, options={'maxiter': 50, 'ftol': 1e-10})
            if res.success or res.fun < best_err:
                best_err = res.fun
                U_opt = res.x[:9*rank].reshape(9, rank)
                V_opt = res.x[9*rank:18*rank].reshape(9, rank)
                W_opt = res.x[18*rank:].reshape(9, rank)
                
                # Round to half-integers
                U_r = np.round(U_opt * 2) / 2
                V_r = np.round(V_opt * 2) / 2
                W_r = np.round(W_opt * 2) / 2
                
                if _verify(U_r, V_r, W_r):
                    return U_r, V_r, W_r
        except Exception as e:
            pass
            
    return None

def construct():
    U27, V27, W27 = _get_rank27()
    
    # Try rank 23
    res = _optimize_rank(23)
    if res:
        return {"U": res[0].tolist(), "V": res[1].tolist(), "W": res[2].tolist()}
        
    # Try rank 25
    if time.time() < _DEADLINE - 20:
        res = _optimize_rank(25)
        if res:
            return {"U": res[0].tolist(), "V": res[1].tolist(), "W": res[2].tolist()}
            
    return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
