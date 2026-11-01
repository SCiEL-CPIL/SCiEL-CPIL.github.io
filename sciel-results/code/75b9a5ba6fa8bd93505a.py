"""Matrix multiplication 2x3x3 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]
_DEADLINE = 140.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _DEADLINE - _elapsed()

def construct():
    n, m, p = DIMENSIONS
    # Tensor T has shape (n*m, m*p, p*n) with T[i*m+j, j*p+k, k*n+i] = 1
    # We seek rank-r decomposition: T = sum_c U[:,c] * V[:,c] * W[:,c]
    # i.e., T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    T_flat = T.reshape(-1)
    
    # Schoolbook rank = 18
    rank_sb = n * m * p
    U0 = np.zeros((n*m, rank_sb))
    V0 = np.zeros((m*p, rank_sb))
    W0 = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank_sb
    
    def _flatten_params(U, V, W, r):
        return np.concatenate([U.ravel(), V.ravel(), W.ravel()])
    
    def _unflatten(params, r):
        sz_u = n*m*r
        sz_v = m*p*r
        sz_w = p*n*r
        U = params[:sz_u].reshape(n*m, r)
        V = params[sz_u:sz_u+sz_v].reshape(m*p, r)
        W = params[sz_u+sz_v:sz_u+sz_v+sz_w].reshape(p*n, r)
        return U, V, W
    
    def _objective(params, r):
        U, V, W = _unflatten(params, r)
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_approx.reshape(-1) - T_flat
        return np.sum(diff*diff)
    
    def _objective_grad(params, r):
        U, V, W = _unflatten(params, r)
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_approx - T  # shape (n*m, m*p, p*n)
        # grad_U[a,r] = sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
        gU = np.einsum('abc,br,cr->ar', diff, V, W)
        gV = np.einsum('abc,ar,cr->br', diff, U, W)
        gW = np.einsum('abc,ar,br->cr', diff, U, V)
        return np.concatenate([gU.ravel(), gV.ravel(), gW.ravel()])
    
    def _is_exact(U, V, W, tol=1e-9):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(T_approx - T)) < tol
    
    def _try_rank(rank, n_restarts, rng, max_iter=300):
        """Try to find exact decomposition at given rank."""
        if _remaining() < 10:
            return None
        
        for restart in range(n_restarts):
            if _remaining() < 5:
                return None
            
            # Multiple initialization strategies
            if restart % 4 == 0:
                # Random small
                U = rng.standard_normal((n*m, rank)) * 0.3
                V = rng.standard_normal((m*p, rank)) * 0.3
                W = rng.standard_normal((p*n, rank)) * 0.3
            elif restart % 4 == 1:
                # Random moderate
                U = rng.standard_normal((n*m, rank)) * 0.8
                V = rng.standard_normal((m*p, rank)) * 0.8
                W = rng.standard_normal((p*n, rank)) * 0.8
            elif restart % 4 == 2:
                # Structured: perturb schoolbook by removing columns
                if rank <= rank_sb:
                    # Take first 'rank' columns from schoolbook and add small perturbation
                    U = U0[:, :rank].copy() + rng.standard_normal((n*m, rank)) * 0.05
                    V = V0[:, :rank].copy() + rng.standard_normal((m*p, rank)) * 0.05
                    W = W0[:, :rank].copy() + rng.standard_normal((p*n, rank)) * 0.05
                else:
                    U = rng.standard_normal((n*m, rank)) * 0.5
                    V = rng.standard_normal((m*p, rank)) * 0.5
                    W = rng.standard_normal((p*n, rank)) * 0.5
            else:
                # Sparse random: many zeros
                U = rng.standard_normal((n*m, rank)) * 0.5
                V = rng.standard_normal((m*p, rank)) * 0.5
                W = rng.standard_normal((p*n, rank)) * 0.5
                # Zero out ~30% of entries
                mask_u = rng.random((n*m, rank)) < 0.3
                mask_v = rng.random((m*p, rank)) < 0.3
                mask_w = rng.random((p*n, rank)) < 0.3
                U[mask_u] = 0
                V[mask_v] = 0
                W[mask_w] = 0
            
            params = _flatten_params(U, V, W, rank)
            
            try:
                # First pass: L-BFGS-B
                res = minimize(
                    _objective, params, args=(rank,),
                    jac=lambda p, r: _objective_grad(p, r),
                    method='L-BFGS-B',
                    options={'maxiter': max_iter, 'gtol': 1e-10, 'ftol': 1e-12}
                )
                
                if _remaining() < 5:
                    return None
                
                # Check if we got close to zero
                if res.fun < 1e-8:
                    U_f, V_f, W_f = _unflatten(res.x, rank)
                    
                    # Try to round to half-integers
                    U_r = np.round(U_f * 2) / 2
                    V_r = np.round(V_f * 2) / 2
                    W_r = np.round(W_f * 2) / 2
                    if _is_exact(U_r, V_r, W_r, tol=1e-9):
                        return U_r, V_r, W_r
                    
                    # If not exact after rounding, try a polish pass
                    if res.fun < 1e-6:
                        # Polish: tighter tolerance, more iterations
                        res2 = minimize(
                            _objective, res.x, args=(rank,),
                            jac=lambda p, r: _objective_grad(p, r),
                            method='L-BFGS-B',
                            options={'maxiter': 500, 'gtol': 1e-14, 'ftol': 1e-16}
                        )
                        if res2.fun < 1e-10:
                            U_p, V_p, W_p = _unflatten(res2.x, rank)
                            U_pr = np.round(U_p * 2) / 2
                            V_pr = np.round(V_p * 2) / 2
                            W_pr = np.round(W_p * 2) / 2
                            if _is_exact(U_pr, V_pr, W_pr, tol=1e-9):
                                return U_pr, V_pr, W_pr
                            if _is_exact(U_p, V_p, W_p, tol=1e-9):
                                return U_p, V_p, W_p
                    
                    if _is_exact(U_f, V_f, W_f, tol=1e-9):
                        return U_f, V_f, W_f
            except Exception:
                pass
        
        return None
    
    # Try ranks from 17 down to 13
    rng = np.random.default_rng(42)
    
    for rank in range(rank_sb - 1, 12, -1):
        if _remaining() < 15:
            break
        
        # Try multiple restarts for each rank
        # More restarts for lower ranks (higher priority)
        if rank >= 15:
            n_restarts = min(40, max(15, int(_remaining() / 3)))
        else:
            n_restarts = min(25, max(10, int(_remaining() / 4)))
        
        result = _try_rank(rank, n_restarts, rng)
        
        if result is not None:
            U_f, V_f, W_f = result
            best_U, best_V, best_W = U_f, V_f, W_f
            best_rank = rank
            # Found a valid decomposition; try to go lower
            continue
    
    # If we found a better rank, return it
    if best_rank < rank_sb:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        
        # Verify
        U_v = np.array(U_list)
        V_v = np.array(V_list)
        W_v = np.array(W_list)
        T_approx = np.einsum('ar,br,cr->abc', U_v, V_v, W_v)
        if np.max(np.abs(T_approx - T)) < 1e-9:
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    U_list = [[int(x) for x in row] for row in U0]
    V_list = [[int(x) for x in row] for row in V0]
    W_list = [[int(x) for x in row] for row in W0]
    return {"U": U_list, "V": V_list, "W": W_list}
