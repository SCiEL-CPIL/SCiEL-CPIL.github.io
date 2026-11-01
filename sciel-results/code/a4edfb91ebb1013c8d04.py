"""Matrix multiplication 2x3x3 rank-minimal exact tensor decomposition search."""
import time
import random
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

DIMENSIONS = [2, 3, 3]
# Deadline guard: evaluator kills after 150s; we keep a safety margin.
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
    
    # Start with schoolbook rank = 18
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
    
    # Try to find lower rank via random initialization + gradient-based optimization
    # The objective: minimize sum over (a,b,c) of (T[a,b,c] - sum_r U[a,r]V[b,r]W[c,r])^2
    # with T being the tensor.
    
    def _build_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        return T
    
    T = _build_tensor()
    T_flat = T.reshape(-1)
    n_entries = T_flat.size  # 2*3*3*3*3 = 162
    
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
        # Compute T_approx[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Efficient: for each r, outer product
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
    
    # Try ranks from 17 down to some minimum
    # Known lower bound for 2x3x3: at least 13? Let's try a range.
    # We'll use multiple random restarts for each rank.
    
    max_restarts = 50
    rng = np.random.default_rng(42)
    
    for rank in range(rank_sb - 1, 12, -1):  # Try 17 down to 13
        if _remaining() < 10:
            break
        found = False
        for restart in range(max_restarts):
            if _remaining() < 5:
                break
            # Random initialization
            U = rng.standard_normal((n*m, rank)) * 0.5
            V = rng.standard_normal((m*p, rank)) * 0.5
            W = rng.standard_normal((p*n, rank)) * 0.5
            params = _flatten_params(U, V, W, rank)
            
            try:
                res = minimize(
                    _objective, params, args=(rank,),
                    jac=lambda p, r: _objective_grad(p, r),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-12}
                )
                if _remaining() < 5:
                    break
                final_obj = res.fun
                # Check if we got close to zero (exact decomposition)
                if final_obj < 1e-10:
                    U, V, W = _unflatten(res.x, rank)
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    found = True
                    break
            except Exception:
                pass
        
        if found:
            # Found a valid decomposition at this rank; try to go lower
            continue
        else:
            # If we couldn't find at this rank, we might still try lower but
            # it's unlikely. Let's continue trying lower ranks.
            pass
    
    # If we found a better rank, return it; otherwise return schoolbook
    if best_rank < rank_sb:
        # Convert to lists with integer/half-integer entries if possible
        # The entries should be exact; let's round to nearest half-integer
        U_list = [[round(x*2)/2 for x in row] for row in best_U]
        V_list = [[round(x*2)/2 for x in row] for row in best_V]
        W_list = [[round(x*2)/2 for x in row] for row in best_W]
        
        # Verify the rounded version is still exact
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
