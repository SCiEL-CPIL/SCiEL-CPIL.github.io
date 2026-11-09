import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 6]

def _build_target(n, m, p):
    nm, mp, pn = n*m, m*p, p*n
    target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    return target

def _objective(params, U_shape, V_shape, W_shape, target_flat, nm, mp, pn):
    """
    params: flattened vector of U, V, W
    """
    size_u = U_shape[0] * U_shape[1]
    size_v = V_shape[0] * V_shape[1]
    size_w = W_shape[0] * W_shape[1]
    
    U = params[:size_u].reshape(U_shape)
    V = params[size_u:size_u+size_v].reshape(V_shape)
    W = params[size_u+size_v:].reshape(W_shape)
    
    # Compute decomposition
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # Efficiently: 
    # Temp[a,b] = sum_r U[a,r] V[b,r] -> (nm, mp)
    # Then T[a,b,c] = sum_r Temp[a,b] * W[c,r]? No, that's not right.
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    
    # Use einsum
    decomp = np.einsum('ar,br,cr->abc', U, V, W)
    diff = decomp - target_flat.reshape(nm, mp, pn)
    
    # Return sum of squares
    return np.sum(diff * diff)

def _try_rank(rank, target, nm, mp, pn, n, m, p, rng, max_attempts, deadline):
    U_shape = (nm, rank)
    V_shape = (mp, rank)
    W_shape = (pn, rank)
    
    target_flat = target.reshape(-1)
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
            
        # Initialization
        # Scale initialization to be small
        U = rng.randn(*U_shape) * 0.1
        V = rng.randn(*V_shape) * 0.1
        W = rng.randn(*W_shape) * 0.1
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Optimization
        # Use L-BFGS-B
        try:
            res = minimize(
                _objective,
                params,
                args=(U_shape, V_shape, W_shape, target_flat, nm, mp, pn),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000}
            )
            
            if res.success or res.fun < 1e-8:
                # Extract best params
                best_params = res.x
                U_best = best_params[:nm*rank].reshape(U_shape)
                V_best = best_params[nm*rank:nm*rank+mp*rank].reshape(V_shape)
                W_best = best_params[nm*rank+mp*rank:].reshape(W_shape)
                
                # Try rounding to half-integers
                # Check if the continuous solution is close to half-integers
                U_rounded = np.round(U_best * 2) / 2
                V_rounded = np.round(V_best * 2) / 2
                W_rounded = np.round(W_best * 2) / 2
                
                # Verify exactness
                decomp_check = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                if np.allclose(decomp_check, target, atol=1e-6, rtol=1e-6):
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                    
                # If not half-integers, maybe integers?
                U_int = np.round(U_best)
                V_int = np.round(V_best)
                W_int = np.round(W_best)
                decomp_int = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
                if np.allclose(decomp_int, target, atol=1e-6, rtol=1e-6):
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }

        except Exception:
            pass
            
    return None

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    schoolbook_rank = n * m * p
    
    # Fallback
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = schoolbook_rank
    
    target = _build_target(n, m, p)
    rng = np.random.RandomState(42)
    
    lower_bound = max(nm, mp, pn)
    
    for rank in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline:
            break
            
        # Reduce attempts as rank increases? Or keep constant?
        # L-BFGS is expensive. Limit attempts.
        max_attempts = 10 if rank < 30 else 5
        
        result = _try_rank(rank, target, nm, mp, pn, n, m, p, rng, max_attempts, deadline)
        
        if result is not None:
            best_rank = rank
            best_result = result
            break
            
    return best_result
