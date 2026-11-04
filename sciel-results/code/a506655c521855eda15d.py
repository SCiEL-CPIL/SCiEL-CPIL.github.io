import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 6]

def _verify_half_integer(arr, tol=1e-9):
    doubled = arr * 2
    rounded = np.round(doubled)
    if np.max(np.abs(doubled - rounded)) > tol:
        return False
    return True

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _snap_to_half(arr):
    return np.round(arr * 2) / 2.0

def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _optimize_rank(r, n, m, p, time_limit=10):
    start_time = time.time()
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute index mappings for efficiency
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i*m+j, j*p+k, k*n+i))
    indices = np.array(indices)
    
    def objective(x):
        U = x[:nm*r].reshape(nm, r)
        V = x[nm*r : nm*r + mp*r].reshape(mp, r)
        W = x[nm*r + mp*r :].reshape(pn, r)
        
        # Vectorized computation of all tensor entries
        vals = np.zeros(len(indices))
        for idx in range(len(indices)):
            u_idx, v_idx, w_idx = indices[idx]
            vals[idx] = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
        
        err = np.sum((vals - 1.0)**2)
        return err

    def gradient(x):
        U = x[:nm*r].reshape(nm, r)
        V = x[nm*r : nm*r + mp*r].reshape(mp, r)
        W = x[nm*r + mp*r :].reshape(pn, r)
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for idx in range(len(indices)):
            u_idx, v_idx, w_idx = indices[idx]
            val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
            diff = 2 * (val - 1.0)
            
            grad_U[u_idx, :] += diff * (V[v_idx, :] * W[w_idx, :])
            grad_V[v_idx, :] += diff * (U[u_idx, :] * W[w_idx, :])
            grad_W[w_idx, :] += diff * (U[u_idx, :] * V[v_idx, :])
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    best_loss = float('inf')
    best_x = None
    
    # Multiple initialization strategies
    n_inits = 12
    
    for init_idx in range(n_inits):
        if time.time() - start_time > time_limit:
            break
        
        np.random.seed(init_idx * 7919 + 13)
        
        if init_idx == 0:
            # Small random
            x0 = np.random.randn(nm*r + mp*r + pn*r) * 0.1
        elif init_idx == 1:
            # Sparse random ±1
            x0 = np.zeros(nm*r + mp*r + pn*r)
            num_nonzero = min(r * 6, len(x0))
            indices_rand = np.random.choice(len(x0), num_nonzero, replace=False)
            x0[indices_rand] = np.random.choice([-1, 1], num_nonzero)
        elif init_idx == 2:
            # Medium random
            x0 = np.random.randn(nm*r + mp*r + pn*r) * 0.3
        elif init_idx == 3:
            # Very small random
            x0 = np.random.randn(nm*r + mp*r + pn*r) * 0.05
        elif init_idx == 4:
            # Structured: blocks of ones
            x0 = np.zeros(nm*r + mp*r + pn*r)
            for c in range(r):
                u_start = c * nm
                v_start = nm*r + c * mp
                w_start = nm*r + mp*r + c * pn
                u_idx = np.random.randint(0, nm)
                v_idx = np.random.randint(0, mp)
                w_idx = np.random.randint(0, pn)
                x0[u_start + u_idx] = 1
                x0[v_start + v_idx] = 1
                x0[w_start + w_idx] = 1
        elif init_idx == 5:
            # Structured: diagonal-like
            x0 = np.zeros(nm*r + mp*r + pn*r)
            for c in range(r):
                u_idx = c % nm
                v_idx = c % mp
                w_idx = c % pn
                x0[c*nm + u_idx] = 1
                x0[nm*r + c*mp + v_idx] = 1
                x0[nm*r + mp*r + c*pn + w_idx] = 1
        elif init_idx == 6:
            # Mixed scale
            x0 = np.random.randn(nm*r + mp*r + pn*r) * 0.2
            x0[::3] *= 2
        elif init_idx == 7:
            # Random with some zeros
            x0 = np.random.randn(nm*r + mp*r + pn*r) * 0.4
            mask = np.random.random(len(x0)) < 0.3
            x0[mask] = 0
        elif init_idx == 8:
            # Larger scale
            x0 = np.random.randn(nm*r + mp*r + pn*r) * 0.5
        elif init_idx == 9:
            # Bimodal
            x0 = np.random.randn(nm*r + mp*r + pn*r)
            x0[x0 > 0] = 0.5
            x0[x0 < 0] = -0.5
        elif init_idx == 10:
            # Random with symmetry
            x0 = np.random.randn(nm*r + mp*r + pn*r) * 0.15
        else:
            # Very sparse
            x0 = np.zeros(nm*r + mp*r + pn*r)
            num_nonzero = min(r * 3, len(x0))
            indices_rand = np.random.choice(len(x0), num_nonzero, replace=False)
            x0[indices_rand] = np.random.choice([-1, 1], num_nonzero)

        try:
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                           options={'maxiter': 600, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50})
            
            if res.fun < best_loss:
                best_loss = res.fun
                best_x = res.x.copy()
                
                U = best_x[:nm*r].reshape(nm, r)
                V = best_x[nm*r : nm*r + mp*r].reshape(mp, r)
                W = best_x[nm*r + mp*r :].reshape(pn, r)
                
                U_snap = _snap_to_half(U)
                V_snap = _snap_to_half(V)
                W_snap = _snap_to_half(W)
                
                if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-6):
                    return U_snap, V_snap, W_snap, True
                    
        except Exception:
            pass
            
    # Final polish with the best solution found
    if best_x is not None and best_loss < 1e-4:
        U = best_x[:nm*r].reshape(nm, r)
        V = best_x[nm*r : nm*r + mp*r].reshape(mp, r)
        W = best_x[nm*r + mp*r :].reshape(pn, r)
        
        # Try to snap to half-integers and verify
        U_snap = _snap_to_half(U)
        V_snap = _snap_to_half(V)
        W_snap = _snap_to_half(W)
        
        if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-6):
            return U_snap, V_snap, W_snap, True
        
        # If not half-integer, try to find nearby half-integer solution
        # by optimizing with a penalty for deviation from half-integers
        def penalized_objective(x):
            U = x[:nm*r].reshape(nm, r)
            V = x[nm*r : nm*r + mp*r].reshape(mp, r)
            W = x[nm*r + mp*r :].reshape(pn, r)
            
            vals = np.zeros(len(indices))
            for idx in range(len(indices)):
                u_idx, v_idx, w_idx = indices[idx]
                vals[idx] = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
            
            tensor_err = np.sum((vals - 1.0)**2)
            
            # Penalty for not being half-integers
            half_penalty = np.sum((U * 2 - np.round(U * 2))**2) + \
                          np.sum((V * 2 - np.round(V * 2))**2) + \
                          np.sum((W * 2 - np.round(W * 2))**2)
            
            return tensor_err + 0.1 * half_penalty
        
        def penalized_gradient(x):
            U = x[:nm*r].reshape(nm, r)
            V = x[nm*r : nm*r + mp*r].reshape(mp, r)
            W = x[nm*r + mp*r :].reshape(pn, r)
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for idx in range(len(indices)):
                u_idx, v_idx, w_idx = indices[idx]
                val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                diff = 2 * (val - 1.0)
                
                grad_U[u_idx, :] += diff * (V[v_idx, :] * W[w_idx, :])
                grad_V[v_idx, :] += diff * (U[u_idx, :] * W[w_idx, :])
                grad_W[w_idx, :] += diff * (U[u_idx, :] * V[v_idx, :])
            
            # Add half-integer penalty gradients
            grad_U += 0.4 * (U * 2 - np.round(U * 2))
            grad_V += 0.4 * (V * 2 - np.round(V * 2))
            grad_W += 0.4 * (W * 2 - np.round(W * 2))
            
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        try:
            res2 = minimize(penalized_objective, best_x, jac=penalized_gradient, method='L-BFGS-B',
                           options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-12})
            
            if res2.fun < 1e-3:
                U2 = res2.x[:nm*r].reshape(nm, r)
                V2 = res2.x[nm*r : nm*r + mp*r].reshape(mp, r)
                W2 = res2.x[nm*r + mp*r :].reshape(pn, r)
                
                U2_snap = _snap_to_half(U2)
                V2_snap = _snap_to_half(V2)
                W2_snap = _snap_to_half(W2)
                
                if _verify_decomposition(U2_snap, V2_snap, W2_snap, n, m, p, tol=1e-6):
                    return U2_snap, V2_snap, W2_snap, True
                    
        except Exception:
            pass
            
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    U_school, V_school, W_school = _make_schoolbook(n, m, p)
    
    # Lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # But we need half-integer entries, which constrains things
    # Known: R(3,4,6) is likely around 28-36
    lower_bound = 24
    
    # Try ranks strategically: focus on promising range
    ranks_to_try = list(range(lower_bound, lower_bound + 15))
    # Add some higher ranks as fallback
    ranks_to_try += [lower_bound + 15, lower_bound + 20, lower_bound + 25]
    ranks_to_try = sorted(set(ranks_to_try))
    
    best_result = None
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        # Allocate time based on rank: lower ranks get more time
        if r <= lower_bound + 5:
            time_alloc = min(15, deadline - time.time() - 8)
        elif r <= lower_bound + 10:
            time_alloc = min(10, deadline - time.time() - 8)
        else:
            time_alloc = min(6, deadline - time.time() - 8)
            
        if time_alloc < 2:
            break
            
        U, V, W, success = _optimize_rank(r, n, m, p, time_limit=time_alloc)
        
        if success:
            best_result = (U, V, W)
            break  # Found a valid decomposition, return it
            
    if best_result is not None:
        U, V, W = best_result
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
            
    # Fallback to schoolbook
    U_list = [[float(x) for x in row] for row in U_school]
    V_list = [[float(x) for x in row] for row in V_school]
    W_list = [[float(x) for x in row] for row in W_school]
    return {"U": U_list, "V": V_list, "W": W_list}
