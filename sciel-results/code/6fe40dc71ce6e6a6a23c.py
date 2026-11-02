import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 9]
RNG_SEED = 42

def _build_rank32():
    n, m, p = DIMENSIONS
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[2, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
        col += 1
        U[0, col] = 1
        V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[3, col] = 1
        V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
        col += 1
        U[0, col] = 1; U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
        col += 1
        U[2, col] = 1; U[0, col] = -1
        V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        U[1, col] = 1; U[3, col] = -1
        V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0, col] = 1; V[0*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[1, col] = 1; V[1*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[2, col] = 1; V[0*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1
        U[3, col] = 1; V[1*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)
    
    return U, V, W

def _build_true_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T[i, j, k] = 1.0
    return T

def _rank31_optimize(U_init, V_init, W_init, T_true, nm, mp, pn, rank, max_iter, ftol, gtol):
    """Optimize rank-r decomposition starting from initialization."""
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    def objective(params):
        U_f = params[:nm * rank]
        V_f = params[nm * rank : nm * rank + mp * rank]
        W_f = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
        U_m = U_f.reshape(nm, rank)
        V_m = V_f.reshape(mp, rank)
        W_m = W_f.reshape(pn, rank)
        M = np.einsum('ir,jr,kr->ijk', U_m, V_m, W_m)
        res = M - T_true
        return np.sum(res * res)
    
    def gradient(params):
        U_f = params[:nm * rank]
        V_f = params[nm * rank : nm * rank + mp * rank]
        W_f = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
        U_m = U_f.reshape(nm, rank)
        V_m = V_f.reshape(mp, rank)
        W_m = W_f.reshape(pn, rank)
        M = np.einsum('ir,jr,kr->ijk', U_m, V_m, W_m)
        R = M - T_true
        # dR/dU[i,r] = V[:,r] outer W[:,r] contracted with R
        gU = np.einsum('ijk,jr,kr->ir', R, V_m, W_m)
        gV = np.einsum('ijk,ir,kr->jr', R, U_m, W_m)
        gW = np.einsum('ijk,ir,jr->kr', R, U_m, V_m)
        return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
    
    try:
        result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50})
        if result.fun < 1e-10:
            U_f = result.x[:nm * rank]
            V_f = result.x[nm * rank : nm * rank + mp * rank]
            W_f = result.x[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
            U_sol = U_f.reshape(nm, rank)
            V_sol = V_f.reshape(mp, rank)
            W_sol = W_f.reshape(pn, rank)
            M = np.einsum('ir,jr,kr->ijk', U_sol, V_sol, W_sol)
            max_res = np.max(np.abs(M - T_true))
            if max_res < 1e-8:
                return U_sol, V_sol, max_res
    except Exception:
        pass
    return None

def _als_polish(U, V, W, T_true, nm, mp, pn, rank, max_iters=500, tol=1e-14):
    """ALS polishing for exact solution."""
    for it in range(max_iters):
        # Update U
        for i in range(nm):
            # T[i,:,:] = sum_r U[i,r] * V[:,r] * W[:,r]
            # For fixed V,W: linear in U[i,:]
            # T[i,j,k] = sum_r U[i,r] * (V[j,r]*W[k,r])
            # Let's build: for each (j,k) where T[i,j,k] != 0, we have a constraint
            # Actually, use the full system:
            # vec(T[i,:,:]) = sum_r U[i,r] * vec(V[:,r] * W[:,r])
            # This is: B * u = t where B has columns vec(V[:,r]*W[:,r])
            # But mp*pn = 324, rank=31, so B is 324x31
            B = np.zeros((mp * pn, rank))
            for r in range(rank):
                B[:, r] = (V[:, r] * W[:, r]).flatten()
            t = T_true[i, :, :].flatten()
            # Solve least squares
            try:
                u, _, _, _ = np.linalg.lstsq(B, t, rcond=None)
            except:
                continue
            U[i, :] = u
        
        # Update V
        for j in range(mp):
            B = np.zeros((nm * pn, rank))
            for r in range(rank):
                B[:, r] = (U[:, r] * W[:, r]).flatten()
            t = T_true[:, j, :].flatten()
            try:
                v, _, _, _ = np.linalg.lstsq(B, t, rcond=None)
            except:
                continue
            V[j, :] = v
        
        # Update W
        for k in range(pn):
            B = np.zeros((nm * mp, rank))
            for r in range(rank):
                B[:, r] = (U[:, r] * V[:, r]).flatten()
            t = T_true[:, :, k].flatten()
            try:
                w, _, _, _ = np.linalg.lstsq(B, t, rcond=None)
            except:
                continue
            W[k, :] = w
        
        # Check convergence
        M = np.einsum('ir,jr,kr->ijk', U, V, W)
        loss = np.sum((M - T_true)**2)
        if loss < tol:
            break
    
    M = np.einsum('ir,jr,kr->ijk', U, V, W)
    max_res = np.max(np.abs(M - T_true))
    return U, V, W, max_res

def _try_rank31(U32, V32, W32, T_true, nm, mp, pn, time_budget):
    """Try to find rank-31 decomposition."""
    start = time.time()
    rank = 31
    best = None
    best_loss = float('inf')
    
    rng = np.random.default_rng(RNG_SEED)
    
    # Strategy 1: Drop each column from rank-32, add noise, optimize
    for drop in range(32):
        if time.time() - start > time_budget * 0.7:
            break
        mask = [i for i in range(32) if i != drop]
        U0 = U32[:, mask].copy()
        V0 = V32[:, mask].copy()
        W0 = W32[:, mask].copy()
        
        for attempt in range(3):
            if time.time() - start > time_budget * 0.7:
                break
            noise_scale = 0.01 * (attempt + 1)
            U_init = U0 + rng.normal(0, noise_scale, size=U0.shape)
            V_init = V0 + rng.normal(0, noise_scale, size=V0.shape)
            W_init = W0 + rng.normal(0, noise_scale, size=W0.shape)
            
            result = _rank31_optimize(U_init, V_init, W_init, T_true, nm, mp, pn, rank, 
                                      max_iter=300, ftol=1e-15, gtol=1e-13)
            if result is not None:
                U_sol, V_sol, max_res = result
                if max_res < best_loss:
                    best_loss = max_res
                    best = (U_sol.copy(), V_sol.copy(), V_sol.copy())
                    # Polish
                    U_p, V_p, W_p, mr = _als_polish(U_sol, V_sol, W_sol, T_true, nm, mp, pn, rank)
                    if mr < 1e-10:
                        return U_p, V_p, W_p
    
    # Strategy 2: Random initialization
    for attempt in range(30):
        if time.time() - start > time_budget:
            break
        U_init = rng.normal(0, 0.5, size=(nm, rank))
        V_init = rng.normal(0, 0.5, size=(mp, rank))
        W_init = rng.normal(0, 0.5, size=(pn, rank))
        
        result = _rank31_optimize(U_init, V_init, W_init, T_true, nm, mp, pn, rank,
                                  max_iter=200, ftol=1e-15, gtol=1e-13)
        if result is not None:
            U_sol, V_sol, max_res = result
            if max_res < best_loss:
                best_loss = max_res
                best = (U_sol.copy(), V_sol.copy(), W_sol.copy())
                U_p, V_p, W_p, mr = _als_polish(U_sol, V_sol, W_sol, T_true, nm, mp, pn, rank)
                if mr < 1e-10:
                    return U_p, V_p, W_p
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    U32, V32, W32 = _build_rank32()
    T_true = _build_true_tensor()
    
    def to_list(arr):
        return arr.tolist()
    
    # Try rank 31
    time_remaining = deadline - time.time() - 5.0
    if time_remaining > 10.0:
        result = _try_rank31(U32, V32, W32, T_true, nm, mp, pn, time_remaining)
        if result is not None:
            U_sol, V_sol, W_sol = result
            M = np.einsum('ir,jr,kr->ijk', U_sol, V_sol, W_sol)
            residual = np.max(np.abs(M - T_true))
            if residual < 1e-6:
                return {"U": to_list(U_sol), "V": to_list(V_sol), "W": to_list(W_sol)}
    
    # Fall back to rank 32
    return {"U": to_list(U32), "V": to_list(V32), "W": to_list(W32)}
