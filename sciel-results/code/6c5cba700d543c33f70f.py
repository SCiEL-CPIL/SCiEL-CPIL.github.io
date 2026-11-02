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

def _soft_max_abs(res, beta=100.0):
    """Smooth approximation of max(|res|) using log-sum-exp."""
    a = np.abs(res)
    amax = np.max(a)
    if amax < 1e-15:
        return 0.0
    scaled = beta * (a - amax)
    # log-sum-exp trick for stability
    m = np.max(scaled)
    lse = m + np.log(np.sum(np.exp(scaled - m)))
    return amax + lse / beta

def _soft_max_abs_grad(res, beta=100.0):
    """Gradient of soft max abs."""
    a = np.abs(res)
    amax = np.max(a)
    if amax < 1e-15:
        return np.zeros_like(res)
    scaled = beta * (a - amax)
    m = np.max(scaled)
    exps = np.exp(scaled - m)
    weights = exps / np.sum(exps)
    signs = np.sign(res)
    return beta * weights * signs

def _rank31_optimize(U_init, V_init, W_init, T_true, nm, mp, pn, rank, max_iter, ftol, gtol):
    """Optimize rank-r decomposition starting from initialization."""
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    def _unpack(params):
        U_f = params[:nm * rank]
        V_f = params[nm * rank : nm * rank + mp * rank]
        W_f = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
        return U_f.reshape(nm, rank), V_f.reshape(mp, rank), W_f.reshape(pn, rank)
    
    def _repack(U_m, V_m, W_m):
        return np.concatenate([U_m.flatten(), V_m.flatten(), W_m.flatten()])
    
    # Phase 1: minimize sum of squares
    def objective_sse(params):
        U_m, V_m, W_m = _unpack(params)
        M = np.einsum('ir,jr,kr->ijk', U_m, V_m, W_m)
        res = M - T_true
        return np.sum(res * res)
    
    def gradient_sse(params):
        U_m, V_m, W_m = _unpack(params)
        M = np.einsum('ir,jr,kr->ijk', U_m, V_m, W_m)
        R = M - T_true
        gU = np.einsum('ijk,jr,kr->ir', R, V_m, W_m)
        gV = np.einsum('ijk,ir,kr->jr', R, U_m, W_m)
        gW = np.einsum('ijk,ir,jr->kr', R, U_m, V_m)
        return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
    
    try:
        result = minimize(objective_sse, x0, jac=gradient_sse, method='L-BFGS-B',
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
    
    # Phase 2: minimize smooth max-abs residual from best SSE result
    x1 = result.x if 'result' in dir() else x0
    U_m, V_m, W_m = _unpack(x1)
    M = np.einsum('ir,jr,kr->ijk', U_m, V_m, W_m)
    cur_max = np.max(np.abs(M - T_true))
    
    if cur_max > 1e-6:
        beta = 50.0
        def objective_smax(params):
            U_m2, V_m2, W_m2 = _unpack(params)
            M2 = np.einsum('ir,jr,kr->ijk', U_m2, V_m2, W_m2)
            res2 = M2 - T_true
            return _soft_max_abs(res2, beta)
        
        def gradient_smax(params):
            U_m2, V_m2, W_m2 = _unpack(params)
            M2 = np.einsum('ir,jr,kr->ijk', U_m2, V_m2, W_m2)
            res2 = M2 - T_true
            grad_smax = _soft_max_abs_grad(res2, beta)
            gU = np.einsum('ijk,jr,kr->ir', grad_smax, V_m2, W_m2)
            gV = np.einsum('ijk,ir,kr->jr', grad_smax, U_m2, W_m2)
            gW = np.einsum('ijk,ir,jr->kr', grad_smax, U_m2, V_m2)
            return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
        
        try:
            result2 = minimize(objective_smax, x1, jac=gradient_smax, method='L-BFGS-B',
                               options={'maxiter': max_iter // 2, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30})
            U_f = result2.x[:nm * rank]
            V_f = result2.x[nm * rank : nm * rank + mp * rank]
            W_f = result2.x[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
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
    """ALS polishing for exact solution with high-precision final pass."""
    for it in range(max_iters):
        # Update U
        for i in range(nm):
            B = np.zeros((mp * pn, rank))
            for r in range(rank):
                B[:, r] = (V[:, r] * W[:, r]).flatten()
            t = T_true[i, :, :].flatten()
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
    
    # High-precision final polish: iterate more tightly
    for it2 in range(100):
        for i in range(nm):
            B = np.zeros((mp * pn, rank))
            for r in range(rank):
                B[:, r] = (V[:, r] * W[:, r]).flatten()
            t = T_true[i, :, :].flatten()
            try:
                u, _, _, _ = np.linalg.lstsq(B, t, rcond=None)
            except:
                continue
            U[i, :] = u
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
        M = np.einsum('ir,jr,kr->ijk', U, V, W)
        loss = np.sum((M - T_true)**2)
        if loss < 1e-28:
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
        
        for attempt in range(4):
            if time.time() - start > time_budget * 0.7:
                break
            noise_scale = 0.01 * (attempt + 1)
            U_init = U0 + rng.normal(0, noise_scale, size=U0.shape)
            V_init = V0 + rng.normal(0, noise_scale, size=V0.shape)
            W_init = W0 + rng.normal(0, noise_scale, size=W0.shape)
            
            result = _rank31_optimize(U_init, V_init, W_init, T_true, nm, mp, pn, rank, 
                                      max_iter=200, ftol=1e-15, gtol=1e-13)
            if result is not None:
                U_sol, V_sol, max_res = result
                if max_res < best_loss:
                    best_loss = max_res
                    best = (U_sol.copy(), V_sol.copy(), W_sol.copy())
                    # Polish
                    U_p, V_p, W_p, mr = _als_polish(U_sol, V_sol, W_sol, T_true, nm, mp, pn, rank)
                    if mr < 1e-10:
                        return U_p, V_p, W_p
    
    # Strategy 2: Random initialization with varied scales
    for attempt in range(40):
        if time.time() - start > time_budget:
            break
        scale = 0.3 + 0.4 * (attempt % 5) / 5.0
        U_init = rng.normal(0, scale, size=(nm, rank))
        V_init = rng.normal(0, scale, size=(mp, rank))
        W_init = rng.normal(0, scale, size=(pn, rank))
        
        result = _rank31_optimize(U_init, V_init, W_init, T_true, nm, mp, pn, rank,
                                  max_iter=150, ftol=1e-15, gtol=1e-13)
        if result is not None:
            U_sol, V_sol, max_res = result
            if max_res < best_loss:
                best_loss = max_res
                best = (U_sol.copy(), V_sol.copy(), W_sol.copy())
                U_p, V_p, W_p, mr = _als_polish(U_sol, V_sol, W_sol, T_true, nm, mp, pn, rank)
                if mr < 1e-10:
                    return U_p, V_p, W_p
    
    # Strategy 3: If we have a near-miss, try harder polish on best
    if best is not None and best_loss < 1e-4:
        U_b, V_b, W_b = best
        U_p, V_p, W_p, mr = _als_polish(U_b, V_b, W_b, T_true, nm, mp, pn, rank, max_iters=1000)
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
