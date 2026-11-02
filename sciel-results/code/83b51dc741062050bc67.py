import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 9]
RNG_SEED = 42
TIME_LIMIT = 145.0  # Safety margin before 150s evaluator kill

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

def _build_rank33():
    """Build rank-33 solution: 3 blocks of rank-11 for M_{2,2,3}."""
    n, m, p = DIMENSIONS
    rank = 33
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen_block(c0, c1):
        nonlocal col
        U[0, col] = 1; U[3, col] = 1
        V[0*p + c0, col] = 1; V[1*p + c1, col] = 1
        W[c0*n + 0, col] = 1; W[c1*n + 1, col] = 1
        col += 1
        U[2, col] = 1; U[3, col] = 1
        V[0*p + c0, col] = 1
        W[c0*n + 1, col] = 1; W[c1*n + 1, col] = -1
        col += 1
        U[0, col] = 1
        V[0*p + c1, col] = 1; V[1*p + c1, col] = -1
        W[c1*n + 0, col] = 1; W[c1*n + 1, col] = 1
        col += 1
        U[3, col] = 1
        V[1*p + c0, col] = 1; V[0*p + c0, col] = -1
        W[c0*n + 0, col] = 1; W[c0*n + 1, col] = 1
        col += 1
        U[0, col] = 1; U[1, col] = 1
        V[1*p + c1, col] = 1
        W[c0*n + 0, col] = -1; W[c1*n + 0, col] = 1
        col += 1
        U[2, col] = 1; U[0, col] = -1
        V[0*p + c0, col] = 1; V[0*p + c1, col] = 1
        W[c1*n + 1, col] = 1
        col += 1
        U[1, col] = 1; U[3, col] = -1
        V[1*p + c0, col] = 1; V[1*p + c1, col] = 1
        W[c0*n + 0, col] = 1
        col += 1

    def add_trivial_block(c):
        nonlocal col
        U[0, col] = 1; V[0*p + c, col] = 1; W[c*n + 0, col] = 1
        col += 1
        U[1, col] = 1; V[1*p + c, col] = 1; W[c*n + 0, col] = 1
        col += 1
        U[2, col] = 1; V[0*p + c, col] = 1; W[c*n + 1, col] = 1
        col += 1
        U[3, col] = 1; V[1*p + c, col] = 1; W[c*n + 1, col] = 1
        col += 1

    add_strassen_block(0, 1)
    add_trivial_block(2)
    add_strassen_block(3, 4)
    add_trivial_block(5)
    add_strassen_block(6, 7)
    add_trivial_block(8)
    
    return U, V, W

def _compute_residual(U, V, W, T_true):
    M = np.einsum('ir,jr,kr->ijk', U, V, W)
    residual = M - T_true
    return np.max(np.abs(residual))

def _soft_max_abs(residual, scale=100.0):
    """Smooth approximation to max(|residual|) via log-sum-exp."""
    r = np.abs(residual)
    r_scaled = r * scale
    max_r = np.max(r_scaled)
    if max_r < 1e-30:
        return 0.0
    return (np.log(np.sum(np.exp(r_scaled - max_r))) + max_r) / scale

def _compute_grad_soft_max(residual, scale=100.0):
    """Gradient of soft_max_abs w.r.t. residual."""
    r = np.abs(residual)
    r_scaled = r * scale
    max_r = np.max(r_scaled)
    if max_r < 1e-30:
        return np.zeros_like(residual)
    weights = np.exp(r_scaled - max_r)
    weights /= np.sum(weights)
    # d/d residual of |residual| = sign(residual)
    return weights * np.sign(residual) / 1.0  # The 1/scale is in the function, gradient w.r.t. residual is weights * sign

def _try_rank31_from_rank33(U33, V33, W33, T_true, time_budget):
    """Try to find a rank-31 decomposition starting from rank-33 using soft-max objective."""
    start_time = time.time()
    
    nm, mp, pn = U33.shape[0], V33.shape[0], W33.shape[0]
    target_rank = 31
    
    best_score = float('inf')
    best_result = None
    
    rng = np.random.default_rng(RNG_SEED)
    n_restarts = 40
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_budget * 0.8:
            break
        
        # Strategy: drop 2 columns and perturb
        cols_to_drop = rng.choice(33, size=2, replace=False)
        cols_to_keep = [c for c in range(33) if c not in cols_to_drop]
        
        U31 = U33[:, cols_to_keep].copy()
        V31 = V33[:, cols_to_keep].copy()
        W31 = W33[:, cols_to_keep].copy()
        
        # Add perturbation - vary scale per restart
        noise_scale = 0.01 + 0.04 * (restart % 7)
        U31 = U31 + rng.normal(0, noise_scale, size=U31.shape)
        V31 = V31 + rng.normal(0, noise_scale, size=V31.shape)
        W31 = W31 + rng.normal(0, noise_scale, size=W31.shape)
        
        x0 = np.concatenate([U31.flatten(), V31.flatten(), W31.flatten()])
        
        def objective(params):
            U_flat = params[:nm * target_rank]
            V_flat = params[nm * target_rank : nm * target_rank + mp * target_rank]
            W_flat = params[nm * target_rank + mp * target_rank : nm * target_rank + mp * target_rank + pn * target_rank]
            
            U = U_flat.reshape(nm, target_rank)
            V = V_flat.reshape(mp, target_rank)
            W = W_flat.reshape(pn, target_rank)
            
            M = np.einsum('ir,jr,kr->ijk', U, V, W)
            residual = M - T_true
            return _soft_max_abs(residual, scale=100.0)
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if result.fun < 1e-6:
                U_flat = result.x[:nm * target_rank]
                V_flat = result.x[nm * target_rank : nm * target_rank + mp * target_rank]
                W_flat = result.x[nm * target_rank + mp * target_rank : nm * target_rank + mp * target_rank + pn * target_rank]
                U_sol = U_flat.reshape(nm, target_rank)
                V_sol = V_flat.reshape(mp, target_rank)
                W_sol = W_flat.reshape(pn, target_rank)
                
                score = _compute_residual(U_sol, V_sol, W_sol, T_true)
                if score < 1e-8 and score < best_score:
                    best_score = score
                    best_result = (U_sol.copy(), V_sol.copy(), W_sol.copy(), score)
        except Exception:
            continue
    
    return best_result

def _polish_solution(U, V, W, T_true, time_budget):
    """High-precision polish pass with tighter tolerances using sum-of-squares."""
    start_time = time.time()
    nm, mp, pn = U.shape[0], V.shape[0], W.shape[0]
    rank = U.shape[1]
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def objective(params):
        U_flat = params[:nm * rank]
        V_flat = params[nm * rank : nm * rank + mp * rank]
        W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
        
        U = U_flat.reshape(nm, rank)
        V = V_flat.reshape(mp, rank)
        W = W_flat.reshape(pn, rank)
        
        M = np.einsum('ir,jr,kr->ijk', U, V, W)
        residual = M - T_true
        return np.sum(residual**2)
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-18, 'gtol': 1e-14})
        if result.fun < 1e-14:
            U_flat = result.x[:nm * rank]
            V_flat = result.x[nm * rank : nm * rank + mp * rank]
            W_flat = result.x[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
            U_sol = U_flat.reshape(nm, rank)
            V_sol = V_flat.reshape(mp, rank)
            W_sol = W_flat.reshape(pn, rank)
            return U_sol, V_sol, W_sol
    except Exception:
        pass
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    def to_list(arr):
        return arr.tolist()
    
    # Build true tensor
    T_true = np.zeros((n * m, m * p, p * n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_true[i, j, k] = 1.0
    
    # Build rank-32 solution (known good fallback)
    U32, V32, W32 = _build_rank32()
    
    # Build rank-33 solution
    U33, V33, W33 = _build_rank33()
    
    # Verify rank-33 is correct
    residual33 = _compute_residual(U33, V33, W33, T_true)
    if residual33 > 1e-10:
        return {"U": to_list(U32), "V": to_list(V32), "W": to_list(W32)}
    
    # Try to find rank-31 from rank-33
    time_remaining = deadline - time.time()
    if time_remaining > 20.0:
        result = _try_rank31_from_rank33(U33, V33, W33, T_true, time_remaining - 5.0)
        if result is not None:
            U_sol, V_sol, W_sol, score = result
            # Polish
            time_remaining = deadline - time.time()
            if time_remaining > 3.0:
                U_sol, V_sol, W_sol = _polish_solution(U_sol, V_sol, W_sol, T_true, time_remaining - 1.0)
            # Final verification
            M = np.einsum('ir,jr,kr->ijk', U_sol, V_sol, W_sol)
            residual = np.max(np.abs(M - T_true))
            if residual < 1e-8:
                return {"U": to_list(U_sol), "V": to_list(V_sol), "W": to_list(W_sol)}
    
    # Fall back to rank-32
    return {"U": to_list(U32), "V": to_list(V32), "W": to_list(W32)}
