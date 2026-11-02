import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 9]
RNG_SEED = 42

def _build_rank32():
    """Build rank-32 solution using Strassen for pairs of columns."""
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

def _build_rank30():
    """
    Build rank-30 solution by exploiting the 3-block structure.
    
    The 2x2x9 tensor can be viewed as 3 copies of 2x2x3.
    Each 2x2x3 has rank 11 (known optimal).
    But we can potentially share structure across blocks.
    
    Strategy: Use a structured decomposition where we treat the 9 columns
    as 3 groups of 3, and use a global optimization to find rank-30.
    
    Actually, let's try a different approach: use the fact that M_{2,2,9}
    can be decomposed more efficiently than 3 * M_{2,2,3}.
    
    Known: M_{2,2,3} has rank 11. So 3*M_{2,2,3} = 33.
    But maybe we can do better by sharing.
    
    Let's try a direct construction with rank 30 using a pattern.
    """
    n, m, p = DIMENSIONS
    rank = 30
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # We'll use a pattern based on grouping the 9 columns into 3 groups of 3
    # and using a shared Strassen-like structure
    
    # Group 1: columns 0,1,2
    # Group 2: columns 3,4,5  
    # Group 3: columns 6,7,8
    
    col = 0
    
    # For each group of 3 columns, we need 11 terms
    # But we'll try to use a total of 30 terms with some sharing
    
    # Actually, let's just build a valid rank-33 and try to optimize down to 30
    # For now, return the rank-33 construction as a starting point
    # The optimization will try to reduce it
    
    # Let's build a rank-30 attempt directly
    # Pattern: 10 terms per group of 3 columns
    
    for group in range(3):
        base_col = group * 3
        # Use a compact 10-term decomposition for each 2x2x3 block
        # This is based on known near-optimal decompositions
        
        # Term 1
        U[0, col] = 1; U[3, col] = 1
        V[0*p + base_col, col] = 1; V[1*p + base_col + 1, col] = 1
        W[base_col*n + 0, col] = 1; W[(base_col+1)*n + 1, col] = 1
        col += 1
        
        # Term 2
        U[2, col] = 1; U[3, col] = 1
        V[0*p + base_col, col] = 1
        W[base_col*n + 1, col] = 1; W[(base_col+1)*n + 1, col] = -1
        col += 1
        
        # Term 3
        U[0, col] = 1
        V[0*p + base_col + 1, col] = 1; V[1*p + base_col + 1, col] = -1
        W[(base_col+1)*n + 0, col] = 1; W[(base_col+1)*n + 1, col] = 1
        col += 1
        
        # Term 4
        U[3, col] = 1
        V[1*p + base_col, col] = 1; V[0*p + base_col, col] = -1
        W[base_col*n + 0, col] = 1; W[base_col*n + 1, col] = 1
        col += 1
        
        # Term 5
        U[0, col] = 1; U[1, col] = 1
        V[1*p + base_col + 1, col] = 1
        W[base_col*n + 0, col] = -1; W[(base_col+1)*n + 0, col] = 1
        col += 1
        
        # Term 6
        U[2, col] = 1; U[0, col] = -1
        V[0*p + base_col, col] = 1; V[0*p + base_col + 1, col] = 1
        W[(base_col+1)*n + 1, col] = 1
        col += 1
        
        # Term 7
        U[1, col] = 1; U[3, col] = -1
        V[1*p + base_col, col] = 1; V[1*p + base_col + 1, col] = 1
        W[base_col*n + 0, col] = 1
        col += 1
        
        # Term 8: handle column base_col+2
        U[0, col] = 1
        V[0*p + base_col + 2, col] = 1
        W[(base_col+2)*n + 0, col] = 1
        col += 1
        
        # Term 9
        U[1, col] = 1
        V[1*p + base_col + 2, col] = 1
        W[(base_col+2)*n + 0, col] = 1
        col += 1
        
        # Term 10
        U[2, col] = 1
        V[0*p + base_col + 2, col] = 1
        W[(base_col+2)*n + 1, col] = 1
        col += 1
    
    # We have 30 terms, but they don't cover all outputs yet
    # We need additional terms for the cross terms
    # This construction is incomplete - we need to verify and fix
    
    return U, V, W

def _compute_residual(U, V, W, T_true):
    M = np.einsum('ir,jr,kr->ijk', U, V, W)
    residual = M - T_true
    return np.max(np.abs(residual))

def _objective_sum_sq(params, T_true, nm, mp, pn, rank):
    U_flat = params[:nm * rank]
    V_flat = params[nm * rank : nm * rank + mp * rank]
    W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
    
    U = U_flat.reshape(nm, rank)
    V = V_flat.reshape(mp, rank)
    W = W_flat.reshape(pn, rank)
    
    M = np.einsum('ir,jr,kr->ijk', U, V, W)
    residual = M - T_true
    return np.sum(residual**2)

def _objective_max_abs(params, T_true, nm, mp, pn, rank, scale=50.0):
    U_flat = params[:nm * rank]
    V_flat = params[nm * rank : nm * rank + mp * rank]
    W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
    
    U = U_flat.reshape(nm, rank)
    V = V_flat.reshape(mp, rank)
    W = W_flat.reshape(pn, rank)
    
    M = np.einsum('ir,jr,kr->ijk', U, V, W)
    residual = M - T_true
    r = np.abs(residual) * scale
    max_r = np.max(r)
    if max_r < 1e-30:
        return 0.0
    return np.log(np.sum(np.exp(r - max_r))) / scale + max_r / scale

def _try_reduce_rank(U_init, V_init, W_init, T_true, target_rank, time_budget, rng):
    """Try to find a decomposition of target_rank from initial higher-rank solution."""
    start_time = time.time()
    nm, mp, pn = U_init.shape[0], V_init.shape[0], W_init.shape[0]
    init_rank = U_init.shape[1]
    
    best_score = float('inf')
    best_result = None
    
    n_restarts = 30
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_budget * 0.9:
            break
        
        # Strategy: select target_rank columns from init_rank
        if init_rank > target_rank:
            cols_to_keep = rng.choice(init_rank, size=target_rank, replace=False)
            cols_to_keep = np.sort(cols_to_keep)
        else:
            cols_to_keep = np.arange(target_rank)
        
        U_t = U_init[:, cols_to_keep].copy()
        V_t = V_init[:, cols_to_keep].copy()
        W_t = W_init[:, cols_to_keep].copy()
        
        # Add perturbation
        noise_scale = 0.01 + 0.02 * (restart % 5)
        U_t = U_t + rng.normal(0, noise_scale, size=U_t.shape)
        V_t = V_t + rng.normal(0, noise_scale, size=V_t.shape)
        W_t = W_t + rng.normal(0, noise_scale, size=W_t.shape)
        
        x0 = np.concatenate([U_t.flatten(), V_t.flatten(), W_t.flatten()])
        
        def objective(params):
            return _objective_sum_sq(params, T_true, nm, mp, pn, target_rank)
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if result.fun < 1e-8:
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
                    
                    # Early exit if we found a very good solution
                    if score < 1e-10:
                        break
        except Exception:
            continue
    
    return best_result

def _polish_solution(U, V, W, T_true, time_budget):
    """High-precision polish pass."""
    start_time = time.time()
    nm, mp, pn = U.shape[0], V.shape[0], W.shape[0]
    rank = U.shape[1]
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def objective(params):
        return _objective_sum_sq(params, T_true, nm, mp, pn, rank)
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-12})
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
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
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
    
    # Verify rank-32
    residual32 = _compute_residual(U32, V32, W32, T_true)
    if residual32 > 1e-10:
        # This shouldn't happen, but just in case
        U36 = np.zeros((n * m, n * m * p))
        V36 = np.zeros((m * p, n * m * p))
        W36 = np.zeros((p * n, n * m * p))
        col = 0
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    i = a * m + b
                    j = b * p + c
                    k = c * n + a
                    U36[i, col] = 1
                    V36[j, col] = 1
                    W36[k, col] = 1
                    col += 1
        return {"U": to_list(U36), "V": to_list(V36), "W": to_list(W36)}
    
    # Build rank-33 solution
    U33, V33, W33 = _build_rank33()
    
    # Verify rank-33
    residual33 = _compute_residual(U33, V33, W33, T_true)
    
    rng = np.random.default_rng(RNG_SEED)
    
    # Try to find rank-31 from rank-33
    time_remaining = deadline - time.time()
    if time_remaining > 40.0 and residual33 < 1e-10:
        result31 = _try_reduce_rank(U33, V33, W33, T_true, 31, time_remaining - 20.0, rng)
        if result31 is not None:
            U_sol, V_sol, W_sol, score = result31
            # Polish
            time_remaining = deadline - time.time()
            if time_remaining > 10.0:
                U_sol, V_sol, W_sol = _polish_solution(U_sol, V_sol, W_sol, T_true, time_remaining - 5.0)
            # Final verification
            M = np.einsum('ir,jr,kr->ijk', U_sol, V_sol, W_sol)
            residual = np.max(np.abs(M - T_true))
            if residual < 1e-8:
                return {"U": to_list(U_sol), "V": to_list(V_sol), "W": to_list(W_sol)}
    
    # Try to find rank-30 from rank-33
    time_remaining = deadline - time.time()
    if time_remaining > 30.0 and residual33 < 1e-10:
        result30 = _try_reduce_rank(U33, V33, W33, T_true, 30, time_remaining - 15.0, rng)
        if result30 is not None:
            U_sol, V_sol, W_sol, score = result30
            # Polish
            time_remaining = deadline - time.time()
            if time_remaining > 10.0:
                U_sol, V_sol, W_sol = _polish_solution(U_sol, V_sol, W_sol, T_true, time_remaining - 5.0)
            # Final verification
            M = np.einsum('ir,jr,kr->ijk', U_sol, V_sol, W_sol)
            residual = np.max(np.abs(M - T_true))
            if residual < 1e-8:
                return {"U": to_list(U_sol), "V": to_list(V_sol), "W": to_list(W_sol)}
    
    # Fall back to rank-32
    return {"U": to_list(U32), "V": to_list(V32), "W": to_list(W32)}
