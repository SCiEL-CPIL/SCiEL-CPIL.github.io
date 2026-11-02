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

def _build_laderman_2x2x16():
    """
    Build the Laderman decomposition for 2x2x16 matrix multiplication.
    The 16 columns are grouped as 4 blocks of 4, each block corresponding
    to a 2x2x4 sub-problem solved with rank 7 (Strassen-like) plus overhead.
    
    Actually, let's use the known rank-28 decomposition for 2x2x16.
    The structure: we have A (2x2) times B (2x16) = C (2x16).
    We can view B as 4 blocks of 2x4, and use Strassen on each 2x2x4.
    
    For 2x2x4: rank is 7 (same as 2x2x2 since the 4th column is trivial).
    Wait, 2x2x4 has rank 7? Let me think...
    
    Actually for 2x2xN, the rank is 7 for N>=2 (Strassen applies to first 2 cols,
    remaining N-2 cols are trivial with rank 4 each). So 2x2x4 has rank 7+4=11?
    No wait - Strassen handles 2x2x2 at rank 7, then 2 more columns at rank 4 each = 15?
    
    Let me reconsider. The standard result: rank(2x2xN) = 7 + 4*(N-2) for N >= 2.
    So rank(2x2x16) = 7 + 4*14 = 63. That's not 28.
    
    Hmm, the directive mentions "known rank-28 Laderman decomposition for 2x2x16".
    Let me check: Laderman's result is for 3x3x3. For 2x2xN, the rank is indeed
    7 + 4(N-2).
    
    Wait - maybe the directive is referring to a different embedding. Let me re-read:
    "embedding the 9-column dimension into a 16-dimensional cyclic group structure,
    allowing the use of the known rank-28 Laderman decomposition for 2x2x16"
    
    Perhaps this is a hypothetical or I'm misunderstanding. Let me try a different approach:
    try to find a rank-28 decomposition directly via optimization, using the
    cyclic group structure as an initialization hint.
    
    For now, let me implement a search for rank 28, 29, 30, 31 decompositions.
    """
    pass

def _compute_residual(U, V, W, n, m, p):
    """Compute the maximum absolute residual of the tensor decomposition."""
    rank = U.shape[1]
    M = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    T_true = np.zeros((n * m, m * p, p * n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_true[i, j, k] = 1.0
    
    residual = M - T_true
    return np.max(np.abs(residual))

def _try_rank_search(U0, V0, W0, n, m, p, target_rank, time_budget, num_restarts=5):
    """Try to find a decomposition at target_rank using optimization."""
    start_time = time.time()
    
    nm, mp, pn = n * m, m * p, p * n
    
    # Precompute true tensor
    T_true = np.zeros((nm, mp, pn))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_true[i, j, k] = 1.0
    
    best_score = float('inf')
    best_result = None
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_budget * 0.9:
            break
        
        rng = np.random.default_rng(RNG_SEED + restart * 1000 + target_rank)
        
        # Initialize by dropping columns from the rank-32 solution and adding noise
        if target_rank < 32:
            # Drop random columns
            cols_to_keep = rng.choice(32, size=target_rank, replace=False)
            cols_to_keep = np.sort(cols_to_keep)
            U_r = U0[:, cols_to_keep].copy()
            V_r = V0[:, cols_to_keep].copy()
            W_r = W0[:, cols_to_keep].copy()
            
            # Add small noise to help escape local minima
            noise_scale = 0.01
            U_r += rng.normal(0, noise_scale, size=U_r.shape)
            V_r += rng.normal(0, noise_scale, size=V_r.shape)
            W_r += rng.normal(0, noise_scale, size=W_r.shape)
        else:
            U_r = rng.normal(0, 1, size=(nm, target_rank))
            V_r = rng.normal(0, 1, size=(mp, target_rank))
            W_r = rng.normal(0, 1, size=(pn, target_rank))
        
        x0 = np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()])
        
        def objective(params):
            U_flat = params[:nm * target_rank]
            V_flat = params[nm * target_rank : nm * target_rank + mp * target_rank]
            W_flat = params[nm * target_rank + mp * target_rank : nm * target_rank + mp * target_rank + pn * target_rank]
            
            U = U_flat.reshape(nm, target_rank)
            V = V_flat.reshape(mp, target_rank)
            W = W_flat.reshape(pn, target_rank)
            
            M = np.einsum('ir,jr,kr->ijk', U, V, W)
            residual = M - T_true
            
            return np.sum(residual**2)
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B',
                              options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
            
            if result.fun < 1e-6:
                U_flat = result.x[:nm * target_rank]
                V_flat = result.x[nm * target_rank : nm * target_rank + mp * target_rank]
                W_flat = result.x[nm * target_rank + mp * target_rank : nm * target_rank + mp * target_rank + pn * target_rank]
                U_sol = U_flat.reshape(nm, target_rank)
                V_sol = V_flat.reshape(mp, target_rank)
                W_sol = W_flat.reshape(pn, target_rank)
                
                score = _compute_residual(U_sol, V_sol, W_sol, n, m, p)
                if score < 1e-6 and score < best_score:
                    best_score = score
                    best_result = (U_sol.copy(), V_sol.copy(), W_sol.copy(), score)
        except Exception:
            continue
    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # First, build the rank-32 solution as fallback
    U32, V32, W32 = _build_rank32()
    
    def to_list(arr):
        return arr.tolist()
    
    # Try to find lower rank decompositions, starting from rank 28
    for target_rank in [28, 29, 30, 31]:
        remaining_time = deadline - time.time()
        if remaining_time < 10.0:
            break
        
        # Allocate time based on how many ranks we still need to try
        ranks_remaining = 31 - target_rank + 1  # 28,29,30,31 -> 4,3,2,1
        time_budget = min(30.0, remaining_time / max(1, ranks_remaining) * 0.8)
        
        result = _try_rank_search(U32, V32, W32, n, m, p, target_rank, time_budget, num_restarts=3)
        
        if result is not None:
            U_sol, V_sol, W_sol, score = result
            return {"U": to_list(U_sol), "V": to_list(V_sol), "W": to_list(W_sol)}
    
    # Fall back to rank-32
    return {"U": to_list(U32), "V": to_list(V32), "W": to_list(W32)}
