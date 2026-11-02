"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, n, m, p, tol=1e-8):
    """Vectorized verification."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return np.all(np.abs(T - 1.0) < tol)

def _round_to_half(x):
    """Round to nearest half-integer."""
    return np.round(x * 2) / 2.0

def _try_rank_target(n, m, p, target_rank, num_trials=50, time_limit=20):
    """
    Try to find a decomposition of exactly target_rank using random initialization
    and local search with half-integer constraints.
    """
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    for trial in range(num_trials):
        if time.time() - start_time > time_limit:
            break
        
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        # Initialize with sparse random half-integer patterns
        for r in range(target_rank):
            # Each rank component: choose a small number of active entries
            u_active = rng.choice(n*m, size=rng.randint(3, min(8, n*m)), replace=False)
            v_active = rng.choice(m*p, size=rng.randint(3, min(8, m*p)), replace=False)
            w_active = rng.choice(p*n, size=rng.randint(3, min(8, p*n)), replace=False)
            
            for idx in u_active:
                U[idx, r] = rng.choice([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5])
            for idx in v_active:
                V[idx, r] = rng.choice([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5])
            for idx in w_active:
                W[idx, r] = rng.choice([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5])
        
        # Local search: iteratively fix errors
        for iteration in range(100):
            if time.time() - start_time > time_limit:
                break
            
            if _verify_fast(U, V, W, n, m, p, tol=1e-6):
                break
            
            U_r = U.reshape(n, m, target_rank)
            V_r = V.reshape(m, p, target_rank)
            W_r = W.reshape(p, n, target_rank)
            T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            errors = np.abs(T - 1.0)
            
            worst_idx = np.unravel_index(np.argmax(errors), errors.shape)
            wi, wj, wk = worst_idx
            
            # Try adjusting values at the worst position
            ui = wi * m + wj
            vj = wj * p + wk
            wk_idx = wk * n + wi
            
            # Try flipping or changing values in U, V, W at this position
            for r in range(target_rank):
                if U[ui, r] != 0:
                    old = U[ui, r]
                    for new_val in [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]:
                        if new_val == old:
                            continue
                        U[ui, r] = new_val
                        if _verify_fast(U, V, W, n, m, p, tol=1e-6):
                            U[ui, r] = old  # revert, we'll handle below
                            return _round_to_half(U), _round_to_half(V), _round_to_half(W)
                    U[ui, r] = old
                
                if V[vj, r] != 0:
                    old = V[vj, r]
                    for new_val in [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]:
                        if new_val == old:
                            continue
                        V[vj, r] = new_val
                        if _verify_fast(U, V, W, n, m, p, tol=1e-6):
                            V[vj, r] = old
                            return _round_to_half(U), _round_to_half(V), _round_to_half(W)
                    V[vj, r] = old
                
                if W[wk_idx, r] != 0:
                    old = W[wk_idx, r]
                    for new_val in [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]:
                        if new_val == old:
                            continue
                        W[wk_idx, r] = new_val
                        if _verify_fast(U, V, W, n, m, p, tol=1e-6):
                            W[wk_idx, r] = old
                            return _round_to_half(U), _round_to_half(V), _round_to_half(W)
                    W[wk_idx, r] = old
    
    return None, None, None

def _try_block_strassen(n, m, p, time_limit=15):
    """
    Try a block decomposition using Strassen-like ideas for 2x2 blocks.
    Partition: A (4x5) -> A1(2x5), A2(2x5); B (5x5) -> B1(5x3), B2(5x2)
    C = [[A1*B1, A1*B2], [A2*B1, A2*B2]]
    """
    start_time = time.time()
    
    # For now, use schoolbook for each block and see if we can share
    # A1*B1: 2x5 * 5x3 -> 2x3, rank 6
    # A1*B2: 2x5 * 5x2 -> 2x2, rank 4
    # A2*B1: 2x5 * 5x3 -> 2x3, rank 6
    # A2*B2: 2x5 * 5x2 -> 2x2, rank 4
    # Total: 20
    
    # This is still 100 with schoolbook per block. Need better.
    return None, None, None, 100

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # Safety margin before 150s
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    # Try decreasing target ranks
    for target_rank in [90, 85, 80, 75, 70]:
        if time.time() > deadline - 10:
            break
        
        U_t, V_t, W_t = _try_rank_target(n, m, p, target_rank, 
                                          num_trials=30, 
                                          time_limit=min(20, deadline - time.time() - 5))
        
        if U_t is not None and _verify_fast(U_t, V_t, W_t, n, m, p, tol=1e-6):
            best_U, best_V, best_W = U_t, V_t, W_t
            best_rank = target_rank
            break
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
