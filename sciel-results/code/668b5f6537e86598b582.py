"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # 1. Build Schoolbook as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = start_time + 130.0

    def compute_residual_fast(U_flat, V_flat, W_flat, rank):
        U = U_flat.reshape(n, m, rank)
        V = V_flat.reshape(m, p, rank)
        W = W_flat.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        return np.sum((T - 1.0) ** 2)

    def objective(params):
        U_flat = params[:n * m * rank]
        V_flat = params[n * m * rank : n * m * rank + m * p * rank]
        W_flat = params[n * m * rank + m * p * rank : ]
        return compute_residual_fast(U_flat, V_flat, W_flat, rank)

    # 2. Construct Block Initialization (Rank 75)
    # This structure is closer to potential low-rank solutions than random.
    # We split 5x5 into 2x2, 2x3, 3x2, 3x3 blocks.
    # A is 3x5 -> A1 (3x2), A2 (3x3)
    # B is 5x5 -> B11 (2x2), B12 (2x3), B21 (3x2), B22 (3x3)
    
    # We will try to optimize to rank 71, 72, 73, 74.
    # The block structure itself is rank 75.
    # We initialize with the block structure and let the optimizer merge terms.
    
    # Initialize U, V, W with the block structure
    U_init = np.zeros((n * m, schoolbook_rank))
    V_init = np.zeros((m * p, schoolbook_rank))
    W_init = np.zeros((p * n, schoolbook_rank))
    
    # Fill U_init, V_init, W_init with schoolbook pattern for the blocks
    # This is just the schoolbook, but we can use it as a starting point.
    # Actually, random initialization might be better for finding global minima,
    # but block structure might guide it to the "right" subspace.
    
    # Let's try optimizing from the schoolbook solution itself, 
    # but with a reduced rank constraint? No, dimensions change.
    
    # We will try ranks 71 to 74.
    for target_rank in range(71, 75):
        if time.time() > deadline - 10:
            break
            
        rank = target_rank
        # Initialize with random noise added to a scaled schoolbook?
        # Or just random.
        # Let's try a few random restarts.
        
        max_attempts = 5
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            U_rand = np.random.uniform(-1, 1, n * m * rank)
            V_rand = np.random.uniform(-1, 1, m * p * rank)
            W_rand = np.random.uniform(-1, 1, p * n * rank)
            
            params_init = np.concatenate([U_rand, V_rand, W_rand])
            
            try:
                res = minimize(objective, params_init, method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            except Exception:
                continue
                
            if res.fun < 1e-6:
                params_opt = res.x
                U_opt = params_opt[:n * m * rank].reshape(n * m, rank)
                V_opt = params_opt[n * m * rank : n * m * rank + m * p * rank].reshape(m * p, rank)
                W_opt = params_opt[n * m * rank + m * p * rank : ].reshape(p * n, rank)
                
                U_snap = np.round(U_opt * 2) / 2
                V_snap = np.round(V_opt * 2) / 2
                W_snap = np.round(W_opt * 2) / 2
                
                res_snapped = compute_residual_fast(
                    U_snap.flatten(), V_snap.flatten(), W_snap.flatten(), rank
                )
                
                if res_snapped < 1e-6:
                    U_list = [[float(U_snap[i, j]) for j in range(rank)] for i in range(n * m)]
                    V_list = [[float(V_snap[i, j]) for j in range(rank)] for i in range(m * p)]
                    W_list = [[float(W_snap[i, j]) for j in range(rank)] for i in range(p * n)]
                    
                    if rank < best_rank:
                        best = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                    break # Found solution for this rank, try lower rank? No, we are iterating up.
                    # But we want the MINIMAL. So if we find 71, we should stop?
                    # Yes, 71 is better than 72.
                    # But we iterate 71, 72, 73, 74.
                    # If we find 71, we break the inner loop and the outer loop?
                    # Yes.
                    return best

    return best
# EVOLVE-BLOCK-END
