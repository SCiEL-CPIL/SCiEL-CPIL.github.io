"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build schoolbook as fallback
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
    deadline = start_time + 130.0  # safety margin

    # Precompute indices for efficient residual calculation if needed, 
    # but einsum is likely faster for small dims.
    
    def compute_residual_fast(U_flat, V_flat, W_flat, rank):
        U = U_flat.reshape(n, m, rank)
        V = V_flat.reshape(m, p, rank)
        W = W_flat.reshape(p, n, rank)
        
        # Calculate tensor T_ijk = sum_c U_ijk V_jkc W_kic
        # Note: U index is (i, j, c), V is (j, k, c), W is (k, i, c)
        # einsum: 'ijc,jkc,kic->ijk'
        try:
            T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        except Exception:
            # Fallback to loops if einsum fails or is too slow (unlikely)
            T = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] = np.sum(U[i, j, :] * V[j, k, :] * W[k, i, :])
        
        return np.sum((T - 1.0) ** 2)

    def objective(params):
        U_flat = params[:n * m * rank]
        V_flat = params[n * m * rank : n * m * rank + m * p * rank]
        W_flat = params[n * m * rank + m * p * rank : ]
        return compute_residual_fast(U_flat, V_flat, W_flat, rank)

    # Try ranks from 12 up to schoolbook_rank - 1
    # We can start higher and go down, or low and up. 
    # Known lower bound for 3x5x5 is likely around 15-20.
    # Schoolbook is 75.
    # Let's try a range.
    min_rank_to_try = 12
    max_rank_to_try = min(schoolbook_rank - 1, 40)
    
    # Iterate from low to high? Or high to low?
    # Usually finding a solution at a lower rank is harder.
    # But if we find one, we want the lowest.
    # Let's try increasing rank. If we find a solution at rank R, we check R-1?
    # Or just scan.
    
    for rank in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() > deadline - 20:
            break
            
        # Number of attempts depends on remaining time
        remaining_time = deadline - 5 - time.time()
        if remaining_time <= 0:
            break
            
        # Estimate time per attempt. L-BFGS-B on small problem is fast.
        # Let's say 0.5s per attempt.
        max_attempts = int(remaining_time / 0.5)
        if max_attempts < 1:
            break
        if max_attempts > 100:
            max_attempts = 100

        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
                
            # Random initialization
            # Scale matters. 
            U_init = np.random.uniform(-1, 1, n * m * rank)
            V_init = np.random.uniform(-1, 1, m * p * rank)
            W_init = np.random.uniform(-1, 1, p * n * rank)
            
            params_init = np.concatenate([U_init, V_init, W_init])
            
            # Optimize
            try:
                res = minimize(objective, params_init, method='L-BFGS-B', 
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            except Exception:
                continue
                
            if res.success or res.fun < 1e-6:
                # Extract solution
                params_opt = res.x
                U_opt = params_opt[:n * m * rank].reshape(n * m, rank)
                V_opt = params_opt[n * m * rank : n * m * rank + m * p * rank].reshape(m * p, rank)
                W_opt = params_opt[n * m * rank + m * p * rank : ].reshape(p * n, rank)
                
                # Snap to half-integers
                U_snap = np.round(U_opt * 2) / 2
                V_snap = np.round(V_opt * 2) / 2
                W_snap = np.round(W_opt * 2) / 2
                
                # Verify exactness with snapped values
                # Use fast residual check on snapped values
                res_snapped = compute_residual_fast(
                    U_snap.flatten(), V_snap.flatten(), W_snap.flatten(), rank
                )
                
                if res_snapped < 1e-6:
                    # Found valid solution
                    U_list = [[float(U_snap[i, j]) for j in range(rank)] for i in range(n * m)]
                    V_list = [[float(V_snap[i, j]) for j in range(rank)] for i in range(m * p)]
                    W_list = [[float(W_snap[i, j]) for j in range(rank)] for i in range(p * n)]
                    
                    if rank < best_rank:
                        best = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        # If we found a solution, we can try to go lower
                        # But since we are iterating up, we need to restart or just break and continue loop?
                        # The loop continues to next rank (higher). 
                        # We want the MINIMAL rank. 
                        # So if we found rank R, we should stop searching for higher ranks 
                        # and maybe try to find lower ones? 
                        # But we started from 12. If we found 15, we should have found 12-14 already?
                        # Unless they failed. 
                        # To be safe, if we find a solution, we can set a flag to stop 
                        # searching higher ranks, but we might have missed lower ones 
                        # if they were hard. 
                        # However, usually lower rank is harder. 
                        # If we iterate 12, 13, 14... and find 15, it means 12-14 failed.
                        # So 15 is the current best. 
                        # We can break the inner loop and continue to next rank? 
                        # No, next rank is 16, which is worse. 
                        # So we should break the outer loop too? 
                        # Yes, if we found a solution at rank R, any R' > R is worse.
                        # We only care about the minimum.
                        break
        if best_rank == rank:
            # Found solution at this rank, no need to try higher ranks
            break
    
    return best
# EVOLVE-BLOCK-END
