"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Helper to verify exact decomposition (within floating point tolerance)
    def is_valid(U, V, W, rank):
        # Check dimensions
        if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
            return False
        # Check values
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Strategy 1: Exploit rectangular structure p=2m.
    # C (4x8) = A (4x4) * B (4x8)
    # Split B into B1 (4x4) and B2 (4x4)
    # C = A*B1 + A*B2
    # If we have a rank R decomposition for 4x4*4x4, we can form a rank 2R decomposition for 4x4*4x8.
    # U_total = [U_4x4, U_4x4] (columns 0..R-1 for first part, R..2R-1 for second)
    # V_total = [V_4x4_B1, V_4x4_B2]
    # W_total = [W_4x4, W_4x4]
    
    # We need a good 4x4*4x4 decomposition.
    # Schoolbook is 64.
    # Known lower bounds for 4x4 are around 49-56.
    # Let's try to find a low-rank 4x4*4x4 decomposition first using random restarts + least squares.
    # Then double it.
    
    # However, the current program's approach of direct optimization on the full tensor is also valid.
    # The prompt suggests: "Model the 4x4 by 4x8 multiplication as the tensor product of two 4x4 by 4x4 multiplications... rank 28 for 4x4 ... rank 56 solution".
    # Rank 28 is likely not achievable for 4x4 (Strassen is 7 for 2x2, 4x4 is at least 49).
    # But let's try to find the best 4x4 rank we can, say R4, and then use 2*R4.
    # Or just optimize the full 4x4x8 tensor directly but with a better initialization based on the block structure.
    
    # Let's try optimizing the full tensor with ranks derived from 4x4 estimates.
    # If 4x4 rank is ~49, then 4x4x8 rank might be ~98?
    # Schoolbook is 128.
    # Let's try ranks: 100, 90, 80, 70.
    
    # Better initialization:
    # Initialize U, V, W such that they respect the block structure initially.
    # U (16 x R), V (32 x R), W (32 x R).
    
    # Let's try a specific structured initialization.
    # For 4x4*4x4, a common decomposition is based on Strassen or similar.
    # But since we don't have a hardcoded Strassen 4x4, we will use random init but maybe scale differently.
    
    # Let's try to optimize directly.
    # To speed up convergence, we can use a simple gradient descent or Levenberg-Marquardt.
    # The current code uses least_squares which is good but can be slow if residuals are computed naively.
    
    # Vectorized residual computation:
    # Residual for (i,j,k) is sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] - 1
    # We can precompute indices.
    
    # Let's try a few target ranks.
    # Based on literature, 4x4x4 rank is 49 (Lader). 4x4x8 might be around 98?
    # Let's try 100, 96, 92, 88.
    
    target_ranks = [100, 96, 92, 88, 84]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break
        if target_rank >= best_rank:
            continue
            
        success = False
        # Use more restarts for lower ranks? Or just a few good ones.
        for restart in range(3):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(100 + restart)
            
            # Initialize with small random values
            U_init = np.random.randn(n * m, target_rank) * 0.1
            V_init = np.random.randn(m * p, target_rank) * 0.1
            W_init = np.random.randn(p * n, target_rank) * 0.1
            
            # Flatten
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Precompute index arrays for vectorization if possible, but least_squares expects a vector.
            # We can compute residuals vectorized inside the function.
            
            def residuals(params):
                U_flat = params[:n*m*target_rank]
                V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank]
                W_flat = params[n*m*target_rank + m*p*target_rank :]
                
                U_mat = U_flat.reshape(n*m, target_rank)
                V_mat = V_flat.reshape(m*p, target_rank)
                W_mat = W_flat.reshape(p*n, target_rank)
                
                # Compute all dot products
                # U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                # This is a sum over t.
                
                res = np.zeros(n*m*p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_row = U_mat[i*m+j, :]
                            v_row = V_mat[j*p+k, :]
                            w_row = W_mat[k*n+i, :]
                            val = np.dot(u_row, v_row * w_row)
                            res[idx] = val - 1.0
                            idx += 1
                return res
            
            try:
                # Use 'lm' for small problems, 'trf' for larger.
                # 16*R + 32*R + 32*R = 80*R variables.
                # For R=100, 8000 variables. LM might be memory heavy or slow.
                # Let's try 'trf' with bounds? No bounds.
                # Actually, least_squares with 'lm' requires no bounds.
                # Let's stick with 'lm' if possible, but it can be slow.
                # Alternative: use scipy.optimize.minimize with BFGS on the sum of squares.
                # Or just use least_squares.
                
                result = least_squares(residuals, x0, method='lm', max_nfev=1000, ftol=1e-8, xtol=1e-8, gtol=1e-8)
                
                if result.cost < 1e-8:
                    U_sol = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_sol = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_sol = result.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                    
                    # Verify
                    if is_valid(U_sol, V_sol, W_sol, target_rank):
                        best_U, best_V, best_W = U_sol, V_sol, W_sol
                        best_rank = target_rank
                        success = True
                        break
            except Exception:
                pass
        
        if success:
            continue
        else:
            # If we fail at a rank, lower ranks are likely harder, but maybe not.
            # However, if we can't solve 100, 96 might be possible? Unlikely if 100 failed due to local minima.
            # But let's continue trying lower ranks as they have fewer variables and might converge easier?
            # Actually, lower rank means tighter constraints, harder to satisfy.
            # So if 100 failed, 96 is harder. We should probably break.
            # But maybe the random seed was bad.
            # Let's try one more set of seeds for the next rank?
            # For now, break to save time.
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

from scipy.optimize import least_squares
# EVOLVE-BLOCK-END
