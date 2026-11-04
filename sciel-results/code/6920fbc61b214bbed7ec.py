import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Finds a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix
    by a 4x5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 115  # 5-second safety margin
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Schoolbook decomposition (rank = n*m*p = 60)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=float)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=float)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    # Build the target tensor
    T = np.ones((n, m, p))
    
    def check_decomposition(U, V, W, T, n, m, p):
        """Check if decomposition is exact using einsum"""
        try:
            U_reshaped = U.reshape(n, m, -1)
            V_reshaped = V.reshape(m, p, -1)
            W_reshaped = W.reshape(p, n, -1)
            result = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            return np.allclose(result, T, atol=1e-9)
        except Exception:
            return False

    def flatten_params(params, R):
        nU = n * m * R
        nV = m * p * R
        U_flat = params[:nU]
        V_flat = params[nU : nU + nV]
        W_flat = params[nU + nV :]
        return U_flat.reshape(n, m, R), V_flat.reshape(m, p, R), W_flat.reshape(p, n, R)

    def residuals(params, R):
        U, V, W = flatten_params(params, R)
        result = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        res_main = (result - T).flatten()
        
        # Constraint to fix scaling: ||V_r||_2^2 = 1
        # V is (m, p, R). We want norm of V[:,:,r] to be 1.
        v_norms = np.linalg.norm(V, axis=(0, 1))
        res_cons = v_norms - 1.0
        
        # We can weight the constraint to ensure it's satisfied without dominating
        # If the main residual is large, we want to minimize it first.
        # But the scaling issue prevents convergence.
        # Let's just concatenate.
        return np.concatenate([res_main, res_cons])

    # Optimization loop
    # Try ranks from lower bound to schoolbook
    # Lower bound is max(n*m, m*p, p*n) = 20
    for target_rank in range(20, 60):
        if time.time() > deadline:
            break
            
        found = False
        # Increase restarts for lower ranks
        num_restarts = 50 if target_rank == 20 else 20
        
        for seed in range(num_restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(seed + target_rank * 1000)
            
            # Initialize
            # Strategy 0: Small random
            if seed % 3 == 0:
                U_init = np.random.randn(n, m, target_rank) * 0.5
                V_init = np.random.randn(m, p, target_rank) * 0.5
                W_init = np.random.randn(p, n, target_rank) * 0.5
            # Strategy 1: Sparse
            elif seed % 3 == 1:
                U_init = np.zeros((n, m, target_rank))
                V_init = np.zeros((m, p, target_rank))
                W_init = np.zeros((p, n, target_rank))
                for r in range(target_rank):
                    U_init[np.random.randint(n), np.random.randint(m), r] = 1
                    V_init[np.random.randint(m), np.random.randint(p), r] = 1
                    W_init[np.random.randint(p), np.random.randint(n), r] = 1
            # Strategy 2: Moderate random with norm fix
            else:
                U_init = np.random.randn(n, m, target_rank)
                V_init = np.random.randn(m, p, target_rank)
                W_init = np.random.randn(p, n, target_rank)
                # Normalize V to satisfy constraint approximately
                v_norms = np.linalg.norm(V_init, axis=(0, 1))
                v_norms[v_norms == 0] = 1
                V_init /= v_norms[None, None, :]
                
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                # Use least_squares
                # 'lm' is fast but sensitive to scaling. 'trf' is more robust.
                # Let's try 'trf' with the constraint residuals.
                result = least_squares(residuals, x0, args=(target_rank,), method='trf', max_nfev=1000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                
                # Check convergence
                # The residual includes the constraint. We need to check the main part.
                U_opt, V_opt, W_opt = flatten_params(result.x, target_rank)
                
                # Verify the main decomposition
                if check_decomposition(U_opt, V_opt, W_opt, T, n, m, p):
                    best_U = U_opt.flatten()
                    best_V = V_opt.flatten()
                    best_W = W_opt.flatten()
                    best_rank = target_rank
                    found = True
                    break
                    
            except Exception:
                continue
        
        if found:
            break # Found a valid decomposition for the current lowest tried rank
    
    # If no lower rank was found, keep schoolbook
    if best_rank == rank_schoolbook:
        best_U, best_V, best_W = U_sb, V_sb, W_sb
    else:
        # Reshape back to (nm, R) etc for output
        best_U = best_U.reshape(n * m, best_rank)
        best_V = best_V.reshape(m * p, best_rank)
        best_W = best_W.reshape(p * n, best_rank)
        
        # Try to snap to half-integers
        def to_half_int_matrix(M):
            M_rounded = np.round(M * 2) / 2
            if np.allclose(M, M_rounded, atol=1e-6):
                return M_rounded
            return M

        U_snapped = to_half_int_matrix(best_U)
        V_snapped = to_half_int_matrix(best_V)
        W_snapped = to_half_int_matrix(best_W)
        
        if check_decomposition(U_snapped, V_snapped, W_snapped, T, n, m, p):
            best_U = U_snapped
            best_V = V_snapped
            best_W = W_snapped
        else:
            # Keep floats if snapping fails, but ensure they are exact enough
            if not check_decomposition(best_U, best_V, best_W, T, n, m, p):
                 # Fallback to schoolbook if something went wrong
                 best_U, best_V, best_W = U_sb, V_sb, W_sb
                 best_rank = rank_schoolbook

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
