import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Finds a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix
    by a 4x5 matrix. Uses vectorized tensor contraction and least squares optimization.
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
    
    def check_decomposition(U, V, W, T, n, m, p, atol=1e-8):
        """Check if decomposition is exact using einsum"""
        try:
            # U: (nm, R), V: (mp, R), W: (pn, R)
            # We need to reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
            U_reshaped = U.reshape(n, m, -1)
            V_reshaped = V.reshape(m, p, -1)
            W_reshaped = W.reshape(p, n, -1)
            
            # Contract over the last dimension (rank)
            # Result shape: (n, m, p)
            # Note: W is indexed by (k, i) in the prompt's W[k*n+i, r].
            # So W_reshaped[k, i, r].
            # We want sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            # einsum: 'ijr,jkr,kir->ijk'
            result = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            
            # Check if result matches T
            return np.allclose(result, T, atol=atol)
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
        # Compute the tensor product
        # U: (n, m, R), V: (m, p, R), W: (p, n, R)
        result = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        # Residual is difference from T
        return (result - T).flatten()

    def try_optimize(U_init, V_init, W_init, R, max_nfev=500):
        """Try to optimize from given initial conditions"""
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = least_squares(residuals, x0, args=(R,), method='lm', max_nfev=max_nfev)
            cost = result.cost
            if cost < 1e-10:
                U_opt, V_opt, W_opt = flatten_params(result.x, R)
                return U_opt, V_opt, W_opt, cost
        except Exception:
            pass
        
        # Try trf method as fallback
        try:
            result = least_squares(residuals, x0, args=(R,), method='trf', max_nfev=max_nfev)
            cost = result.cost
            if cost < 1e-10:
                U_opt, V_opt, W_opt = flatten_params(result.x, R)
                return U_opt, V_opt, W_opt, cost
        except Exception:
            pass
        
        return None, None, None, float('inf')

    # Strategy: Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    lower_bound = max(n * m, m * p, p * n)
    
    # Try each rank from lower_bound to schoolbook_rank - 1
    for target_rank in range(lower_bound, rank_schoolbook):
        if time.time() > deadline:
            break
        
        found = False
        # Multiple restart strategies
        n_restarts = 8
        for seed in range(n_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(seed * 1000 + target_rank * 7)
            
            # Different initialization strategies
            if seed % 3 == 0:
                # Small random initialization
                U_init = np.random.randn(n, m, target_rank) * 0.1
                V_init = np.random.randn(m, p, target_rank) * 0.1
                W_init = np.random.randn(p, n, target_rank) * 0.1
            elif seed % 3 == 1:
                # Structured initialization: spread the mass
                U_init = np.ones((n, m, target_rank)) / np.sqrt(target_rank)
                V_init = np.ones((m, p, target_rank)) / np.sqrt(target_rank)
                W_init = np.ones((p, n, target_rank)) / np.sqrt(target_rank)
            else:
                # Mixed: some structured, some random
                U_init = np.random.randn(n, m, target_rank) * 0.5 + 0.1
                V_init = np.random.randn(m, p, target_rank) * 0.5 + 0.1
                W_init = np.random.randn(p, n, target_rank) * 0.5 + 0.1
            
            U_opt, V_opt, W_opt, cost = try_optimize(U_init, V_init, W_init, target_rank)
            
            if U_opt is not None and check_decomposition(U_opt, V_opt, W_opt, T, n, m, p):
                best_U = U_opt
                best_V = V_opt
                best_W = W_opt
                best_rank = target_rank
                found = True
                break  # Found a valid decomposition for this rank
        
        if found:
            # Found minimal rank (since we're iterating from low to high)
            break
    
    # If we found a better rank, format the output
    if best_rank < rank_schoolbook:
        # Reshape to (nm, R), (mp, R), (pn, R)
        best_U = best_U.reshape(n * m, best_rank)
        best_V = best_V.reshape(m * p, best_rank)
        best_W = best_W.reshape(p * n, best_rank)
        
        # Try to snap to half-integers for cleaner output
        def to_half_int_matrix(M, atol=1e-6):
            M_rounded = np.round(M * 2) / 2
            if np.allclose(M, M_rounded, atol=atol):
                return M_rounded
            return M
        
        best_U = to_half_int_matrix(best_U)
        best_V = to_half_int_matrix(best_V)
        best_W = to_half_int_matrix(best_W)
        
        # Final verification after snapping
        if not check_decomposition(best_U, best_V, best_W, T, n, m, p):
            # If snapping broke it, use the un-snapped version
            best_U = best_U.reshape(n, m, best_rank).reshape(n * m, best_rank)
            best_V = best_V.reshape(m, p, best_rank).reshape(m * p, best_rank)
            best_W = best_W.reshape(p, n, best_rank).reshape(p * n, best_rank)
            
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
