"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5-second safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
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
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if decomposition is exact"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - T[i,j,k]) > tol:
                        return False
        return True
    
    # Try to find decompositions with rank < 60
    # Start from rank 20 and go up
    from scipy.optimize import least_squares
    
    # Lower bound for matrix multiplication tensor is max(n*m, m*p, p*n) = max(12,20,15) = 20
    # But in practice, we need to search more carefully
    for target_rank in range(20, 60):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for seed in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(target_rank * 100 + seed)
            U_init = np.random.randn(n*m, target_rank)
            V_init = np.random.randn(m*p, target_rank)
            W_init = np.random.randn(p*n, target_rank)
            
            # Flatten parameters
            def flatten_params(params):
                U_flat = params[:n*m*target_rank]
                V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank]
                W_flat = params[n*m*target_rank + m*p*target_rank :]
                return U_flat.reshape(n*m, target_rank), V_flat.reshape(m*p, target_rank), W_flat.reshape(p*n, target_rank)
            
            def residuals(params):
                U, V, W = flatten_params(params)
                res = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            res.append(val - T[i,j,k])
                return np.array(res)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=500)
                if result.success or result.cost < 1e-10:
                    U_opt, V_opt, W_opt = flatten_params(result.x)
                    if check_decomposition(U_opt, V_opt, W_opt):
                        best_U, best_V, best_W = U_opt, V_opt, W_opt
                        best_rank = target_rank
                        break  # Found a valid lower-rank decomposition
            except Exception:
                continue
        
        if best_rank < rank_schoolbook:
            break
    
    # Convert to integer/half-integer if possible
    def to_half_int_matrix(M):
        """Try to convert to half-integers"""
        M_rounded = np.round(M * 2) / 2
        if np.allclose(M, M_rounded, atol=1e-6):
            return M_rounded
        return M
    
    best_U = to_half_int_matrix(best_U)
    best_V = to_half_int_matrix(best_V)
    best_W = to_half_int_matrix(best_W)
    
    # Verify final result
    if not check_decomposition(best_U, best_V, best_W):
        # Fall back to schoolbook
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = rank_schoolbook
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
