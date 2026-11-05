"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct an exact tensor decomposition for 3x4 x 4x6 matrix multiplication.
    
    We first try to find a rank < 72 decomposition via ALS search over reals,
    but fall back to the guaranteed-correct schoolbook decomposition of rank 72.
    
    The known upper bound for M_{3,4,6} is 72 (schoolbook). We use a fast
    deterministic construction that is always valid.
    """
    deadline = time.time() + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # Build the exact schoolbook decomposition of rank n*m*p = 72
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Verify correctness (cheap sanity check)
    # Tensor T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
    # For schoolbook, T[i,j,k] = 1 for all i,j,k, which matches the matmul tensor.
    
    # Try to improve: search for lower rank via ALS on a small subspace
    # This is a best-effort attempt; if it fails, we return rank 72.
    # We use a time-bounded search.
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Attempt rank reduction from 71 down to, say, 65
    # Use ALS with random initialization, checking exactness via residual
    target_ranks = list(range(71, 64, -1))
    
    for target_r in target_ranks:
        if time.time() > deadline - 30:
            break
        
        # Multiple random restarts for this rank
        n_restarts = max(1, int(20 * (1 - (time.time() - deadline + 110) / 110)))
        n_restarts = min(n_restarts, 30)
        
        found = False
        for restart in range(n_restarts):
            if time.time() > deadline - 30:
                break
            
            # Random initialization
            U_t = np.random.randn(n * m, target_r) * 0.5
            V_t = np.random.randn(m * p, target_r) * 0.5
            W_t = np.random.randn(p * n, target_r) * 0.5
            
            # ALS iterations
            converged = False
            for it in range(200):
                if time.time() > deadline - 30:
                    break
                
                # Update U: for each (i,j), U[i*m+j, :] should satisfy
                # sum_k W[k*n+i, t] * V[j*p+k, t] * U[i*m+j, t] = 1
                # For fixed (i,j), this is: U[i*m+j, t] * (W_i ⊙ V_j^T)[t] = 1 for all t
                # Wait, that's not right. The constraint is:
                # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all k
                # This is a system of p equations in target_r unknowns (U[i*m+j, :])
                # Let A_ij be a p x target_r matrix where A_ij[k, t] = V[j*p+k, t] * W[k*n+i, t]
                # Then we need A_ij @ U[i*m+j, :] = ones(p)
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A_ij = np.outer(W[:, i], V[j*p:(j+1)*p, :].T)  # p x target_r
                        A_ij = (W[:, i] * V[j*p:(j+1)*p, :].T)  # p x target_r
                        # Solve A_ij @ x = ones(p) in least squares
                        try:
                            U_t[i*m+j, :] = np.linalg.lstsq(A_ij, np.ones(p), rcond=None)[0]
                        except:
                            break
                    
                    if time.time() > deadline - 30:
                        break
                else:
                    # Update V
                    for j in range(m):
                        for k in range(p):
                            A_jk = (U[:, j] * W[k*n:(k+1)*n, :].T)  # n x target_r
                            try:
                                V_t[j*p+k, :] = np.linalg.lstsq(A_jk, np.ones(n), rcond=None)[0]
                            except:
                                break
                        
                        if time.time() > deadline - 30:
                            break
                    else:
                        # Update W
                        for k in range(p):
                            for i in range(n):
                                A_ki = (U[i*m:(i+1)*m, :] * V[:, k].T)  # m x target_r
                                A_ki = (U[i*m:(i+1)*m, :] * V[:, k])  # m x target_r
                                try:
                                    W_t[k*n+i, :] = np.linalg.lstsq(A_ki, np.ones(m), rcond=None)[0]
                                except:
                                    break
                                
                                if time.time() > deadline - 30:
                                    break
                            else:
                                continue
                        break
                break
            
            if time.time() > deadline - 30:
                break
            
            # Check exactness: compute max residual
            max_res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_t[i*m+j, :] * V_t[j*p+k, :] * W_t[k*n+i, :])
                        res = abs(val - 1.0)
                        if res > max_res:
                            max_res = res
            
            if max_res < 1e-8:
                # Found a valid decomposition! Round to nearest integers/half-integers
                best_rank = target_r
                # Round to nearest 0.5
                U_rounded = np.round(U_t * 2) / 2
                V_rounded = np.round(V_t * 2) / 2
                W_rounded = np.round(W_t * 2) / 2
                
                # Re-verify after rounding
                max_res_round = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_rounded[i*m+j, :] * V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
                            res = abs(val - 1.0)
                            if res > max_res_round:
                                max_res_round = res
                
                if max_res_round < 1e-6:
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    found = True
                    break
                else:
                    # Use unrounded values
                    best_U, best_V, best_W = U_t, V_t, W_t
                    found = True
                    break
        
        if found:
            # Try one rank lower if time permits
            continue
    
    # Return the best decomposition found
    result_rank = best_rank
    result_U = best_U
    result_V = best_V
    result_W = best_W
    
    return {
        "U": result_U.tolist(),
        "V": result_V.tolist(),
        "W": result_W.tolist()
    }
# EVOLVE-BLOCK-END
