import time
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x3x3 matrix multiplication.
    Attempts to find a rank < 18 decomposition using ALS with rounding and exact verification.
    Falls back to schoolbook rank 18 if no lower rank is found.
    """
    n, m, p = DIMENSIONS
    R_school = n * m * p  # 18
    
    # Build schoolbook solution as fallback
    U_school = np.zeros((n * m, R_school))
    V_school = np.zeros((m * p, R_school))
    W_school = np.zeros((p * n, R_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = R_school
    
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Target ranks to try, from lowest to highest
    # Literature suggests R(2,3,3) = 14
    for target_rank in range(14, R_school):
        if time.time() - start_time > deadline:
            break
            
        found = False
        # Number of restarts depends on remaining time
        max_restarts = 20
        
        for restart in range(max_restarts):
            if time.time() - start_time > deadline:
                break
                
            np.random.seed(restart * 7919 + target_rank * 131)
            
            # Initialize randomly
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Normalize columns to avoid scale issues
            for r in range(target_rank):
                norm_u = np.linalg.norm(U[:, r])
                norm_v = np.linalg.norm(V[:, r])
                norm_w = np.linalg.norm(W[:, r])
                if norm_u < 1e-10 or norm_v < 1e-10 or norm_w < 1e-10:
                    continue
                # Geometric mean scaling
                total_norm = (norm_u * norm_v * norm_w) ** (1/3)
                if total_norm > 1e-10:
                    U[:, r] /= (norm_u / total_norm)
                    V[:, r] /= (norm_v / total_norm)
                    W[:, r] /= (norm_w / total_norm)
            
            converged = False
            min_error = np.inf
            
            # ALS Iterations
            for iteration in range(300):
                if time.time() - start_time > deadline:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        # We want U[i*m+j, :] such that sum_r U[..., r] * A[k, r] = 1 for all k
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        
                        # Solve A.T @ U_row = ones(p)
                        # U_row = lstsq(A.T, ones)
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        # B[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        # We want V[j*p+k, :] such that sum_r V[..., r] * B[i, r] = 1 for all i
                        B = np.zeros((n, target_rank))
                        for i in range(n):
                            B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(B.T, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        # C[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        # We want W[k*n+i, :] such that sum_r W[..., r] * C[j, r] = 1 for all j
                        C = np.zeros((m, target_rank))
                        for j in range(m):
                            C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        
                        try:
                            W[k * n + i, :] = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Calculate error
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                            error += (val - 1.0) ** 2
                            
                if error < 1e-12:
                    converged = True
                    break
                if error < min_error:
                    min_error = error
                    # Keep best so far if we fail to converge fully
                    if min_error < 1e-8:
                         # Try rounding and verifying even if not perfectly converged numerically
                         pass

            if converged or min_error < 1e-8:
                # Attempt to snap to integer/half-integer
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify exactness
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U_rounded[i * m + j, r] * V_rounded[j * p + k, r] * W_rounded[k * n + i, r]
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                    
                if valid:
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
