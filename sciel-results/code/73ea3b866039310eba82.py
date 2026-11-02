import numpy as np
import time
import random

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Try to find lower rank
    # The tensor T has T[i, j, k] = 1 for all i, j, k
    # We want sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    
    for target_rank in range(rank_sb - 1, 10, -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            # Random U, V
            U = np.random.rand(n * m, target_rank)
            V = np.random.rand(m * p, target_rank)
            
            # Build linear system for W
            # Equations: for each (i, j, k), sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
            # Variables: W[k*n+i, r]
            # Total variables: p * n * target_rank
            # Total equations: n * m * p
            
            num_vars = p * n * target_rank
            num_eqs = n * m * p
            
            A = np.zeros((num_eqs, num_vars))
            b = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Coefficients for W[k*n+i, r]
                        for r in range(target_rank):
                            var_idx = (k * n + i) * target_rank + r
                            A[eq_idx, var_idx] = U[i * m + j, r] * V[j * p + k, r]
                        eq_idx += 1
            
            # Solve A W_flat = b
            # Use least squares
            try:
                W_flat, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                # Check if solution is exact (residuals should be 0)
                if np.allclose(A @ W_flat, b, atol=1e-9):
                    W = W_flat.reshape(p * n, target_rank)
                    # Verify
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U[i * m + j], V[j * p + k]) # No, sum over r
                                # val = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                                # This is not a simple dot product of U and V rows.
                                # It's sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
                                # Let's compute it properly
                                prod = U[i * m + j] * V[j * p + k]
                                val = np.dot(prod, W[k * n + i])
                                if not np.isclose(val, 1.0, atol=1e-9):
                                    valid = False
                                    break
                        if not valid: break
                    if valid:
                        # Convert to list of lists
                        best_result = {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                        best_rank = target_rank
                        break # Found a rank, try lower
            except Exception:
                pass
                
    return best_result
