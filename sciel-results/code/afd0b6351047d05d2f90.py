import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Target tensor T[i,j,k] = 1
    # We search for rank R
    # Start from a reasonable guess, go down
    # Schoolbook is 72. 
    # Let's try to find a rank < 72.
    
    best_result = None
    best_rank = n * m * p
    
    # Pre-allocate schoolbook as fallback
    U_fb = [[0] * best_rank for _ in range(n * m)]
    V_fb = [[0] * best_rank for _ in range(m * p)]
    W_fb = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}

    # Try to find better rank
    # R must be at least max(n*m, m*p, p*n) ? No, lower bound is max(n*m, m*p, p*n) is not true.
    # Lower bound is max(n*m, m*p, p*n) is for border rank? No.
    # Simple lower bound: R >= max(n*m, m*p, p*n) is false.
    # R >= n*m is true? No.
    # Let's just iterate R from 65 down to 40.
    
    for R in range(65, 39, -1):
        if time.time() > deadline:
            break
        
        # Number of attempts per R
        # Adjust attempts based on remaining time?
        # Let's do a fixed number of fast attempts.
        attempts = 200
        
        for _ in range(attempts):
            if time.time() > deadline:
                break
            
            # Random U, V with entries in {-1, 0, 1}
            # Ensure some non-zeros to avoid trivial null spaces
            U = np.random.randint(-1, 2, size=(n * m, R))
            V = np.random.randint(-1, 2, size=(m * p, R))
            
            # We need to solve for W
            # For each i, k:
            # Sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all j
            # This is a linear system M[i,k] @ W[k,i] = ones(m)
            
            W_valid = True
            W_solution = np.zeros((p * n, R))
            
            for i in range(n):
                for k in range(p):
                    # Construct M[i,k] of shape (m, R)
                    # M[j, c] = U[i*m + j, c] * V[j*p + k, c]
                    # U is (n*m, R), so U[i*m + j, c]
                    # V is (m*p, R), so V[j*p + k, c]
                    
                    M = np.zeros((m, R))
                    for j in range(m):
                        M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    # Solve M @ w = ones(m)
                    # Use least squares
                    try:
                        w, residuals, rank, s = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    except:
                        W_valid = False
                        break
                    
                    # Check residual
                    if residuals and residuals[0] > 1e-9:
                        W_valid = False
                        break
                    
                    # Try to round to integers
                    w_int = np.round(w)
                    
                    # Check if w_int is a solution
                    # M @ w_int should be close to ones(m)
                    if not np.allclose(M @ w_int, np.ones(m), atol=1e-6):
                        # Maybe try nearest integers with small offsets?
                        # Or just reject
                        W_valid = False
                        break
                    
                    W_solution[k * n + i, :] = w_int
            
            if W_valid:
                # Verify full decomposition exactly
                # C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_c U[i,j,c] V[j,k,c] W[k,i,c]
                # We need sum_j sum_c ... = delta_{i,i'} delta_{k,k'}?
                # No, the tensor decomposition defines the multiplication.
                # The condition is that the reconstructed tensor matches the identity tensor T.
                # T[i,j,k] = 1.
                # Let's verify T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] == 1 for all i,j,k.
                
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(R):
                                val += U[i * m + j, c] * V[j * p + k, c] * W_solution[k * n + i, c]
                            if abs(val - 1) > 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Convert to list of lists of integers
                    U_list = [[int(x) for x in row] for row in U]
                    V_list = [[int(x) for x in row] for row in V]
                    W_list = [[int(x) for x in row] for row in W_solution]
                    return {"U": U_list, "V": V_list, "W": W_list}

    return best_result
# EVOLVE-BLOCK-END
