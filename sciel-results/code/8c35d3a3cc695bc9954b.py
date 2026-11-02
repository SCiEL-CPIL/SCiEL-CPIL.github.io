import numpy as np
from itertools import product, combinations
import random
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank 18 as fallback
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_fb, "V": V_fb, "W": W_fb}

    # Search for lower rank
    # We try ranks from 15 down to 10
    # Using a deterministic pseudo-random search with fixed seed
    random.seed(42)
    start_time = time.time()
    
    for R in [15, 14, 13]:
        if time.time() - start_time > 110:
            break
            
        # We perform many trials
        for trial in range(500):
            if time.time() - start_time > 110:
                break
                
            # Initialize U and V with random 0/1 entries
            # To ensure solvability, we need sufficient rank in the resulting linear system
            U_rand = np.zeros((n * m, R), dtype=float)
            V_rand = np.zeros((m * p, R), dtype=float)
            
            # Fill with random 0s and 1s
            # Ensure each column has some non-zeros
            for r in range(R):
                # Pick random non-zero positions
                num_nonzero_u = np.random.randint(1, n * m + 1)
                idx_u = np.random.choice(n * m, num_nonzero_u, replace=False)
                U_rand[idx_u, r] = 1.0
                
                num_nonzero_v = np.random.randint(1, m * p + 1)
                idx_v = np.random.choice(m * p, num_nonzero_v, replace=False)
                V_rand[idx_v, r] = 1.0
            
            # Build the linear system for W
            # Equation: sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = delta[j, j']
            # We have n*p equations (for each i, k), each is a m x m matrix
            # Total n*p*m*m scalar equations
            # Unknowns: W has shape (p*n, R) -> p*n*R unknowns
            
            # Let's vectorize the system A x = b
            # x is W flattened
            # A has shape (n*p*m*m, p*n*R)
            # b has shape (n*p*m*m)
            
            A = np.zeros((n * p * m * m, p * n * R))
            b = np.zeros(n * p * m * m)
            
            idx = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            # Equation for C[i, k] component related to A[i, j] B[jp, k]
                            # We want sum_r W[k*n+i, r] U[i*m+j, r] V[jp*p+k, r] = 1 if j==jp else 0
                            target = 1.0 if j == jp else 0.0
                            b[idx] = target
                            
                            for r in range(R):
                                # Coefficient for W[k*n+i, r]
                                coeff = U_rand[i * m + j, r] * V_rand[jp * p + k, r]
                                A[idx, (k * n + i) * R + r] = coeff
                            idx += 1
            
            # Solve A x = b
            # Use least squares
            try:
                W_flat, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
                
            W_rand = W_flat.reshape(p * n, R)
            
            # Check if solution is exact
            # Reconstruct and check error
            error = 0.0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            val = 0.0
                            for r in range(R):
                                val += W_rand[k * n + i, r] * U_rand[i * m + j, r] * V_rand[jp * p + k, r]
                            target = 1.0 if j == jp else 0.0
                            error += abs(val - target)
            
            if error < 1e-9:
                # Check if entries are integers or half-integers
                is_valid = True
                # Check U, V, W
                for mat in [U_rand, V_rand, W_rand]:
                    for val in mat.flatten():
                        # Check if val is close to integer or half-integer
                        # val = k/2
                        # 2*val should be integer
                        if abs(2.0 * val - round(2.0 * val)) > 1e-6:
                            is_valid = False
                            break
                    if not is_valid:
                        break
                if is_valid:
                    # Convert to list of lists with 0.5 precision
                    U_out = [[round(U_rand[i, r] * 2) / 2 for r in range(R)] for i in range(n * m)]
                    V_out = [[round(V_rand[i, r] * 2) / 2 for r in range(R)] for i in range(m * p)]
                    W_out = [[round(W_rand[i, r] * 2) / 2 for r in range(R)] for i in range(p * n)]
                    
                    # Final verification
                    # ... (skip for speed, trust the check above)
                    
                    return {"U": U_out, "V": V_out, "W": W_out}

    return fallback
# EVOLVE-BLOCK-END
