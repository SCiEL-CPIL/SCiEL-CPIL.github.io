import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target rank search
    # Start from a reasonable lower bound
    # Lower bound for 2x3 by 3x4 is likely around 15-16
    # Seed is 24
    
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Precompute indices
    # U rows: i*m + j
    # V rows: j'*p + k
    # W rows: k*n + i
    
    # We want to find U, V, W of rank R
    # such that sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] == 1 if j==j' else 0
    
    # We will search R from 15 to 24
    for R in range(15, 25):
        if time.time() > deadline:
            break
            
        # Try multiple restarts for this R
        max_restarts = 50
        restart_count = 0
        found = False
        
        while not found and restart_count < max_restarts and time.time() < deadline:
            restart_count += 1
            
            # Initialize U, V, W with random small integers
            # Using -1, 0, 1
            U = np.random.randint(-1, 2, (n*m, R), dtype=float)
            V = np.random.randint(-1, 2, (m*p, R), dtype=float)
            W = np.random.randint(-1, 2, (p*n, R), dtype=float)
            
            # ALS iterations
            for _ in range(20):
                if time.time() > deadline:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        row_idx = i * m + j
                        # Build system A x = b
                        # x is U[row_idx, :] (length R)
                        # Equations for all j', k
                        A_sys = np.zeros((m*p, R))
                        b_sys = np.zeros(m*p)
                        idx = 0
                        for jp in range(m): # j'
                            for k in range(p):
                                # Sum_r U[row_idx, r] * V[jp*p+k, r] * W[k*n+i, r] = delta(j, jp)
                                # Coeff for U[row_idx, r] is V[jp*p+k, r] * W[k*n+i, r]
                                A_sys[idx, :] = V[jp*p+k, :] * W[k*n+i, :]
                                b_sys[idx] = 1.0 if j == jp else 0.0
                                idx += 1
                        # Solve least squares
                        U[row_idx, :], residuals, rank, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        
                # Update V
                for jp in range(m): # j'
                    for k in range(p):
                        row_idx = jp * p + k
                        A_sys = np.zeros((n*m, R))
                        b_sys = np.zeros(n*m)
                        idx = 0
                        for i in range(n):
                            for j in range(m):
                                # Sum_r U[i*m+j, r] * V[row_idx, r] * W[k*n+i, r] = delta(j, jp)
                                A_sys[idx, :] = U[i*m+j, :] * W[k*n+i, :]
                                b_sys[idx] = 1.0 if j == jp else 0.0
                                idx += 1
                        V[row_idx, :], residuals, rank, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        
                # Update W
                for k in range(p):
                    for i in range(n):
                        row_idx = k * n + i
                        A_sys = np.zeros((m*m, R))
                        b_sys = np.zeros(m*m)
                        idx = 0
                        for j in range(m):
                            for jp in range(m):
                                # Sum_r U[i*m+j, r] * V[jp*p+k, r] * W[row_idx, r] = delta(j, jp)
                                A_sys[idx, :] = U[i*m+j, :] * V[jp*p+k, :]
                                b_sys[idx] = 1.0 if j == jp else 0.0
                                idx += 1
                        W[row_idx, :], residuals, rank, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
            
            # Check if we have a valid solution
            # Round to nearest half-integer
            U_int = np.round(U * 2) / 2
            V_int = np.round(V * 2) / 2
            W_int = np.round(W * 2) / 2
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(R):
                                val += U_int[i*m+j, r] * V_int[jp*p+k, r] * W_int[k*n+i, r]
                            target = 1.0 if j == jp else 0.0
                            if abs(val - target) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if not valid: break
            if not valid: break
            
            if valid:
                # Convert to lists of floats (or ints if they are integers)
                # The task says integer/half-integer
                return {
                    "U": U_int.tolist(),
                    "V": V_int.tolist(),
                    "W": W_int.tolist()
                }
                
    # If no better solution found, return seed
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
