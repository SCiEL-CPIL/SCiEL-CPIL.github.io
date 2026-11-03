import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 150.0
    
    # Try ranks from low to high
    for R in range(15, 61):
        if time.time() - start_time > deadline - 10:
            break
            
        # Try multiple random initializations
        for attempt in range(500):
            if time.time() - start_time > deadline - 5:
                break
                
            # Initialize U and V with small random integers
            U = np.zeros((n * m, R), dtype=np.float64)
            V = np.zeros((m * p, R), dtype=np.float64)
            
            # Use sparse random matrices
            for i in range(n * m):
                for r in range(R):
                    if random.random() < 0.2:
                        U[i, r] = random.choice([-1, 1])
                        
            for j in range(m * p):
                for r in range(R):
                    if random.random() < 0.2:
                        V[j, r] = random.choice([-1, 1])
            
            # Construct the linear system for W
            A_mat = np.zeros((n * m * p, p * n * R))
            b_vec = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        w_row = k * n + i
                        for r in range(R):
                            coeff = U[i * m + j, r] * V[j * p + k, r]
                            if coeff != 0:
                                A_mat[eq_idx, w_row * R + r] = coeff
                        eq_idx += 1
            
            try:
                W_flat, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except:
                continue
                
            W = W_flat.reshape((p * n, R))
            
            # Check if W is close to half-integers
            W_scaled = W * 2
            W_rounded = np.round(W_scaled)
            
            error = np.linalg.norm(W_scaled - W_rounded)
            
            if error < 1e-5:
                # Verify the solution exactly
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for r in range(R):
                                s += U[i * m + j, r] * V[j * p + k, r] * (W_rounded[k * n + i, r] / 2.0)
                            if abs(s - 1.0) > 1e-5:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
                
                if valid:
                    U_list = [[int(round(U[i, r])) for r in range(R)] for i in range(n * m)]
                    V_list = [[int(round(V[j, r])) for r in range(R)] for j in range(m * p)]
                    W_list = [[float(W_rounded[i, r] / 2.0) for r in range(R)] for i in range(p * n)]
                    return {"U": U_list, "V": V_list, "W": W_list}
            
    # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
