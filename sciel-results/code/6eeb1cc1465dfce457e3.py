"""
Strategy: Stochastic Local Search for Rank-Minimal Tensor Decomposition.
We target a rank of 40 for the 2x3x9 matrix multiplication tensor.
We initialize with random integer/half-integer coefficients and use
simulated annealing to drive the reconstruction error to exactly 0.
"""

import time
import random
import math

DIMENSIONS = [2, 3, 9]
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin for 150s limit

def construct():
    n, m, p = DIMENSIONS
    target_rank = 40  # We attempt to find a rank-40 decomposition
    
    # Initialize U, V, W with random small integers/half-integers
    # Using a set of allowed values to keep search space manageable
    allowed_vals = [-1, -0.5, 0, 0.5, 1]
    
    U = [[random.choice(allowed_vals) for _ in range(target_rank)] for _ in range(n * m)]
    V = [[random.choice(allowed_vals) for _ in range(target_rank)] for _ in range(m * p)]
    W = [[random.choice(allowed_vals) for _ in range(target_rank)] for _ in range(p * n)]
    
    def compute_error():
        err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(target_rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    err += s * s
        return err

    best_error = compute_error()
    if best_error == 0:
        return {"U": U, "V": W, "W": W} # Placeholder, actual return below

    current_error = best_error
    current_U, current_V, current_W = U, V, W
    
    temp = 10.0
    cooling = 0.999
    
    while time.time() - START_TIME < TIME_LIMIT:
        # Pick a random matrix and index to perturb
        mat_idx = random.randint(0, 2)
        if mat_idx == 0:
            mat = current_U
            rows = n * m
        elif mat_idx == 1:
            mat = current_V
            rows = m * p
        else:
            mat = current_W
            rows = p * n
            
        r = random.randint(0, rows - 1)
        c = random.randint(0, target_rank - 1)
        
        old_val = mat[r][c]
        new_val = random.choice(allowed_vals)
        if new_val == old_val:
            continue
            
        mat[r][c] = new_val
        new_error = compute_error()
        
        if new_error < current_error or (random.random() < math.exp((current_error - new_error) / temp)):
            current_error = new_error
            if current_error < best_error:
                best_error = current_error
                best_U, best_V, best_W = [row[:] for row in current_U], [row[:] for row in current_V], [row[:] for row in current_W]
                if best_error == 0:
                    break
        else:
            mat[r][c] = old_val
            
        temp *= cooling
        if temp < 0.01:
            temp = 0.01

    if best_error == 0:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # If we fail to find rank 40, fallback to schoolbook rank 54
    rank = n * m * p
    U_fallback = [[0] * rank for _ in range(n * m)]
    V_fallback = [[0] * rank for _ in range(m * p)]
    W_fallback = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i*m+j][col] = 1
                V_fallback[j*p+k][col] = 1
                W_fallback[k*n+i][col] = 1
                col += 1
    return {"U": U_fallback, "V": V_fallback, "W": W_fallback}
