import numpy as np
import time
import random

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

def try_random_decomposition(R, time_limit=100):
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Build the tensor T
    # T[i, j, k] = 1
    # We want sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
    
    # Variables: W[k, i, c]
    # Equations: for each i, j, k: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
    
    # We can flatten W to a vector of length R * p * n
    # The coefficient for W[k,i,c] in equation (i,j,k) is U[i,j,c] * V[j,k,c]
    
    # Let's build the matrix A and vector b
    # A has shape (n*m*p, R*p*n)
    # b has shape (n*m*p,)
    
    A = np.zeros((n * m * p, R * p * n))
    b = np.ones(n * m * p)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_idx = i * m * p + j * p + k
                b[eq_idx] = 1
                
                for c in range(R):
                    # Coefficient for W[k, i, c]
                    # W index in vector: c * p * n + k * n + i
                    w_idx = c * p * n + k * n + i
                    coeff = 0 # Will be filled after U, V are chosen
                    # We can't fill A yet because U, V are random
                    pass
    
    # We need to sample U, V first
    
    for attempt in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        # Sample U, V
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        
        for c in range(R):
            for i in range(n):
                for j in range(m):
                    if random.random() < 0.5:
                        U[i * m + j, c] = 1
            for j in range(m):
                for k in range(p):
                    if random.random() < 0.5:
                        V[j * p + k, c] = 1
        
        # Build A
        A = np.zeros((n * m * p, R * p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for c in range(R):
                        w_idx = c * p * n + k * n + i
                        coeff = U[i * m + j, c] * V[j * p + k, c]
                        A[eq_idx, w_idx] = coeff
        
        # Solve A W_flat = b
        try:
            W_flat, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
        
        # Check if W_flat is close to integer/half-integer
        W_rounded = np.round(W_flat * 2) / 2
        if np.allclose(W_flat, W_rounded, atol=1e-6):
            # Found a solution
            W = W_rounded.reshape(R, p, n)
            # Return
            return U.tolist(), V.tolist(), W.transpose(1, 2, 0).reshape(p * n, R).tolist()
            
    return None
