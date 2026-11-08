"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def check_decomposition(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = U[i * m + j, :]
                    v = V[j * p + k, :]
                    w = W[k * n + i, :]
                    val = np.dot(u * v, w)
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    def error_decomposition(U, V, W):
        """Compute the error of the decomposition."""
        r = U.shape[1]
        total_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = U[i * m + j, :]
                    v = V[j * p + k, :]
                    w = W[k * n + i, :]
                    val = np.dot(u * v, w)
                    total_err += (val - 1) ** 2
        return total_err
    
    def coordinate_descent(U, V, W, max_iter=200, tol=1e-10):
        """Optimize one matrix at a time using least squares."""
        U = U.astype(float)
        V = V.astype(float)
        W = W.astype(float)
        r = U.shape[1]
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            prev_err = error_decomposition(U, V, W)
            
            # Optimize U: for each row i*m+j, solve least squares
            for i in range(n):
                for j in range(m):
                    row_idx = i * m + j
                    # We need: sum_c U[row_idx, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
                    # Let A[row_idx, c] = V[j*p+k, c] * W[k*n+i, c] for each k
                    # This is: sum_c U[row_idx, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1
                    # Stack for all k: system of p equations in r unknowns
                    A = np.zeros((p, r))
                    b = np.ones(p)
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve least squares
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        U[row_idx, :] = sol
                    except:
                        pass
            
            # Optimize V: for each row j*p+k
            for j in range(m):
                for k in range(p):
                    row_idx = j * p + k
                    A = np.zeros((n, r))
                    b = np.ones(n)
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        V[row_idx, :] = sol
                    except:
                        pass
            
            # Optimize W: for each row k*n+i
            for k in range(p):
                for i in range(n):
                    row_idx = k * n + i
                    A = np.zeros((m, r))
                    b = np.ones(m)
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        W[row_idx, :] = sol
                    except:
                        pass
            
            curr_err = error_decomposition(U, V, W)
            if abs(prev_err - curr_err) < tol:
                break
        
        # Round to nearest half-integer and check
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        if check_decomposition(U_rounded, V_rounded, W_rounded):
            return U_rounded, V_rounded, W_rounded, True
        else:
            return U_rounded, V_rounded, W_rounded, False
    
    # Try various ranks from low to high
    for target_rank in range(50, 97):
        if time.time() > deadline:
            break
        
        if target_rank >= best_rank:
            continue
        
        found = False
        attempts = 0
        max_attempts = 30
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Random initialization with small integers and half-integers
            U = np.random.randint(-4, 5, (n * m, target_rank)) / 2.0
            V = np.random.randint(-4, 5, (m * p, target_rank)) / 2.0
            W = np.random.randint(-4, 5, (p * n, target_rank)) / 2.0
            
            # Run coordinate descent
            U_opt, V_opt, W_opt, success = coordinate_descent(U, V, W, max_iter=100)
            
            if success:
                found = True
                best_rank = target_rank
                best_result = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
                break
        
        if found:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
