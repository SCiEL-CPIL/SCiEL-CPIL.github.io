"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import numpy as np
import time
import random
import sys

# EVOLVE-BLOCK-START

def _compute_error(U, V, W, n, m, p):
    """Compute the sum of squared errors for the decomposition."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i*m + j]
                v_row = V[j*p + k]
                w_row = W[k*n + i]
                s = np.dot(u_row, v_row * w_row)
                err += (s - 1.0) ** 2
    return err

def _snap_and_verify(U, V, W, n, m, p, tol=1e-6):
    """Try to snap to half-integers and verify. Returns dict if valid, else None."""
    # Try rounding to half-integers
    U_h = np.round(U * 2) / 2
    V_h = np.round(V * 2) / 2
    W_h = np.round(W * 2) / 2
    
    err_h = _compute_error(U_h, V_h, W_h, n, m, p)
    if err_h < 1e-9:
        return {
            "U": U_h.tolist(),
            "V": V_h.tolist(),
            "W": W_h.tolist()
        }
    
    # Try rounding to integers
    U_i = np.round(U)
    V_i = np.round(V)
    W_i = np.round(W)
    
    err_i = _compute_error(U_i, V_i, W_i, n, m, p)
    if err_i < 1e-9:
        return {
            "U": U_i.tolist(),
            "V": V_i.tolist(),
            "W": W_i.tolist()
        }
    
    return None

def _gradient_descent(U, V, W, n, m, p, steps=200, lr=0.01, time_limit=10.0):
    """Perform gradient descent to minimize the error."""
    start = time.time()
    R = U.shape[1]
    
    for step in range(steps):
        if time.time() > start + time_limit:
            break
            
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i*m + j]
                    v_row = V[j*p + k]
                    w_row = W[k*n + i]
                    
                    s = np.dot(u_row, v_row * w_row)
                    err_term = 2 * (s - 1.0)
                    
                    dU[i*m + j] += err_term * v_row * w_row
                    dV[j*p + k] += err_term * u_row * w_row
                    dW[k*n + i] += err_term * u_row * v_row
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
    return U, V, W

def _structured_init(n, m, p, target_rank):
    """Initialize with structure based on known decompositions."""
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    # Use a mix of random and structured initialization
    # Initialize some columns with 1s in specific positions
    num_structured = min(target_rank // 3, 15)
    for col in range(num_structured):
        i = col % n
        j = (col // n) % m
        k = (col // (n * m)) % p
        U[i * m + j, col] = 1.0
        V[j * p + k, col] = 1.0
        W[k * n + i, col] = 1.0
    
    # Random initialization for remaining columns
    for col in range(num_structured, target_rank):
        U[:, col] = np.random.randn(n * m) * 0.3
        V[:, col] = np.random.randn(m * p) * 0.3
        W[:, col] = np.random.randn(p * n) * 0.3
    
    return U, V, W

def _try_rank_with_grad(n, m, p, target_rank, time_limit, num_tries=10):
    """Try to find a decomposition with target_rank using gradient descent."""
    start_time = time.time()
    
    for trial in range(num_tries):
        if time.time() > start_time + time_limit:
            break
            
        if trial == 0:
            # First trial uses structured initialization
            U, V, W = _structured_init(n, m, p, target_rank)
        else:
            # Subsequent trials use random initialization
            U = np.random.randn(n * m, target_rank) * 0.2
            V = np.random.randn(m * p, target_rank) * 0.2
            W = np.random.randn(p * n, target_rank) * 0.2
        
        # Run gradient descent
        remaining = time_limit - (time.time() - start_time)
        if remaining < 1.0:
            break
            
        U, V, W = _gradient_descent(U, V, W, n, m, p, steps=150, lr=0.03, time_limit=remaining * 0.8)
        
        # Check error
        err = _compute_error(U, V, W, n, m, p)
        
        # If error is very small, try to snap to half-integers
        if err < 1e-4:
            result = _snap_and_verify(U, V, W, n, m, p)
            if result is not None:
                return result
                
    return None

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _strassen_block_decomposition(n, m, p):
    """
    Try to find a decomposition using Strassen-like block strategies.
    For 3x5x5, we can view this as multiplying a 3x5 by 5x5.
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Schoolbook fallback
    schoolbook = _schoolbook_decomposition(n, m, p)
    schoolbook_rank = n * m * p
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Search from schoolbook_rank - 1 down to max(1, schoolbook_rank - 25)
    # Don't stop at first failure - continue searching lower ranks
    for target_rank in range(schoolbook_rank - 1, max(1, schoolbook_rank - 25) - 1, -1):
        if time.time() > deadline - 10.0:
            break
            
        time_per_rank = min(15.0, deadline - time.time() - 5.0)
        if time_per_rank < 1.0:
            break
            
        result = _try_rank_with_grad(n, m, p, target_rank, time_per_rank, num_tries=8)
        
        if result is not None:
            best_result = result
            best_rank = target_rank
            # Continue searching for lower rank (don't break)
            
    if best_result is not None:
        return best_result
        
    return schoolbook

# EVOLVE-BLOCK-END
