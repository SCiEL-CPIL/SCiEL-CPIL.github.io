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

def _compute_error_fast(U, V, W, n, m, p):
    """Faster error computation using vectorized operations."""
    R = U.shape[1]
    # Precompute for each (i,j,k) the product v*w
    # U shape: (n*m, R)
    # V shape: (m*p, R)
    # W shape: (p*n, R)
    
    err = 0.0
    for i in range(n):
        for j in range(m):
            u_row = U[i*m + j]  # shape (R,)
            for k in range(p):
                v_row = V[j*p + k]  # shape (R,)
                w_row = W[k*n + i]  # shape (R,)
                s = np.dot(u_row, v_row * w_row)
                err += (s - 1.0) ** 2
    return err

def _snap_and_verify(U, V, W, n, m, p, tol=1e-9):
    """Try to snap to half-integers and verify. Returns dict if valid, else None."""
    # Try rounding to half-integers
    U_h = np.round(U * 2) / 2
    V_h = np.round(V * 2) / 2
    W_h = np.round(W * 2) / 2
    
    err_h = _compute_error(U_h, V_h, W_h, n, m, p)
    if err_h < tol:
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
    if err_i < tol:
        return {
            "U": U_i.tolist(),
            "V": V_i.tolist(),
            "W": W_i.tolist()
        }
    
    return None

def _gradient_descent_adaptive(U, V, W, n, m, p, steps=500, lr_init=0.05, lr_decay=0.999, time_limit=10.0):
    """Perform adaptive gradient descent to minimize the error."""
    start = time.time()
    R = U.shape[1]
    lr = lr_init
    
    for step in range(steps):
        if time.time() > start + time_limit:
            break
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                u_row = U[i*m + j]
                for k in range(p):
                    v_row = V[j*p + k]
                    w_row = W[k*n + i]
                    
                    s = np.dot(u_row, v_row * w_row)
                    err_term = 2 * (s - 1.0)
                    
                    dU[i*m + j] += err_term * v_row * w_row
                    dV[j*p + k] += err_term * u_row * w_row
                    dW[k*n + i] += err_term * u_row * v_row
        
        # Adaptive learning rate
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        lr *= lr_decay
        
    return U, V, W

def _try_rank_with_grad(n, m, p, target_rank, time_limit, num_tries=8):
    """Try to find a decomposition with target_rank using gradient descent."""
    start_time = time.time()
    
    for trial in range(num_tries):
        if time.time() > start_time + time_limit:
            break
            
        # Random initialization with varying scales
        scale = 0.05 + 0.1 * random.random()
        U = np.random.randn(n * m, target_rank) * scale
        V = np.random.randn(m * p, target_rank) * scale
        W = np.random.randn(p * n, target_rank) * scale
        
        # Run gradient descent
        remaining = time_limit - (time.time() - start_time)
        if remaining < 1.0:
            break
            
        U, V, W = _gradient_descent_adaptive(U, V, W, n, m, p, steps=200, lr_init=0.05, time_limit=remaining * 0.7)
        
        # Check error
        err = _compute_error(U, V, W, n, m, p)
        
        # If error is very small, try to snap to half-integers
        if err < 1e-3:
            result = _snap_and_verify(U, V, W, n, m, p, tol=1e-8)
            if result is not None:
                return result
                
    return None

def _embed_5x5x5_into_3x5x5(U5, V5, W5, n=3, m=5, p=5):
    """
    Embed a 5x5x5 decomposition into a 3x5x5 decomposition.
    U5 shape: (5*5, R) = (25, R)
    V5 shape: (5*5, R) = (25, R)
    W5 shape: (5*5, R) = (25, R)
    
    For 3x5x5:
    U shape: (3*5, R) = (15, R) - take rows 0..14 (i=0,1,2; j=0..4)
    V shape: (5*5, R) = (25, R) - all rows
    W shape: (5*3, R) = (15, R) - take rows where k=0..4, i=0..2
      W5 index: k*5 + i, so for i=0,1,2: rows 0,1,2, 5,6,7, 10,11,12, 15,16,17, 20,21,22
    """
    R = U5.shape[1]
    
    # U: rows i*5+j for i in 0..2, j in 0..4 -> rows 0..14
    U3 = U5[0:15, :].copy()
    
    # V: all rows (m*p = 25)
    V3 = V5.copy()
    
    # W: rows k*5+i for k in 0..4, i in 0..2
    w_rows = []
    for k in range(5):
        for i in range(3):
            w_rows.append(k * 5 + i)
    W3 = W5[w_rows, :].copy()
    
    return U3, V3, W3

def _try_5x5x5_decomposition(n, m, p, target_rank, time_limit):
    """
    Try to find a 5x5x5 decomposition of given rank, then embed into 3x5x5.
    """
    start_time = time.time()
    N = 5  # square dimension
    
    num_tries = 5
    for trial in range(num_tries):
        if time.time() > start_time + time_limit:
            break
            
        scale = 0.05 + 0.1 * random.random()
        U5 = np.random.randn(N * N, target_rank) * scale
        V5 = np.random.randn(N * N, target_rank) * scale
        W5 = np.random.randn(N * N, target_rank) * scale
        
        remaining = time_limit - (time.time() - start_time)
        if remaining < 1.0:
            break
        
        # Gradient descent on 5x5x5
        U5, V5, W5 = _gradient_descent_adaptive(U5, V5, W5, N, N, N, steps=150, lr_init=0.04, time_limit=remaining * 0.7)
        
        # Check 5x5x5 error
        err5 = _compute_error(U5, V5, W5, N, N, N)
        
        if err5 < 1e-3:
            # Embed into 3x5x5
            U3, V3, W3 = _embed_5x5x5_into_3x5x5(U5, V5, W5, n, m, p)
            
            # Check 3x5x5 error
            err3 = _compute_error(U3, V3, W3, n, m, p)
            
            if err3 < 1e-3:
                result = _snap_and_verify(U3, V3, W3, n, m, p, tol=1e-8)
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Schoolbook fallback
    schoolbook = _schoolbook_decomposition(n, m, p)
    schoolbook_rank = n * m * p  # 75
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Strategy: Try to find decompositions of rank < 75
    # Use both direct 3x5x5 search and 5x5x5 embedding
    
    # Try ranks from 74 down to 50
    # If we find a valid rank R, continue trying R-1
    # Since valid ranks form an interval, if R fails, all lower ranks fail
    
    # First, try the embedding approach for a few ranks
    for target_rank in range(74, 49, -1):
        if time.time() > deadline - 15.0:
            break
        
        remaining = deadline - time.time() - 5.0
        if remaining < 2.0:
            break
        
        # Try embedding approach first (might be more effective)
        time_per_method = remaining * 0.5
        result = _try_5x5x5_decomposition(n, m, p, target_rank, time_per_method)
        
        if result is None and time.time() < deadline - 10.0:
            # Try direct approach
            time_left = deadline - time.time() - 5.0
            if time_left > 1.0:
                result = _try_rank_with_grad(n, m, p, target_rank, time_left, num_tries=3)
        
        if result is not None:
            best_result = result
            best_rank = target_rank
            # Continue searching for lower rank
        else:
            # If this rank failed, lower ranks should also fail (interval property)
            # But be cautious - numerical issues might cause false negatives
            # Try one more rank lower before breaking
            break
    
    if best_result is not None:
        return best_result
    
    return schoolbook

# EVOLVE-BLOCK-END
