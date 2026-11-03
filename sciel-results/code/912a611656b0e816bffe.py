"""
Rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
Uses local search with structured initialization to find a decomposition
of rank less than the schoolbook rank of 36.
"""

import numpy as np
import random
import time

DIMENSIONS = [3, 3, 4]

def construct():
    """
    Return a tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' where:
        - U has shape (n*m, rank) = (9, rank)
        - V has shape (m*p, rank) = (12, rank)  
        - W has shape (p*n, rank) = (12, rank)
    
    The decomposition satisfies:
        For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    
    # Try to find a decomposition with rank < 36
    # Lower bound: max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    
    # Try ranks from 24 to 35 (we know 36 works)
    # Start with lower ranks and work up
    
    deadline = time.time() + 110  # Leave 10s safety margin
    
    for target_rank in range(24, 36):
        if time.time() > deadline:
            break
        decomp = _try_decomposition(n, m, p, target_rank, deadline)
        if decomp is not None:
            return decomp
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    """Generate the schoolbook decomposition."""
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

def _compute_error(U, V, W, n, m, p, target_rank):
    """Compute the squared error of the decomposition."""
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(target_rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                error += (s - 1.0) ** 2
    return error

def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition is exact."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_decomposition(n, m, p, target_rank, deadline):
    """
    Try to find a decomposition of the given rank using local search.
    Returns None if no solution found, otherwise returns the decomposition.
    """
    # Multiple restarts with different initializations
    max_restarts = 5
    max_iters_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize with sparse random matrices
        U = np.zeros((n * m, target_rank), dtype=int)
        V = np.zeros((m * p, target_rank), dtype=int)
        W = np.zeros((p * n, target_rank), dtype=int)
        
        # For each rank term, pick random positions
        for r in range(target_rank):
            u_idx = random.randint(0, n * m - 1)
            v_idx = random.randint(0, m * p - 1)
            w_idx = random.randint(0, p * n - 1)
            
            u_val = random.choice([-1, 1])
            v_val = random.choice([-1, 1])
            w_val = random.choice([-1, 1])
            
            U[u_idx, r] = u_val
            V[v_idx, r] = v_val
            W[w_idx, r] = w_val
        
        # Local search
        current_error = _compute_error(U, V, W, n, m, p, target_rank)
        
        for it in range(max_iters_per_restart):
            if time.time() > deadline:
                break
            
            if current_error == 0:
                break
            
            # Randomly pick a matrix, row, and column to modify
            mat_choice = random.randint(0, 2)
            if mat_choice == 0:
                mat = U
                rows = n * m
            elif mat_choice == 1:
                mat = V
                rows = m * p
            else:
                mat = W
                rows = p * n
            
            row = random.randint(0, rows - 1)
            col = random.randint(0, target_rank - 1)
            
            old_val = mat[row, col]
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val:
                continue
            
            mat[row, col] = new_val
            new_error = _compute_error(U, V, W, n, m, p, target_rank)
            
            if new_error <= current_error:
                current_error = new_error
            else:
                mat[row, col] = old_val
        
        # Check if we found a perfect decomposition
        if current_error == 0:
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    return None
