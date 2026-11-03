# EVOLVE-BLOCK-START
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x6 matrix.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 90
    
    # Store best result found so far
    best_rank = schoolbook_rank
    best_result = _schoolbook_decomposition(n, m, p)
    
    # Try to find better decompositions using random integer search
    # We search from rank 89 down to 80
    ranks_to_try = list(range(schoolbook_rank - 1, max(70, schoolbook_rank - 10), -1))
    
    for rank in ranks_to_try:
        if time.time() >= deadline - 5:
            break
            
        # Try to find a decomposition of this rank
        result = _try_random_integer_search(n, m, p, rank, deadline)
        if result is not None:
            best_rank = rank
            best_result = result
            break
            
    return best_result

def _schoolbook_decomposition(n, m, p):
    """Create the standard schoolbook decomposition of rank n*m*p."""
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

def _compute_error(U, V, W, n, m, p):
    """
    Compute the sum of squared errors.
    """
    error = 0.0
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]
            for k in range(p):
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                val = np.dot(u_row * v_row, w_row)
                error += (val - 1.0) ** 2
    return error

def _try_random_integer_search(n, m, p, rank, deadline):
    """
    Random search for integer/half-integer solutions.
    """
    max_attempts = 50
    max_steps = 1000
    
    for attempt in range(max_attempts):
        if time.time() >= deadline - 5:
            return None
            
        # Initialize with small random integers
        U = np.random.randint(-1, 2, (n * m, rank))
        V = np.random.randint(-1, 2, (m * p, rank))
        W = np.random.randint(-1, 2, (p * n, rank))
        
        best_error = _compute_error(U, V, W, n, m, p)
        
        for step in range(max_steps):
            if time.time() >= deadline - 5:
                break
                
            if best_error < 1e-9:
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                
            # Pick a random variable to flip
            var_type = np.random.randint(0, 3)
            old_val = 0
            if var_type == 0:
                idx = np.random.randint(0, n * m)
                col = np.random.randint(0, rank)
                old_val = U[idx, col]
                U[idx, col] += np.random.choice([-1, 1])
            elif var_type == 1:
                idx = np.random.randint(0, m * p)
                col = np.random.randint(0, rank)
                old_val = V[idx, col]
                V[idx, col] += np.random.choice([-1, 1])
            else:
                idx = np.random.randint(0, p * n)
                col = np.random.randint(0, rank)
                old_val = W[idx, col]
                W[idx, col] += np.random.choice([-1, 1])
                
            new_error = _compute_error(U, V, W, n, m, p)
            if new_error < best_error:
                best_error = new_error
            else:
                # Revert
                if var_type == 0:
                    U[idx, col] = old_val
                elif var_type == 1:
                    V[idx, col] = old_val
                else:
                    W[idx, col] = old_val
                
    return None
# EVOLVE-BLOCK-END
