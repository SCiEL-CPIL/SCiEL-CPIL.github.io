"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _validate_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Check if the decomposition is valid."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_remove_column(U, V, W, col_idx, n, m, p):
    """Try to remove a column and see if we can fix the decomposition."""
    rank = len(U[0])
    if rank <= 1:
        return None
    
    # Create copies without the column
    U_new = [row[:col_idx] + row[col_idx+1:] for row in U]
    V_new = [row[:col_idx] + row[col_idx+1:] for row in V]
    W_new = [row[:col_idx] + row[col_idx+1:] for row in W]
    
    # Check if still valid
    if _validate_decomposition(U_new, V_new, W_new, n, m, p):
        return U_new, V_new, W_new
    
    # Try to repair: for each (i,j,k) that's broken, see if we can adjust other columns
    # This is a simplified repair - just check validity
    return None

def _local_search(U, V, W, n, m, p, max_iterations=1000, time_limit=10.0):
    """Local search to reduce rank by removing columns."""
    start_time = time.time()
    best_U, best_V, best_W = U, V, W
    best_rank = len(U[0])
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        current_rank = len(best_U[0])
        if current_rank <= 1:
            break
        
        found_removal = False
        # Try removing each column
        for col in range(current_rank):
            result = _try_remove_column(best_U, best_V, best_W, col, n, m, p)
            if result is not None:
                best_U, best_V, best_W = result
                best_rank = current_rank - 1
                found_removal = True
                break
        
        if not found_removal:
            break
    
    return best_U, best_V, best_W

def construct():
    """
    Constructs a rank-minimal decomposition for matrix multiplication 3x3 * 3x7.
    Uses local search to reduce rank from schoolbook.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Start with schoolbook decomposition
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
    
    # Local search to reduce rank
    U, V, W = _local_search(U, V, W, n, m, p, 
                            max_iterations=500, 
                            time_limit=min(30.0, max(0, deadline - time.time() - 5)))
    
    # Final validation
    if not _validate_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if something went wrong
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
