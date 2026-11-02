"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 8]

def _validate_decomposition(U, V, W, n, m, p):
    """Validate that the decomposition exactly represents matrix multiplication."""
    r = len(U[0])
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _random_decomposition(n, m, p, rank, seed=None):
    """Generate a random decomposition with small integer/half-integer entries."""
    if seed is not None:
        random.seed(seed)
    U = [[random.choice([0, 1, -1, 0.5, -0.5]) for _ in range(rank)] for _ in range(n*m)]
    V = [[random.choice([0, 1, -1, 0.5, -0.5]) for _ in range(rank)] for _ in range(m*p)]
    W = [[random.choice([0, 1, -1, 0.5, -0.5]) for _ in range(rank)] for _ in range(p*n)]
    return U, V, W

def _try_reduce_rank(n, m, p, current_rank, U, V, W, time_limit, start_time):
    """Try to reduce rank by removing one term at a time."""
    best_rank = current_rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Try removing each column and see if we can still represent the tensor
    for c_remove in range(current_rank):
        if time.time() - start_time > time_limit:
            break
        # Create reduced matrices without column c_remove
        U_red = [row[:c_remove] + row[c_remove+1:] for row in U]
        V_red = [row[:c_remove] + row[c_remove+1:] for row in V]
        W_red = [row[:c_remove] + row[c_remove+1:] for row in W]
        
        if _validate_decomposition(U_red, V_red, W_red, n, m, p):
            best_rank = current_rank - 1
            best_U, best_V, best_W = U_red, V_red, W_red
            # Try to remove more
            new_U, new_V, new_W, new_rank, new_start = _try_reduce_rank(n, m, p, current_rank-1, U_red, V_red, W_red, time_limit, start_time)
            if new_rank < best_rank:
                best_rank = new_rank
                best_U, best_V, best_W = new_U, new_V, new_W
            break
    
    return best_U, best_V, best_W, best_rank, start_time

def _local_search(n, m, p, U, V, W, rank, time_limit, start_time):
    """Local search: try small perturbations to see if we can reduce rank."""
    best_rank = rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Try zeroing out random entries and checking if decomposition still holds
    for iteration in range(10000):
        if time.time() - start_time > time_limit:
            break
        
        # Try setting a random entry to 0 in one of the matrices
        mat_idx = random.randint(0, 2)
        row_idx = random.randint(0, (n*m, m*p, p*n)[mat_idx]-1)
        col_idx = random.randint(0, rank-1)
        
        if mat_idx == 0:
            U[row_idx][col_idx] = 0
            if _validate_decomposition(U, V, W, n, m, p):
                # Keep this change, try to remove more
                pass
            else:
                U[row_idx][col_idx] = random.choice([1, -1, 0.5, -0.5])
        elif mat_idx == 1:
            V[row_idx][col_idx] = 0
            if _validate_decomposition(U, V, W, n, m, p):
                pass
            else:
                V[row_idx][col_idx] = random.choice([1, -1, 0.5, -0.5])
        else:
            W[row_idx][col_idx] = 0
            if _validate_decomposition(U, V, W, n, m, p):
                pass
            else:
                W[row_idx][col_idx] = random.choice([1, -1, 0.5, -0.5])
    
    return best_U, best_V, best_W, best_rank, start_time

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication.
    Returns dict with 'U', 'V', 'W' matrices.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 100  # seconds, leaving margin before the 120s limit
    
    # Start with schoolbook decomposition
    U, V, W = _schoolbook(n, m, p)
    current_rank = n * m * p
    
    # Try to reduce rank by removing redundant terms
    U, V, W, current_rank, start_time = _try_reduce_rank(n, m, p, current_rank, U, V, W, time_limit, start_time)
    
    # If we have time, try local search for further reductions
    if time.time() - start_time < time_limit:
        U, V, W, current_rank, start_time = _local_search(n, m, p, U, V, W, current_rank, time_limit, start_time)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
