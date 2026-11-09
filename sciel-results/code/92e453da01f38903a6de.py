"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 5]
SEED_TIME_BUDGET = 120.0  # seconds

def _verify(U, V, W, n, m, p):
    """Check if decomposition is exact."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    u_val = U[i*m+j][c] if isinstance(U, list) else U[i*m+j, c]
                    v_val = V[j*p+k][c] if isinstance(V, list) else V[j*p+k, c]
                    w_val = W[k*n+i][c] if isinstance(W, list) else W[k*n+i, c]
                    s += u_val * v_val * w_val
                if abs(s - 1.0) > 1e-8:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
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
    """Compute total error of decomposition."""
    rank = len(U[0])
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                error += abs(s - 1.0)
    return error

def _structured_search(rank, n, m, p, seed=42, max_iters=2000):
    """
    Search for low-rank decomposition using structured random rank-1 terms
    with local polishing.
    """
    random.seed(seed)
    
    coeff_range = [-2, -1, 1, 2]
    
    best_error = float('inf')
    best_solution = None
    
    for iteration in range(max_iters):
        # Generate random sparse structure for each rank-1 term
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for c in range(rank):
            # Randomly select non-zero entries with small integer coefficients
            num_nonzero_u = random.randint(1, min(4, n * m))
            num_nonzero_v = random.randint(1, min(4, m * p))
            num_nonzero_w = random.randint(1, min(4, p * n))
            
            u_indices = random.sample(range(n * m), num_nonzero_u)
            v_indices = random.sample(range(m * p), num_nonzero_v)
            w_indices = random.sample(range(p * n), num_nonzero_w)
            
            for idx in u_indices:
                U[idx][c] = random.choice(coeff_range)
            for idx in v_indices:
                V[idx][c] = random.choice(coeff_range)
            for idx in w_indices:
                W[idx][c] = random.choice(coeff_range)
        
        error = _compute_error(U, V, W, n, m, p)
        
        if error == 0:
            return U, V, W
        
        if error < best_error:
            best_error = error
            best_solution = (U, V, W)
    
    # Polish the best solution found
    if best_solution is not None:
        polished = _polish_solution(best_solution, n, m, p, seed=seed + 1000, max_iters=3000)
        if polished is not None:
            return polished
    
    return best_solution

def _polish_solution(solution, n, m, p, seed=42, max_iters=5000):
    """
    Polish a near-solution by adjusting individual entries to reduce error.
    """
    random.seed(seed)
    
    U, V, W = solution
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    
    rank = len(U[0])
    best_error = _compute_error(U, V, W, n, m, p)
    
    if best_error == 0:
        return U, V, W
    
    # Local search: try changing individual entries
    for iteration in range(max_iters):
        if best_error == 0:
            return U, V, W
        
        # Randomly select a matrix and entry to modify
        mat_choice = random.choice(['U', 'V', 'W'])
        
        if mat_choice == 'U':
            row = random.randint(0, n*m-1)
            col = random.randint(0, rank-1)
            old_val = U[row][col]
            # Try nearby integer values
            for new_val in [old_val - 2, old_val - 1, old_val + 1, old_val + 2]:
                if new_val == old_val:
                    continue
                U[row][col] = new_val
                new_error = _compute_error(U, V, W, n, m, p)
                if new_error < best_error:
                    best_error = new_error
                    break
                else:
                    U[row][col] = old_val
        elif mat_choice == 'V':
            row = random.randint(0, m*p-1)
            col = random.randint(0, rank-1)
            old_val = V[row][col]
            for new_val in [old_val - 2, old_val - 1, old_val + 1, old_val + 2]:
                if new_val == old_val:
                    continue
                V[row][col] = new_val
                new_error = _compute_error(U, V, W, n, m, p)
                if new_error < best_error:
                    best_error = new_error
                    break
                else:
                    V[row][col] = old_val
        else:
            row = random.randint(0, p*n-1)
            col = random.randint(0, rank-1)
            old_val = W[row][col]
            for new_val in [old_val - 2, old_val - 1, old_val + 1, old_val + 2]:
                if new_val == old_val:
                    continue
                W[row][col] = new_val
                new_error = _compute_error(U, V, W, n, m, p)
                if new_error < best_error:
                    best_error = new_error
                    break
                else:
                    W[row][col] = old_val
    
    if best_error == 0:
        return U, V, W
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + SEED_TIME_BUDGET - 15.0  # 15s safety margin
    
    # Start with schoolbook as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # For 3x3x3x5, expect rank less than 45
    # Try ranks from low to high
    for rank in range(20, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random seeds for this rank
        for seed in range(10):
            if time.time() > deadline:
                break
            
            result = _structured_search(rank, n, m, p, seed=seed + 100, max_iters=500)
            if result is not None:
                U, V, W = result
                # Verify the solution
                if _verify(U, V, W, n, m, p):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    break  # Found a decomposition at this rank, try lower
    
    return best_result
# EVOLVE-BLOCK-END
