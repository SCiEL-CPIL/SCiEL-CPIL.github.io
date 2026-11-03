"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import random
import sys
import numpy as np

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Verify using numpy for speed."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64).reshape(n, m, rank)
    V_arr = np.array(V, dtype=np.float64).reshape(m, p, rank)
    W_arr = np.array(W, dtype=np.float64).reshape(p, n, rank)
    
    # Compute sum over rank: U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Use einsum
    reconstructed = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    expected = np.ones((n, m, p), dtype=np.float64)
    return np.allclose(expected, reconstructed, atol=tol)

def _try_random_search(n, m, p, target_rank, time_budget, rng):
    """Try to find a decomposition of rank target_rank using random search."""
    deadline = time.time() + time_budget
    n_rows_u = n * m
    n_rows_v = m * p
    n_rows_w = p * n
    
    # Build the constraint matrix: for each (i,j,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is a system of quadratic equations. We'll use a heuristic approach.
    
    # Strategy: initialize with random values and use gradient-free optimization
    # or try structured patterns.
    
    # For small ranks, try brute-force style with structured initialization
    for attempt in range(1000):
        if time.time() > deadline:
            break
        
        # Initialize with random half-integer values
        U = np.zeros((n_rows_u, target_rank), dtype=np.float64)
        V = np.zeros((n_rows_v, target_rank), dtype=np.float64)
        W = np.zeros((n_rows_w, target_rank), dtype=np.float64)
        
        # Random initialization with values from {-1, -0.5, 0, 0.5, 1}
        choices = [-1, -0.5, 0, 0.5, 1]
        for c in range(target_rank):
            U[:, c] = [rng.choice(choices) for _ in range(n_rows_u)]
            V[:, c] = [rng.choice(choices) for _ in range(n_rows_v)]
            W[:, c] = [rng.choice(choices) for _ in range(n_rows_w)]
        
        # Verify
        if _verify_decomposition_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    schoolbook_rank = n * m * p
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    # Verify schoolbook
    if not _verify_decomposition_fast(U_school, V_school, W_school, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    best_rank = schoolbook_rank
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find better decompositions
    # Known lower bounds and known results:
    # For 4x4 * 4x5, the tensor rank is at most 80 (schoolbook)
    # Try ranks below 80
    
    rng = random.Random(42)
    
    # Try structured approaches first
    # Approach 1: Try rank 72, 70, 68, etc.
    for target_rank in [72, 70, 68, 66, 64, 60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 10, 8, 6, 5, 4, 3, 2]:
        if time.time() > deadline:
            break
        
        time_per_attempt = min(2.0, max(0.1, (deadline - time.time()) / 100))
        result = _try_random_search(n, m, p, target_rank, time_per_attempt, rng)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
    
    # If no improvement found, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
