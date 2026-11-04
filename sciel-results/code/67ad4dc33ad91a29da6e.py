"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that the tensor decomposition is exact."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m + j], V_arr[j*p + k] * W_arr[k*n + i])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _is_valid_entry(x):
    """Check if x is integer or half-integer."""
    x2 = 2.0 * float(x)
    return abs(x2 - round(x2)) < 1e-9

def _check_all_entries(U, V, W):
    """Check all entries are integer or half-integer."""
    for mat in [U, V, W]:
        for row in mat:
            for x in row:
                if not _is_valid_entry(x):
                    return False
    return True

def _try_random(n, m, p, rank, max_iters=20, seed=42):
    """Try random search for given rank."""
    rng = random.Random(seed)
    values = [-1, -0.5, 0, 0.5, 1]
    
    for it in range(max_iters):
        U = [[rng.choice(values) for _ in range(rank)] for _ in range(n*m)]
        V = [[rng.choice(values) for _ in range(rank)] for _ in range(m*p)]
        W = [[rng.choice(values) for _ in range(rank)] for _ in range(p*n)]
        
        if _verify(U, V, W, n, m, p):
            if _check_all_entries(U, V, W):
                return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Try to find a better decomposition than schoolbook (rank 48)
    for target_rank in range(36, 48):
        if time.time() > deadline - 5:
            break
        result = _try_random(n, m, p, target_rank, max_iters=10, seed=target_rank*1000+42)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
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
