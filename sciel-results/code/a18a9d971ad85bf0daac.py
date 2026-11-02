"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    For each (i, j, k): sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1
    """
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for t in range(r):
                    s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                if s != 1:
                    return False
    return True

def _solve_w_over_f2(U, V, n, m, p, r):
    """
    Given binary U (nm x r) and V (mp x r), solve for W (pn x r) over F2.
    For each (k, i), we have m equations:
      sum_t (U[i*m+j, t] * V[j*p+k, t]) * W[k*n+i, t] = 1  for j=0..m-1
    This is a linear system in variables W[k*n+i, 0..r-1] over F2.
    Returns W as a numpy array of 0/1, or None if no solution.
    """
    W = np.zeros((p * n, r), dtype=np.int64)
    
    for k in range(p):
        for i in range(n):
            # Build m x r system over F2
            A = np.zeros((m, r), dtype=np.int64)
            for j in range(m):
                for t in range(r):
                    A[j, t] = (U[i * m + j, t] * V[j * p + k, t]) % 2
            b = np.ones(m, dtype=np.int64)
            
            # Solve A @ x = b over F2
            # Use Gaussian elimination over F2
            aug = np.column_stack([A, b])  # m x (r+1)
            # Gaussian elimination
            row = 0
            for col in range(r):
                # Find pivot
                pivot = -1
                for rr in range(row, m):
                    if aug[rr, col] % 2 == 1:
                        pivot = rr
                        break
                if pivot == -1:
                    continue
                # Swap
                aug[[row, pivot]] = aug[[pivot, row]]
                # Eliminate
                for rr in range(m):
                    if rr != row and aug[rr, col] % 2 == 1:
                        aug[rr] = (aug[rr] + aug[row]) % 2
                row += 1
                if row == m:
                    break
            
            # Check consistency
            for rr in range(row, m):
                if aug[rr, :r].sum() == 0 and aug[rr, r] == 1:
                    return None
            
            # Back-substitute (set free variables to 0)
            x = [0] * r
            for rr in range(row):
                # Find the pivot column for this row
                # Since we eliminated, the pivot is the first 1 in the row
                pivot_col = -1
                for cc in range(r):
                    if aug[rr, cc] % 2 == 1:
                        pivot_col = cc
                        break
                if pivot_col != -1:
                    x[pivot_col] = aug[rr, r] % 2
            
            W[k * n + i, :] = x
    
    # Verify over integers (entries are 0 or 1, so product is 0 or 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for t in range(r):
                    s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                if s != 1:
                    return None

    return W

def _try_rank_f2(target_rank, n, m, p, deadline, seed_offset=0):
    """Try to find a rank decomposition using F2 search."""
    random.seed(42 + seed_offset)
    attempts = 0
    max_attempts = 50000 # Increased attempts for better coverage
    
    while attempts < max_attempts:
        if time.time() > deadline:
            break
        attempts += 1
        
        U = np.zeros((n * m, target_rank), dtype=np.int64)
        V = np.zeros((m * p, target_rank), dtype=np.int64)
        
        # Vary sparsity for diversity
        # Ensure each row has at least 1 non-zero
        for i in range(n * m):
            num_nonzero = random.randint(1, 3)
            cols = random.sample(range(target_rank), num_nonzero)
            for c in cols:
                U[i, c] = 1
        
        for i in range(m * p):
            num_nonzero = random.randint(1, 3)
            cols = random.sample(range(target_rank), num_nonzero)
            for c in cols:
                V[i, c] = 1
        
        W = _solve_w_over_f2(U, V, n, m, p, target_rank)
        if W is not None:
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Trivial decomposition as fallback
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, t] = 1
                V_trivial[j * p + k, t] = 1
                W_trivial[k * n + i, t] = 1
                t += 1
    
    best_result = (rank_trivial, U_trivial, V_trivial, W_trivial)
    
    # Try ranks from 24 upward, with more attempts for lower ranks
    # Theoretical lower bound is 21 (for 3x3x3) + 3 (for 3x3x1) = 24.
    # So 24 is a very strong target.
    for target_rank in [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]:
        if time.time() > deadline:
            break
        
        result = _try_rank_f2(target_rank, n, m, p, deadline, seed_offset=target_rank)
        if result is not None:
            U, V, W = result
            best_result = (target_rank, U, V, W)
            break
    
    r, U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
