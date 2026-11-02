import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    For each (i, j, k): sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1
    Uses numpy for speed.
    """
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if s != 1:
                    return False
    return True

def _solve_w_over_q(U, V, n, m, p, r):
    """
    Given integer U (nm x r) and V (mp x r), solve for W (pn x r) over Q.
    For each (k, i), we have m equations:
      sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1  for j=0..m-1
    This is a linear system in variables W[k*n+i, 0..r-1].
    Returns W as a numpy array of integers, or None if no integer solution.
    """
    W = np.zeros((p * n, r), dtype=np.int64)
    
    for k in range(p):
        for i in range(n):
            # Build m x r system over Q
            A = np.zeros((m, r), dtype=object)
            for j in range(m):
                for t in range(r):
                    A[j, t] = Fraction(U[i * m + j, t] * V[j * p + k, t])
            
            b = [Fraction(1)] * m
            
            # Solve A @ x = b over Q using Gaussian Elimination
            # Augmented matrix
            aug = [row[:] + [b[j]] for j, row in enumerate(A)]
            
            # Gaussian elimination
            row = 0
            for col in range(r):
                # Find pivot
                pivot = -1
                for rr in range(row, m):
                    if aug[rr][col] != 0:
                        pivot = rr
                        break
                if pivot == -1:
                    continue
                # Swap
                aug[[row, pivot]] = aug[[pivot, row]]
                # Normalize pivot row
                pivot_val = aug[row][col]
                aug[row] = [x / pivot_val for x in aug[row]]
                # Eliminate
                for rr in range(m):
                    if rr != row and aug[rr][col] != 0:
                        factor = aug[rr][col]
                        aug[rr] = [x - factor * y for x, y in zip(aug[rr], aug[row])]
                row += 1
                if row == m:
                    break
            
            # Check for inconsistency
            for rr in range(row, m):
                if aug[rr][r] != 0:
                    return None  # Inconsistent
            
            # Free variables: set to 0
            x = [Fraction(0)] * r
            for rr in range(row):
                # Find which column is the pivot for this row
                for col in range(r):
                    if aug[rr][col] != 0:
                        x[col] = aug[rr][r]
                        break
            
            # Check if solution is integer
            valid = True
            for val in x:
                if val.denominator != 1:
                    valid = False
                    break
            if not valid:
                return None
            
            W[k * n + i, :] = np.array([val.numerator for val in x], dtype=np.int64)
    
    # Verify over integers
    if not _verify_decomposition(U, V, W, n, m, p):
        return None
    
    return W

def _try_rank_with_search(rank, n, m, p, max_attempts, deadline, seed=42):
    """
    Try to find a valid decomposition of given rank using random search.
    Returns (U, V, W) or None.
    """
    rng = random.Random(seed)
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Generate random sparse patterns with entries in {-1, 0, 1}
        # U: 9 x rank, V: 12 x rank
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        
        # For U: each row has 2-4 nonzeros from {-1, 1}
        for i in range(n * m):
            num_nonzero = rng.randint(2, min(4, rank))
            cols = rng.sample(range(rank), num_nonzero)
            for c in cols:
                U[i, c] = 1 if rng.random() < 0.5 else -1
        
        # For V: each row has 2-4 nonzeros from {-1, 1}
        for i in range(m * p):
            num_nonzero = rng.randint(2, min(4, rank))
            cols = rng.sample(range(rank), num_nonzero)
            for c in cols:
                V[i, c] = 1 if rng.random() < 0.5 else -1
        
        # Solve for W over Q
        W = _solve_w_over_q(U, V, n, m, p, rank)
        if W is not None:
            return U, V, W
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safer margin
    
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
    
    # Try ranks from 23 upward
    for rank in range(23, min(rank_trivial + 1, 30)):
        if time.time() > deadline:
            break
        
        base_attempts = 2000
        num_seeds = 5
        
        for seed_offset in range(num_seeds):
            if time.time() > deadline:
                break
            
            result = _try_rank_with_search(
                rank, n, m, p, 
                base_attempts // num_seeds, 
                deadline, 
                seed=42 + seed_offset * 1000
            )
            
            if result is not None:
                U, V, W = result
                best_result = (rank, U, V, W)
                break
        
        if best_result[0] == rank:
            break
    
    r, U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
