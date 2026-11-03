"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=2, m=3, p=7):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _compute_error(U, V, W, n, m, p):
    """Compute max error over all (i,j,k) triples."""
    rank = len(U[0])
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                err = abs(s - 1)
                if err > max_err:
                    max_err = err
    return max_err

def _local_search(U, V, W, n, m, p, rng, max_iters=5000, deadline=None):
    """Local search: randomly change one entry at a time, accept if error decreases."""
    rank = len(U[0])
    current_err = _compute_error(U, V, W, n, m, p)
    
    # Candidate values for entries
    vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        if current_err < 1e-9:
            break
        
        # Pick a random matrix, row, and column
        mat_choice = rng.randint(0, 2)
        if mat_choice == 0:
            mat = U
            nrows = n * m
        elif mat_choice == 1:
            mat = V
            nrows = m * p
        else:
            mat = W
            nrows = p * n
        
        row = rng.randint(0, nrows - 1)
        col = rng.randint(0, rank - 1)
        
        old_val = mat[row][col]
        new_val = rng.choice(vals)
        if new_val == old_val:
            continue
        
        mat[row][col] = new_val
        new_err = _compute_error(U, V, W, n, m, p)
        
        if new_err < current_err:
            current_err = new_err
        else:
            mat[row][col] = old_val
    
    return current_err

def _try_rank(rank, n=2, m=3, p=7, seed=42, deadline=None):
    """Try to find a rank-rank decomposition. Returns (U, V, W) or None."""
    rng = random.Random(seed)
    
    # Try multiple random initializations
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    vals = [-1, -0.5, 0, 0.5, 1]
    
    for attempt in range(20):
        if deadline and time.time() > deadline:
            break
        
        # Random sparse initialization
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for r in range(rank):
            # Each rank-1 term should be sparse
            # Randomly choose which entries are non-zero
            for i in range(n * m):
                if rng.random() < 0.3:
                    U[i][r] = rng.choice(vals)
            for j in range(m * p):
                if rng.random() < 0.3:
                    V[j][r] = rng.choice(vals)
            for k in range(p * n):
                if rng.random() < 0.3:
                    W[k][r] = rng.choice(vals)
        
        # Local search
        err = _local_search(U, V, W, n, m, p, rng, max_iters=2000, deadline=deadline)
        
        if err < best_err:
            best_err = err
            best_U = [row[:] for row in U]
            best_V = [row[:] for row in V]
            best_W = [row[:] for row in W]
        
        if best_err < 1e-9:
            break
    
    if best_err < 1e-9:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Try ranks from low to high
    # Lower bound: at least max(n*m, m*p, p*n) = 21
    # Upper bound: n*m*p = 42
    
    best = None
    for rank in range(max(n*m, m*p, p*n), n*m*p + 1):
        if time.time() > deadline:
            break
        result = _try_rank(rank, n, m, p, seed=42 + rank, deadline=deadline)
        if result is not None:
            best = result
            break
    
    if best is None:
        best = _schoolbook()
    
    return best
# EVOLVE-BLOCK-END
