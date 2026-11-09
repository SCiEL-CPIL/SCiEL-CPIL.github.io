"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the multiplication tensor T[i,j,k] = 1 if we're multiplying (i,j) by (j,k)
    # The decomposition requires: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _construct_schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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

def _try_random_search(rank, deadline, seed_offset=0):
    """Try to find a decomposition of given rank using random search."""
    n, m, p = DIMENSIONS
    rows_u = n * m
    rows_v = m * p
    rows_w = p * n
    
    # Use integer entries from {-2, -1, 0, 1, 2} for structured search
    candidates = [-2, -1, 0, 1, 2]
    
    random.seed(42 + seed_offset)
    
    best_result = None
    attempts = 0
    max_attempts = 50000
    
    while attempts < max_attempts:
        if time.time() > deadline:
            break
        attempts += 1
        
        # Generate random U, V, W with sparse integer entries
        U = [[0] * rank for _ in range(rows_u)]
        V = [[0] * rank for _ in range(rows_v)]
        W = [[0] * rank for _ in range(rows_w)]
        
        # Each column should have some non-zero entries
        for c in range(rank):
            # Pick random rows for U
            u_rows = random.sample(range(rows_u), min(3, rows_u))
            for r in u_rows:
                U[r][c] = random.choice(candidates)
            
            v_rows = random.sample(range(rows_v), min(3, rows_v))
            for r in v_rows:
                V[r][c] = random.choice(candidates)
            
            w_rows = random.sample(range(rows_w), min(3, rows_w))
            for r in w_rows:
                W[r][c] = random.choice(candidates)
        
        # Verify
        if _verify(U, V, W, n, m, p):
            best_result = {"U": U, "V": V, "W": W}
            return best_result
    
    return best_result

def _try_strassen_like(deadline, seed_offset=0):
    """Try Strassen-like block decomposition approaches."""
    n, m, p = DIMENSIONS
    # For 3x4x7, we can't directly apply Strassen's 2x2x2, but we can try
    # to find a decomposition by considering the structure
    
    # The tensor has 3*4*7 = 84 non-zero entries (all 1s)
    # We need sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
    
    # Try rank 60-83 (slightly below schoolbook)
    best = None
    for rank in range(70, 84):
        if time.time() > deadline:
            break
        result = _try_random_search(rank, deadline, seed_offset)
        if result is not None:
            best = result
            break
    
    return best

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x4x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 84
    
    # Start with schoolbook as fallback
    best = _construct_schoolbook()
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Strategy 1: Random search for various ranks
    for rank in range(60, schoolbook_rank):
        if time.time() > deadline:
            break
        result = _try_random_search(rank, deadline, seed_offset=rank)
        if result is not None:
            best = result
            best_rank = rank
            break  # Found a valid decomposition, try even lower
    
    # Strategy 2: If we still have time, try structured approaches
    if time.time() < deadline - 5:
        result = _try_strassen_like(deadline, seed_offset=1000)
        if result is not None:
            rank = len(result["U"][0])
            if rank < best_rank:
                best = result
                best_rank = rank
    
    return best
# EVOLVE-BLOCK-END
