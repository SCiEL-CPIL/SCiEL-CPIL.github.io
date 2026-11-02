"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact.
    For all i in [0,n), j in [0,m), k in [0,p):
      sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1
    """
    rank = len(U[0])
    # Convert to numpy for speed
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m + j, r] * V_arr[j*p + k, r] * W_arr[k*n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition (rank = n*m*p)."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _random_search(n, m, p, max_rank, deadline, seed=42):
    """
    Random search for a valid decomposition with rank <= max_rank.
    Tries random integer/half-integer entries.
    """
    rng = random.Random(seed)
    best = None
    best_rank = max_rank + 1
    
    # Try decreasing rank from max_rank down to some minimum
    for target_rank in range(max_rank, max(50, max_rank - 30), -1):
        if time.time() > deadline:
            break
        
        # Try several random attempts per rank
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Generate random sparse matrices with small integer entries
            # Use a mix of 0, 1, -1, 0.5, -0.5
            values = [0, 1, -1, 0.5, -0.5]
            
            U = [[rng.choice(values) if rng.random() < 0.3 else 0 for _ in range(target_rank)] for _ in range(n * m)]
            V = [[rng.choice(values) if rng.random() < 0.3 else 0 for _ in range(target_rank)] for _ in range(m * p)]
            W = [[rng.choice(values) if rng.random() < 0.3 else 0 for _ in range(target_rank)] for _ in range(p * n)]
            
            if _verify(U, V, W, n, m, p):
                best = (U, V, W, target_rank)
                best_rank = target_rank
                break
        
        if best is not None:
            break
    
    return best

def _structured_search(n, m, p, deadline):
    """
    Try structured patterns that might reduce rank.
    For example, block-based or Kronecker product structures.
    """
    results = []
    
    # Try block diagonal / Kronecker-like structures
    # For 4x5 x 5x5, we can think of it as blocks
    
    # One approach: use the fact that 4 = 2*2 and 5 is prime
    # Try rank based on factorizations
    
    # Schoolbook is 100. Let's see if we can do better with structured patterns.
    
    # Try a pattern where we group columns
    # For each (i, j), instead of having separate terms for each k,
    # maybe we can share structure across k values.
    
    # Actually, for the standard multiplication tensor, the rank is known to be
    # at least n*m*p / something, but for these small dimensions it's likely close to n*m*p.
    
    # Let's try a simple improvement: check if any two schoolbook terms can be merged.
    # Two terms (i1,j1,k1) and (i2,j2,k2) can potentially be merged if...
    # This is complex, so let's rely on random search.
    
    return results

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    of the tensor decomposition for matrix multiplication of size 4x5 by 5x5.
    The rank is the number of columns in U, V, and W.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin for the 150s limit
    
    # Start with schoolbook as the baseline
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = n * m * p
    
    # Try to find a lower-rank decomposition
    # First, try structured search
    # Then, try random search with decreasing rank
    
    # Try random search starting from rank 99 down
    search_result = _random_search(n, m, p, best_rank - 1, deadline, seed=42)
    
    if search_result is not None:
        U_found, V_found, W_found, rank_found = search_result
        best_result = {"U": U_found, "V": V_found, "W": W_found}
        best_rank = rank_found
    
    # Try more random searches with different seeds if time permits
    if time.time() < deadline - 10:
        for seed in [123, 456, 789]:
            if time.time() > deadline:
                break
            sr = _random_search(n, m, p, best_rank - 1, deadline, seed=seed)
            if sr is not None:
                U_f, V_f, W_f, r_f = sr
                if r_f < best_rank:
                    best_result = {"U": U_f, "V": V_f, "W": W_f}
                    best_rank = r_f
    
    return best_result
# EVOLVE-BLOCK-END
