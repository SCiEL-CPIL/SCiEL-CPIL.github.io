"""Combined program for matrix multiplication 2x3x8 tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _check_exact(U, V, W, n, m, p):
    """Check if decomposition is exact using Fraction arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _check_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast float-based check."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _try_structured_rank(r, n, m, p, deadline, rng):
    """
    Try to construct a decomposition of rank r.
    Uses structured initialization based on the tensor structure.
    """
    if r < max(n*m, m*p, p*n):
        return None
    
    # For M_{2,3,8}, we have n=2, m=3, p=8
    # U is 6 x r, V is 24 x r, W is 16 x r
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
    
    # Strategy: Try various structured patterns
    # Pattern 1: Sparse with small integer/half-integer entries
    # Each column c contributes to specific (i,j,k) triples
    
    # We'll use a local search approach:
    # 1. Initialize with a valid decomposition (e.g., schoolbook truncated or random)
    # 2. Try to merge/eliminate columns
    
    # Actually, let's try a more direct approach for specific ranks
    
    # For rank 40, 42, etc., we can try to find patterns
    # where some columns cover multiple (i,j,k) entries
    
    # Let's try random sparse initialization with local repair
    for attempt in range(50):
        if time.time() > deadline:
            return None
        
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        
        # Initialize each column with a random "pattern"
        # A column c has nonzeros at specific positions
        # For the product U[i,j,c]*V[j,k,c]*W[k,i,c] to be nonzero,
        # all three must be nonzero
        
        for c in range(r):
            # Choose a random structure for this column
            # Option A: Single (i,j,k) - schoolbook-like
            # Option B: Multiple entries via shared j or k
            
            mode = rng.random()
            
            if mode < 0.5:
                # Single term: one (i,j,k)
                i = rng.randint(0, n-1)
                j = rng.randint(0, m-1)
                k = rng.randint(0, p-1)
                val = rng.choice([1, -1, 2, -2, 0.5, -0.5])
                U[i*m+j][c] = val
                V[j*p+k][c] = val
                W[k*n+i][c] = 1.0
            elif mode < 0.8:
                # Two terms sharing j: (i1,j,k1) and (i2,j,k2)
                j = rng.randint(0, m-1)
                i1 = rng.randint(0, n-1)
                i2 = rng.randint(0, n-1)
                k1 = rng.randint(0, p-1)
                k2 = rng.randint(0, p-1)
                if i1 == i2 and k1 == k2:
                    continue
                val = rng.choice([1, -1])
                U[i1*m+j][c] = val
                V[j*p+k1][c] = val
                W[k1*n+i1][c] = 1.0
                U[i2*m+j][c] = val
                V[j*p+k2][c] = val
                W[k2*n+i2][c] = 1.0
            else:
                # Two terms sharing k: (i1,j1,k) and (i2,j2,k)
                k = rng.randint(0, p-1)
                i1 = rng.randint(0, n-1)
                i2 = rng.randint(0, n-1)
                j1 = rng.randint(0, m-1)
                j2 = rng.randint(0, m-1)
                if i1 == i2 and j1 == j2:
                    continue
                val = rng.choice([1, -1])
                U[i1*m+j1][c] = val
                V[j1*p+k][c] = val
                W[k*n+i1][c] = 1.0
                U[i2*m+j2][c] = val
                V[j2*p+k][c] = val
                W[k*n+i2][c] = 1.0
        
        # Check if this works
        if _check_fast(U, V, W, n, m, p):
            if _check_exact(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None

def _local_search_merge(n, m, p, U, V, W, deadline, rng, max_iters=200):
    """
    Try to reduce rank by merging or eliminating columns.
    Start with a valid decomposition and try to find lower rank.
    """
    rank = len(U[0])
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try to eliminate a column by redistributing its contribution
        # Pick a random column c
        c = rng.randint(0, rank-1)
        
        # Find which (i,j,k) entries this column contributes to
        contributing = set()
        for i in range(n):
            for j in range(m):
                if U[i*m+j][c] != 0:
                    for k in range(p):
                        if V[j*p+k][c] != 0 and W[k*n+i][c] != 0:
                            contributing.add((i,j,k))
        
        if not contributing:
            # Column is zero, just remove it
            U = [row[:c] + row[c+1:] for row in U]
            V = [row[:c] + row[c+1:] for row in V]
            W = [row[:c] + row[c+1:] for row in W]
            rank -= 1
            continue
        
        # Try to absorb this column's contribution into other columns
        # This is complex; skip for now and try simple elimination
        pass
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 48
    
    rng = random.Random(42)
    
    # Lower bound
    lower_bound = max(n*m, m*p, p*n)  # 24
    
    # Try candidate ranks from lower_bound upward
    # Focus on ranks that are plausible improvements
    candidate_ranks = list(range(lower_bound, best_rank))
    
    # Prioritize: try a subset first, then expand if time permits
    # Known results suggest rank might be around 40-44 for this size
    priority_ranks = [40, 42, 44, 46, 38, 41, 43, 45, 36, 37, 39, 47, 34, 35, 32, 33, 30, 31, 28, 29, 26, 27, 24, 25]
    priority_ranks = [r for r in priority_ranks if lower_bound <= r < best_rank]
    remaining = [r for r in candidate_ranks if r not in priority_ranks]
    search_order = priority_ranks + remaining
    
    for target_rank in search_order:
        if time.time() > deadline:
            break
        
        # Try structured construction
        result = _try_structured_rank(target_rank, n, m, p, deadline, rng)
        if result is not None:
            best = result
            best_rank = target_rank
            # If we found a good one, try to improve further
            continue
        
        # If we've spent too long, break
        if time.time() > deadline - 10:
            break
    
    # Final verification
    if _check_fast(best["U"], best["V"], best["W"], n, m, p):
        if _check_exact(best["U"], best["V"], best["W"], n, m, p):
            return best
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
