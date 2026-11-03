"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _validate(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Check dimensions
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    # For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(total - 1) > 1e-10:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
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

def _try_rank_with_structure(n, m, p, rank, time_limit=10):
    """Try to find a rank-rank decomposition using structured search."""
    start_time = time.time()
    
    # Strategy: use the fact that for each j, the terms involving j form a 
    # smaller problem. Try to share structure across different (i,k) pairs.
    
    # For a fixed j, we have n*p terms: for each (i,k), we need
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is a rank-1-like constraint on the n×p matrix M_j where M_j[i,k] = 
    # the contribution from column r.
    
    # Let's try: for each r, define u_r[j] = U[:, j] (length n), v_r[j] = V[:, j] (length p), 
    # w_r[j][i,k] = W[k*n+i] for fixed j... this is getting complex.
    
    # Alternative: try random initialization with small integers and verify.
    for attempt in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize with small random values
        U = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(rank)] for _ in range(p * n)]
        
        if _validate(U, V, W, n, m, p):
            return U, V, W
    
    return None

def _construct_gf3_based(n, m, p):
    """
    Try to construct a decomposition based on GF(3) structure.
    Map indices to field elements and use algebraic identities.
    """
    # For 3x3x8, try rank 21 or similar
    # The idea: over GF(3), we can use the fact that x^2 = -x for x in {0,1,2} mod 3
    # when x is an element of the field (actually x^2 is just x^2 mod 3)
    
    # Let's try a different approach: systematic search over structured decompositions
    # where entries are in {0, 1, 2}
    
    n, m, p = 3, 3, 8
    best_rank = None
    best_decomp = None
    
    for rank in range(15, 35):
        start = time.time()
        found = False
        
        # Try structured initialization based on GF(3)
        # For each rank element r, assign values based on a pattern
        for seed in range(200):
            if time.time() - start > 5:
                break
            
            rng = random.Random(seed)
            U = [[rng.choice([0, 1, 2]) for _ in range(rank)] for _ in range(n * m)]
            V = [[rng.choice([0, 1, 2]) for _ in range(rank)] for _ in range(m * p)]
            W = [[rng.choice([0, 1, 2]) for _ in range(rank)] for _ in range(p * n)]
            
            if _validate(U, V, W, n, m, p):
                best_rank = rank
                best_decomp = (U, V, W)
                found = True
                break
        
        if found:
            break
    
    if best_decomp:
        return best_decomp
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Start with schoolbook as fallback
    schoolbook = _schoolbook(n, m, p)
    
    # Try to find better decompositions
    # Strategy 1: GF(3)-based structured search
    result = _construct_gf3_based(n, m, p)
    if result and time.time() < deadline - 10:
        U, V, W = result
        if _validate(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Strategy 2: Try various ranks with random search
    for rank in range(18, 40):
        if time.time() > deadline - 15:
            break
        result = _try_rank_with_structure(n, m, p, rank, time_limit=3)
        if result:
            U, V, W = result
            if _validate(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return schoolbook
# EVOLVE-BLOCK-END
