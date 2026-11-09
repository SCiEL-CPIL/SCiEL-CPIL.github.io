"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def _schoolbook():
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

def _verify(U, V, W):
    """Check that U,V,W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Build the multiplication tensor T[i,j,k] for (i,j) in A, (j,k) in B -> C[i,k]
    # The decomposition says: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # (since C[i,k] = sum_j A[i,j]*B[j,k], the tensor entry for (i,j,k) is 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _try_random_rank(r, max_attempts=200, seed=42):
    """Try to find a rank-r decomposition via random search."""
    n, m, p = DIMENSIONS
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    for attempt in range(max_attempts):
        # Random integer entries in {-1, 0, 1}
        U = np.array([[rng.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n*m)], dtype=int)
        V = np.array([[rng.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m*p)], dtype=int)
        W = np.array([[rng.choice([-1, 0, 1]) for _ in range(r)] for _ in range(p*n)], dtype=int)
        
        # Verify
        ok = True
        for i in range(n):
            if not ok:
                break
            for j in range(m):
                if not ok:
                    break
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if s != 1:
                        ok = False
                        break
        if ok:
            return U.tolist(), V.tolist(), W.tolist()
    return None

def _try_small_rank(r, deadline):
    """Try rank r with various strategies before deadline."""
    n, m, p = DIMENSIONS
    best_result = None
    
    # Strategy 1: pure random with {-1,0,1}
    for seed in range(50):
        if time.time() > deadline:
            break
        result = _try_random_rank(r, max_attempts=20, seed=seed)
        if result:
            return result
    
    # Strategy 2: random with {-2,-1,0,1,2}
    rng = random.Random(1000)
    for attempt in range(100):
        if time.time() > deadline:
            break
        U = np.array([[rng.choice([-2,-1,0,1,2]) for _ in range(r)] for _ in range(n*m)], dtype=int)
        V = np.array([[rng.choice([-2,-1,0,1,2]) for _ in range(r)] for _ in range(m*p)], dtype=int)
        W = np.array([[rng.choice([-2,-1,0,1,2]) for _ in range(r)] for _ in range(p*n)], dtype=int)
        
        ok = True
        for i in range(n):
            if not ok:
                break
            for j in range(m):
                if not ok:
                    break
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if s != 1:
                        ok = False
                        break
        if ok:
            return U.tolist(), V.tolist(), W.tolist()
    
    return best_result

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    # Try ranks from known lower bounds up to schoolbook
    # Known: for 2x4 x 4x6, the rank is at least max(n*m, m*p, p*n) = max(8, 24, 12) = 24
    # Strassen-type improvements may help
    # Try a range of ranks
    for r in range(24, schoolbook_rank):
        if time.time() > deadline - 5.0:
            break
        result = _try_small_rank(r, deadline - 2.0)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
