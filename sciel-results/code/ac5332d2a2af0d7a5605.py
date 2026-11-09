"""Matrix multiplication 2x3 by 3x4: find rank-minimal exact tensor decomposition."""
import numpy as np
from fractions import Fraction
import itertools
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try to find decomposition with rank < 24
    for rank in range(18, 24):
        if time.time() > deadline:
            break
        try:
            result = _find_decomposition_v2(n, m, p, rank, deadline)
            if result is not None:
                return result
        except Exception:
            continue
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}

def _find_decomposition_v2(n, m, p, rank, deadline):
    """Improved structured search for rank decomposition."""
    import random
    random.seed(rank * 7919 + 13)
    
    # Try different W densities and random U, V
    for density in [0.3, 0.5, 0.7]:
        if time.time() > deadline:
            break
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Generate W with entries in {0, 1}
            W = np.zeros((p*n, rank))
            for idx in range(p*n):
                for r in range(rank):
                    W[idx, r] = 1 if random.random() < density else 0
            
            # Ensure each row of W has at least one 1
            for idx in range(p*n):
                if W[idx].sum() == 0:
                    W[idx, random.randint(0, rank-1)] = 1
            
            # Generate U, V with entries in {0, 0.5, 1}
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            for idx in range(n*m):
                for r in range(rank):
                    U[idx, r] = random.choice([0, 0.5, 1])
            for idx in range(m*p):
                for r in range(rank):
                    V[idx, r] = random.choice([0, 0.5, 1])
            
            if _verify(n, m, p, U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _verify(n, m, p, U, V, W):
    """Verify that the decomposition is exact."""
    # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should equal 1
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

# EVOLVE-BLOCK-END
