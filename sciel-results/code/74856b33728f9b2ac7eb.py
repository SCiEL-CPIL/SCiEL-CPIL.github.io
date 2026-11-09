import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

def _verify(U, V, W, tol=1e-10):
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    # Compute tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Compare with expected: 1 if k < 6 else 0 (for all i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - expected) > tol:
                    return False
    return True

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

def construct():
    start_time = time.time()
    
    # First, return schoolbook as baseline
    best = _schoolbook()
    
    # Try to find a lower-rank decomposition using random search
    # The key insight: for matrix multiplication, the tensor is "multilinear"
    # We can try to find decompositions with half-integer entries
    
    n, m, p = DIMENSIONS
    um = n * m  # 9
    vm = m * p  # 18
    wm = p * n  # 18
    
    # Try ranks from 53 down to, say, 30
    # Use random search with structured initialization
    
    for target_rank in range(53, 29, -1):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        # Try a few random searches for this rank
        for attempt in range(5):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            
            # Initialize with random values from {0, 0.5, 1, 1.5, 2}
            U = np.random.choice([0, 0.5, 1, 1.5, 2], size=(um, target_rank))
            V = np.random.choice([0, 0.5, 1, 1.5, 2], size=(vm, target_rank))
            W = np.random.choice([0, 0.5, 1, 1.5, 2], size=(wm, target_rank))
            
            # Verify
            if _verify(U, V, W):
                # Convert to lists
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    return best
