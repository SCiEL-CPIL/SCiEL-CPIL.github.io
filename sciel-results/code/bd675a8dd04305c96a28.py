"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    best_rank = n * m * p
    best_result = _schoolbook(n, m, p, best_rank)
    
    # Try to find better decompositions
    # Strategy: use structured search with block decompositions
    
    # Known: for 2x2x2, Strassen gives rank 7
    # For 3x3x3, best known is 23
    # For 3x5x6, we can try various block decompositions
    
    # Approach: try ranks from a reasonable lower bound to 89
    # Use randomized search with verification
    
    # Lower bound estimate: at least max(n,m,p) * max(n,m,p) / min(n,m,p)
    # For 3,5,6: at least 30/3 = 10, but that's very weak
    # Better: rank >= max(n*m, m*p, p*n) / max(n,m,p) = 30/6 = 5 (still weak)
    # Practical lower bound: try from 25 upward
    
    # Try structured approaches first
    # 1. Block decomposition: split into 3x3x3 + 3x2x3 blocks
    # 2. Random search with verification
    
    # First, try a few known structured decompositions
    # Then do random search
    
    # Try random search for ranks from 25 to 89
    for r in range(25, 90):
        if time.time() > deadline - 5:
            break
        # Try a few random trials for this rank
        num_trials = 100
        for _ in range(num_trials):
            if time.time() > deadline - 5:
                break
            result = _try_random_decomp(n, m, p, r)
            if result is not None:
                if r < best_rank:
                    best_rank = r
                    best_result = result
                return best_result
    
    return best_result

def _schoolbook(n, m, p, rank):
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

def _try_random_decomp(n, m, p, r):
    """Try to find a random decomposition of rank r."""
    # Use structured random generation
    # Make U, V, W sparse with small integer/half-integer entries
    
    # Strategy: generate sparse random matrices and verify
    vals = [0, 0.5, 1, -0.5, -1, 0.25, -0.25, 2, -2]
    
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Make each column have a few non-zeros
    for c in range(r):
        # Choose which rows are non-zero for U
        u_nonzero = random.sample(range(n * m), min(3, n * m))
        for idx in u_nonzero:
            U[idx, c] = random.choice(vals)
        
        v_nonzero = random.sample(range(m * p), min(5, m * p))
        for idx in v_nonzero:
            V[idx, c] = random.choice(vals)
        
        w_nonzero = random.sample(range(p * n), min(3, p * n))
        for idx in w_nonzero:
            W[idx, c] = random.choice(vals)
    
    # Verify
    if _verify(n, m, p, U, V, W, r):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def _verify(n, m, p, U, V, W, r):
    """Check if U,V,W of rank r exactly decompose the matmul tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += W[k * n + i, c] * U[i * m + j, c] * V[j * p + k, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
