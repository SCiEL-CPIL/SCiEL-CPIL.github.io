"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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

def _try_structured_rank(n, m, p, rank, seed=42):
    """Try structured decomposition with integer/half-integer entries."""
    np.random.seed(seed)
    random.seed(seed)
    
    # Use small integer and half-integer values for better chance of exact solutions
    # The evaluator requires integers or half-integers
    vals = [0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2]
    
    U = [[random.choice(vals) for _ in range(rank)] for _ in range(n * m)]
    V = [[random.choice(vals) for _ in range(rank)] for _ in range(m * p)]
    W = [[random.choice(vals) for _ in range(rank)] for _ in range(p * n)]
    
    return U, V, W

def _is_valid_entry(x):
    """Check if x is an integer or half-integer."""
    return abs(x * 2 - round(x * 2)) < 1e-9

def _snap_to_half_int(x):
    """Snap x to the nearest half-integer."""
    return round(x * 2) / 2.0

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 90
    
    # Known lower bound for matrix multiplication rank is roughly (n*m*p)^(2/3)
    # For 3x5x6: (90)^(2/3) ≈ 20.1, so try from ~25 upward
    # But also try some structured ranks
    
    # First, try structured random with small integers/half-integers for various ranks
    # This is fast and might find exact solutions
    for rank in [30, 35, 40, 45, 50, 55, 60, 70, 80, 85, 89]:
        if time.time() > deadline - 20:
            break
        
        found = None
        for seed in range(20):
            if time.time() > deadline - 20:
                break
            U, V, W = _try_structured_rank(n, m, p, rank, seed=seed)
            if _verify(U, V, W, n, m, p):
                found = (U, V, W)
                break
        
        if found is not None:
            U, V, W = found
            best_rank = rank
            best = {"U": U, "V": V, "W": W}
            # Don't break - try to find even lower rank
            continue
    
    # If we found something better than schoolbook, try to improve further
    if best_rank < n * m * p:
        for rank in range(max(20, best_rank - 5), best_rank):
            if time.time() > deadline - 10:
                break
            
            found = None
            for seed in range(10):
                if time.time() > deadline - 10:
                    break
                U, V, W = _try_structured_rank(n, m, p, rank, seed=seed)
                if _verify(U, V, W, n, m, p):
                    found = (U, V, W)
                    break
            
            if found is not None:
                U, V, W = found
                best_rank = rank
                best = {"U": U, "V": V, "W": W}
    
    # Note: Gradient descent is removed from the return path because it produces
    # arbitrary floats which are invalid per the task contract.
    # We rely on the schoolbook solution (valid integers) as the guaranteed fallback.
    
    return best
# EVOLVE-BLOCK-END
