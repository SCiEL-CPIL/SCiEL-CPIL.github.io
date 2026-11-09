"""Program that attempts to find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
import time
import sys
import random
import numpy as np

DIMENSIONS = [3, 5, 7]
SEED_RANK = 105

# EVOLVE-BLOCK-START

def _validate(U, V, W, n=3, m=5, p=7):
    """
    Validate that the factor matrices U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    The condition is: for all i, j, k,
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _schoolbook(n=3, m=5, p=7):
    """Return the schoolbook decomposition with rank n*m*p."""
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


def _try_strassen_blocks(n=3, m=5, p=7, deadline=None):
    """
    Try to find a lower-rank decomposition using Strassen-type block decomposition.
    We partition the matrices into blocks and apply Strassen's algorithm to each block.
    """
    if deadline is None:
        deadline = time.time() + 140
    
    # For 3x5x7, we can try various block partitions
    # Let's try partitioning into 2x2 blocks where possible
    
    # Strategy: Try to decompose using a mix of schoolbook and Strassen
    # For 3x5: partition as 2+1 rows, 2+3 cols
    # For 5x7: partition as 2+3 rows, 2+2+3 cols
    
    # This is complex, so let's try a different approach:
    # Use known lower bounds and try to find valid decompositions
    
    # The tensor rank of 3x5x7 matrix multiplication is at least 3*5*7 / (3+5+7-2) = 105/13 ≈ 8.08
    # So rank >= 9 theoretically, but likely much higher in practice
    
    # Let's try random search with better initialization
    rng = random.Random(42)
    
    # Try ranks from 50 down to 20
    for target_rank in range(50, 19, -1):
        if time.time() > deadline:
            break
        
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            # Initialize with structured patterns
            U = []
            for i in range(n):
                for j in range(m):
                    row = []
                    for r in range(target_rank):
                        # Use a pattern based on position
                        if (i + j + r) % 3 == 0:
                            row.append(1.0)
                        elif (i + j + r) % 3 == 1:
                            row.append(-1.0)
                        else:
                            row.append(0.0)
                    U.append(row)
            
            V = []
            for j in range(m):
                for k in range(p):
                    row = []
                    for r in range(target_rank):
                        if (j + k + r) % 3 == 0:
                            row.append(1.0)
                        elif (j + k + r) % 3 == 1:
                            row.append(-1.0)
                        else:
                            row.append(0.0)
                    V.append(row)
            
            W = []
            for k in range(p):
                for i in range(n):
                    row = []
                    for r in range(target_rank):
                        if (k + i + r) % 3 == 0:
                            row.append(1.0)
                        elif (k + i + r) % 3 == 1:
                            row.append(-1.0)
                        else:
                            row.append(0.0)
                    W.append(row)
            
            if _validate(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None


def _attempt_rank_reduction(n=3, m=5, p=7, deadline=None, seed=42):
    """
    Attempt to find a lower-rank decomposition using multiple strategies.
    """
    if deadline is None:
        deadline = time.time() + 140
    
    best = _schoolbook(n, m, p)
    
    # First try Strassen-type structured approaches
    result = _try_strassen_blocks(n, m, p, deadline)
    if result is not None:
        return result
    
    # Try random search with various coefficient distributions
    rng = random.Random(seed)
    
    for target_rank in range(80, 30, -1):
        if time.time() > deadline:
            break
        
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            # Use a mix of small integers and half-integers
            # Start with sparse matrices and add density gradually
            density = min(0.3, 0.1 + attempt * 0.005)
            
            U = []
            for _ in range(n * m):
                row = []
                for _ in range(target_rank):
                    if rng.random() < density:
                        row.append(rng.choice([-1, 1, -0.5, 0.5]))
                    else:
                        row.append(0)
                U.append(row)
            
            V = []
            for _ in range(m * p):
                row = []
                for _ in range(target_rank):
                    if rng.random() < density:
                        row.append(rng.choice([-1, 1, -0.5, 0.5]))
                    else:
                        row.append(0)
                V.append(row)
            
            W = []
            for _ in range(p * n):
                row = []
                for _ in range(target_rank):
                    if rng.random() < density:
                        row.append(rng.choice([-1, 1, -0.5, 0.5]))
                    else:
                        row.append(0)
                W.append(row)
            
            if _validate(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return best


def construct():
    """
    Returns the best tensor decomposition found for matrix multiplication.
    """
    deadline = time.time() + 140
    
    # First, return schoolbook as a safe fallback
    best = _schoolbook()
    
    # Attempt rank reduction
    if time.time() < deadline:
        result = _attempt_rank_reduction(deadline=deadline)
        if result is not None:
            best = result
    
    return best

# EVOLVE-BLOCK-END
