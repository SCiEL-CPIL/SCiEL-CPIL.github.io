"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    # Build the tensor
    T = np.zeros((n*m, m*p, p*n), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Compute decomposition
    U_arr = np.array(U, dtype=object)
    V_arr = np.array(V, dtype=object)
    W_arr = np.array(W, dtype=object)
    
    for i in range(n*m):
        for j in range(m*p):
            for k in range(p*n):
                val = 0
                for c in range(rank):
                    val += U_arr[i, c] * V_arr[j, c] * W_arr[k, c]
                if T[i, j, k] != 0 and val != 1:
                    return False
                if T[i, j, k] == 0 and val != 0:
                    return False
    return True

def _construct_schoolbook():
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

def _structured_search(rank, time_limit, seed=42):
    """Structured search for low-rank decompositions using algebraic patterns."""
    n, m, p = DIMENSIONS
    rng = random.Random(seed)
    
    # Try different structural patterns
    patterns = ['diagonal', 'block', 'strassen_like']
    
    for pattern in patterns:
        for attempt in range(50):
            if time.time() > time_limit:
                return None
            
            if pattern == 'diagonal':
                # Diagonal-like structure with sparse non-zero entries
                U = [[0] * rank for _ in range(n * m)]
                V = [[0] * rank for _ in range(m * p)]
                W = [[0] * rank for _ in range(p * n)]
                
                # Assign random sparse patterns
                for c in range(rank):
                    # Choose which indices get non-zero values in this column
                    if rng.random() < 0.3:
                        i = rng.randint(0, n-1)
                        j = rng.randint(0, m-1)
                        k = rng.randint(0, p-1)
                        U[i*m+j][c] = 1
                        V[j*p+k][c] = 1
                        W[k*n+i][c] = 1
                    else:
                        # More complex pattern
                        i1 = rng.randint(0, n-1)
                        i2 = rng.randint(0, n-1)
                        j1 = rng.randint(0, m-1)
                        j2 = rng.randint(0, m-1)
                        k1 = rng.randint(0, p-1)
                        k2 = rng.randint(0, p-1)
                        
                        if i1 != i2 or j1 != j2:
                            U[i1*m+j1][c] = 1
                            V[j1*p+k1][c] = 1
                            W[k1*n+i1][c] = 1
                            U[i2*m+j2][c] = -1
                            V[j2*p+k2][c] = 1
                            W[k2*n+i2][c] = 1
            
            elif pattern == 'block':
                # Block structure
                U = [[0] * rank for _ in range(n * m)]
                V = [[0] * rank for _ in range(m * p)]
                W = [[0] * rank for _ in range(p * n)]
                
                for c in range(rank):
                    block_i = rng.randint(0, n-1)
                    block_j = rng.randint(0, m-1)
                    block_k = rng.randint(0, p-1)
                    
                    # Fill a block
                    for di in range(min(2, n - block_i)):
                        for dj in range(min(2, m - block_j)):
                            for dk in range(min(2, p - block_k)):
                                ii = block_i + di
                                jj = block_j + dj
                                kk = block_k + dk
                                sign = 1 if (di + dj + dk) % 2 == 0 else -1
                                U[ii*m+jj][c] += sign
                                V[jj*p+kk][c] += sign
                                W[kk*n+ii][c] += sign
            
            else:  # strassen_like
                # Strassen-like for smaller blocks
                U = [[0] * rank for _ in range(n * m)]
                V = [[0] * rank for _ in range(m * p)]
                W = [[0] * rank for _ in range(p * n)]
                
                for c in range(rank):
                    # Random combination
                    i = rng.randint(0, n-1)
                    j = rng.randint(0, m-1)
                    k = rng.randint(0, p-1)
                    
                    val = rng.choice([1, -1, 0.5, -0.5])
                    U[i*m+j][c] = val
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
            
            # Verify
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None

def _hybrid_search(max_rank, time_limit, seed=42):
    """Combine structured and random search."""
    n, m, p = DIMENSIONS
    rng = random.Random(seed)
    
    for rank in range(50, max_rank + 1):
        if time.time() > time_limit:
            break
        
        # Try structured search first
        result = _structured_search(rank, time_limit, seed=seed + rank)
        if result is not None:
            return result
        
        # Then try random search with limited attempts
        vals = [0, 0.5, 1, -0.5, -1]
        for attempt in range(20):
            if time.time() > time_limit:
                break
            
            U = [[rng.choice(vals) for _ in range(rank)] for _ in range(n * m)]
            V = [[rng.choice(vals) for _ in range(rank)] for _ in range(m * p)]
            W = [[rng.choice(vals) for _ in range(rank)] for _ in range(p * n)]
            
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Start with schoolbook as baseline
    best_result = _construct_schoolbook()
    best_rank = 3 * 5 * 7
    
    # Try to find better decompositions
    for target_rank in [60, 70, 80, 90, 100, 104]:
        if time.time() > deadline:
            break
        
        if target_rank >= best_rank:
            continue
        
        result = _hybrid_search(target_rank, deadline, seed=random.randint(0, 10000))
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
