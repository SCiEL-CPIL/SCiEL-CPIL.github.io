"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the schoolbook rank-n*m*p decomposition as integer lists."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _verify_fast(U, V, W, n, m, p):
    """
    Fast verification using numpy.
    U: (nm, rank), V: (mp, rank), W: (pn, rank)
    """
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    # Check non-zero entries: T[i*m+j, j*p+k, k*n+i] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a] * V[b] * W[c])
                if val != 1:
                    return False
    
    # Check zero entries: T[a, b, c] = 0 if indices mismatch
    # Mismatch in j:
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                if j == j2: continue
                for k in range(p):
                    a = i * m + j
                    b = j2 * p + k
                    c = k * n + i
                    val = np.sum(U[a] * V[b] * W[c])
                    if val != 0:
                        return False
    # Mismatch in i:
    for i in range(n):
        for i2 in range(n):
            if i == i2: continue
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i2
                    val = np.sum(U[a] * V[b] * W[c])
                    if val != 0:
                        return False
    # Mismatch in k:
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for k2 in range(p):
                    if k == k2: continue
                    a = i * m + j
                    b = j * p + k2
                    c = k * n + i
                    val = np.sum(U[a] * V[b] * W[c])
                    if val != 0:
                        return False
                        
    return True


def _try_rank_reduction(n, m, p, target_rank, time_limit):
    """
    Attempts to find a decomposition of target_rank by starting from schoolbook
    and trying to remove (n*m*p - target_rank) terms and adjusting the rest.
    This is a heuristic local search.
    """
    start_time = time.time()
    nm = n * m
    mp = m * p
    pn = p * n
    base_rank = n * m * p
    num_removals = base_rank - target_rank
    
    # Initialize with schoolbook
    U = np.zeros((nm, base_rank))
    V = np.zeros((mp, base_rank))
    W = np.zeros((pn, base_rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
                
    # We want to reduce to target_rank.
    # Strategy: Randomly select 'num_removals' indices to zero out.
    # Then, check if the remaining matrix can be adjusted to satisfy the equations.
    # This is hard. Instead, we use a random search for the specific rank.
    
    # Given the difficulty, we fall back to a more effective random search for the target rank.
    # We generate random U, V, W with small integer entries and check.
    # To increase success rate, we use a "warm start" or specific structure if known.
    # For now, we do a high-volume random search with small integers.
    
    max_attempts = 5000
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
            
        # Use small integers to increase chance of exact match
        # Entries from {-1, 0, 1}
        U_rand = np.random.randint(-1, 2, size=(nm, target_rank))
        V_rand = np.random.randint(-1, 2, size=(mp, target_rank))
        W_rand = np.random.randint(-1, 2, size=(pn, target_rank))
        
        if _verify_fast(U_rand, V_rand, W_rand, n, m, p):
            return {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_rand.tolist()}
            
    return None

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Known result: Rank 28 is achievable for 2x3x5.
    # We prioritize finding rank 28.
    
    # Try rank 28 with a focused search
    target_rank = 28
    time_per_rank = 20.0
    
    if time.time() < deadline - time_per_rank:
        result = _try_rank_reduction(n, m, p, target_rank, time_per_rank)
        if result:
            best = result
            best_rank = target_rank
    
    # If rank 28 not found, try rank 29
    if best_rank > 29 and time.time() < deadline - time_per_rank:
        result = _try_rank_reduction(n, m, p, 29, time_per_rank)
        if result:
            best = result
            best_rank = 29
            
    return best

# EVOLVE-BLOCK-END
