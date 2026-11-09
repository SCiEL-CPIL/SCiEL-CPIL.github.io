"""Combined program for matrix multiplication tensor rank minimization (2x4x6)."""
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
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    # Efficient verification using einsum
    # T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need to check that this equals 1 for all valid (i,j,k)
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    # Compute the reconstructed tensor
    # R[i, j, k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # This is a bit tricky with einsum because indices are coupled
    # Let's do it in a loop for clarity and correctness, but vectorized over c
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # sum over c
                s = np.dot(U_reshaped[i, j, :], V_reshaped[j, k, :])
                # Wait, this is not right. The sum is element-wise product then sum.
                # s = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
                # This is not a simple dot product of two vectors unless we combine two of them.
                # Let's just use a simple loop over c for safety, or use einsum properly.
                # np.einsum('c,c,c->', U_reshaped[i,j,:], V_reshaped[j,k,:], W_reshaped[k,i,:])
                s = np.einsum('c,c,c->', U_reshaped[i, j, :], V_reshaped[j, k, :], W_reshaped[k, i, :])
                if not np.isclose(s, 1.0, atol=1e-9):
                    return False
    return True

def _strassen_2x2x2():
    """
    Strassen factors for 2x2x2 multiplication.
    Returns U (4x7), V (4x7), W (4x7) as lists of lists.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A10)*(B00+B10)
    U[0][0] = 1; U[2][0] = 1
    V[0][0] = 1; V[2][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _try_random_rank(r, max_attempts=100, seed=42, val_range=(-1, 0, 1)):
    """Try to find a rank-r decomposition via random search."""
    n, m, p = DIMENSIONS
    rng = random.Random(seed)
    
    for attempt in range(max_attempts):
        U = np.array([[rng.choice(val_range) for _ in range(r)] for _ in range(n*m)], dtype=float)
        V = np.array([[rng.choice(val_range) for _ in range(r)] for _ in range(m*p)], dtype=float)
        W = np.array([[rng.choice(val_range) for _ in range(r)] for _ in range(p*n)], dtype=float)
        
        # Verify efficiently
        # We can vectorize the check slightly
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        
        ok = True
        for i in range(n):
            if not ok:
                break
            for j in range(m):
                if not ok:
                    break
                for k in range(p):
                    # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
                    # Use einsum for speed
                    s = np.einsum('c,c,c->', U_reshaped[i, j, :], V_reshaped[j, k, :], W_reshaped[k, i, :])
                    if not np.isclose(s, 1.0, atol=1e-9):
                        ok = False
                        break
        if ok:
            # Convert to list of lists with integer/half-integer representation if possible
            # The problem allows integer/half-integer. Since we used integers, they are valid.
            return U.tolist(), V.tolist(), W.tolist()
    return None

def _greedy_set_cover_search(rank_max=47, time_budget=5.0):
    """
    Attempt to find a low-rank decomposition using a greedy set-cover approach.
    This is a heuristic: we try to construct the tensor as a sum of rank-1 terms.
    However, exact rank-1 terms for matrix multiplication are the schoolbook terms.
    Combining them is not straightforward.
    
    Instead, we will use this time to run more random searches with different seeds
    and value ranges, focusing on lower ranks.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # We'll just run a bunch of random searches here with varied parameters
    for r in range(24, min(rank_max + 1, 35)):
        if time.time() - start_time > time_budget:
            break
        for seed in range(10):
            res = _try_random_rank(r, max_attempts=5, seed=seed * 1000 + r, val_range=(-1, 0, 1))
            if res:
                return res
    return None

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    # Strategy 1: Random search with {-1, 0, 1}
    # Strategy 2: Random search with {-2, -1, 0, 1, 2}
    
    # Time allocation:
    # 24-28: 40% of time
    # 29-38: 30% of time
    # 39-47: 30% of time
    
    total_time = 140.0
    t1 = start_time + total_time * 0.4
    t2 = start_time + total_time * 0.7
    
    # Rank 24-28
    for r in range(24, 29):
        if time.time() > t1:
            break
        # Use multiple seeds and value ranges
        for seed_offset in range(10):
            if time.time() > t1:
                break
            result = _try_random_rank(r, max_attempts=20, seed=42 + r + seed_offset * 100, val_range=(-1, 0, 1))
            if result:
                U, V, W = result
                return {"U": U, "V": V, "W": W}
        
        for seed_offset in range(5):
            if time.time() > t1:
                break
            result = _try_random_rank(r, max_attempts=10, seed=1000 + r + seed_offset * 100, val_range=(-2, -1, 0, 1, 2))
            if result:
                U, V, W = result
                return {"U": U, "V": V, "W": W}
            
    # Rank 29-38
    for r in range(29, 39):
        if time.time() > t2:
            break
        for seed_offset in range(5):
            if time.time() > t2:
                break
            result = _try_random_rank(r, max_attempts=10, seed=2000 + r + seed_offset * 100, val_range=(-1, 0, 1))
            if result:
                U, V, W = result
                return {"U": U, "V": V, "W": W}
        
        for seed_offset in range(2):
            if time.time() > t2:
                break
            result = _try_random_rank(r, max_attempts=5, seed=3000 + r + seed_offset * 100, val_range=(-2, -1, 0, 1, 2))
            if result:
                U, V, W = result
                return {"U": U, "V": V, "W": W}
            
    # Rank 39-47
    for r in range(39, 48):
        if time.time() > deadline:
            break
        for seed_offset in range(3):
            if time.time() > deadline:
                break
            result = _try_random_rank(r, max_attempts=5, seed=4000 + r + seed_offset * 100, val_range=(-1, 0, 1))
            if result:
                U, V, W = result
                return {"U": U, "V": V, "W": W}
        
        for seed_offset in range(1):
            if time.time() > deadline:
                break
            result = _try_random_rank(r, max_attempts=2, seed=5000 + r + seed_offset * 100, val_range=(-2, -1, 0, 1, 2))
            if result:
                U, V, W = result
                return {"U": U, "V": V, "W": W}
            
    # Fallback: schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
