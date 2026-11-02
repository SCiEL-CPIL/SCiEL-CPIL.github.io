"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify_full(U, V, W, n, m, p):
    """Full verification of tensor decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(np.dot(U_arr[a], V_arr[b]), W_arr[c])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                if i == i2 and j == j2 and k == k2:
                    continue
                val = np.dot(np.dot(U_arr[a], V_arr[b]), W_arr[c])
                if abs(val) > 1e-9:
                    return False
    return True

def _verify_sampled(U, V, W, n, m, p, num_samples=1000):
    """Sampled verification for speed."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(np.dot(U_arr[a], V_arr[b]), W_arr[c])
                if abs(val - 1.0) > 1e-6:
                    return False
    
    random.seed(98765)
    for _ in range(num_samples):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        c = random.randint(0, p*n-1)
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = c // n
        i2 = c % n
        if i == i2 and j == j2 and k == k2:
            continue
        val = np.dot(np.dot(U_arr[a], V_arr[b]), W_arr[c])
        if abs(val) > 1e-6:
            return False
    return True

def _try_rank_r(n, m, p, r, seed=0):
    """Try to find a rank-r decomposition using randomized search."""
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n*m, r)
    V = rng.randn(m*p, r)
    W = rng.randn(p*n, r)
    
    # Normalize to satisfy valid term constraints
    # For each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is a system of nonlinear equations. Use iterative refinement.
    
    # Simple approach: just check if random works (unlikely)
    if _verify_sampled(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Schoolbook as baseline
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find better decompositions with time guard
    # For now, return schoolbook
    return best
# EVOLVE-BLOCK-END
