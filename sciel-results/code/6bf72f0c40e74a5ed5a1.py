"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Construct the standard rank n*m*p decomposition for n x m * m x p."""
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-10):
    """Check that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all i,j,k."""
    R = U.shape[1]
    # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
    Ur = U.reshape(n, m, R)
    Vr = V.reshape(m, p, R)
    Wr = W.reshape(p, n, R)
    # Compute T[i,j,k] = sum_r Ur[i,j,r] * Vr[j,k,r] * Wr[k,i,r]
    # For each (i,j,k):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += Ur[i, j, r] * Vr[j, k, r] * Wr[k, i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_random_rank(R, n, m, p, time_limit, seed_offset=0):
    """Attempt to find a rank-R decomposition via random initialization + gradient-free search.
    This is extremely unlikely to succeed for R < 36, but we try briefly."""
    if R >= n * m * p:
        return None
    start = time.time()
    rng = np.random.RandomState(42 + seed_offset)
    
    # Initialize with small random values
    for attempt in range(50):
        if time.time() - start > time_limit:
            break
        U = rng.randn(n * m, R) * 0.1
        V = rng.randn(m * p, R) * 0.1
        W = rng.randn(p * n, R) * 0.1
        
        # Quick check
        if _verify(U, V, W, n, m, p):
            return (U, V, W)
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    # 1. Schoolbook Decomposition (Rank 36) - always valid
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = n * m * p  # 36
    
    # 2. Try to find lower rank decompositions
    # Strategy: Try ranks from best_rank-1 downward, with decreasing time budgets
    remaining_time = deadline - time.time()
    if remaining_time < 5.0:
        return best_result
    
    # Try rank 35 with a short time budget
    time_budget_35 = min(10.0, remaining_time * 0.3)
    result_35 = _try_random_rank(35, n, m, p, time_budget_35, seed_offset=0)
    if result_35 is not None:
        U35, V35, W35 = result_35
        if _verify(U35, V35, W35, n, m, p):
            best_result = {
                "U": U35.tolist(),
                "V": V35.tolist(),
                "W": W35.tolist()
            }
            best_rank = 35
    
    remaining_time = deadline - time.time()
    if remaining_time < 5.0:
        return best_result
    
    # Try rank 34 with remaining time
    if best_rank > 34:
        time_budget_34 = min(20.0, remaining_time * 0.5)
        result_34 = _try_random_rank(34, n, m, p, time_budget_34, seed_offset=100)
        if result_34 is not None:
            U34, V34, W34 = result_34
            if _verify(U34, V34, W34, n, m, p):
                best_result = {
                    "U": U34.tolist(),
                    "V": V34.tolist(),
                    "W": W34.tolist()
                }
                best_rank = 34
    
    return best_result
# EVOLVE-BLOCK-END
