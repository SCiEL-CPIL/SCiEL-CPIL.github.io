"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s out of 150s wall-clock
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_school
    
    # Try to find a valid decomposition at various ranks
    # Lower bound: max(n*p, m*p, n*m) = max(18, 30, 15) = 30
    min_rank = max(n * p, m * p, n * m)
    
    # We will try a few random searches for ranks close to the lower bound
    # but limit the time to ensure we return the fallback if it fails.
    
    for target_rank in range(min_rank, rank_school):
        if time.time() > deadline - 10:
            break
            
        # Try a few random initializations
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
                
            # Random initialization with small integer/half-integer values
            vals = [-1, -0.5, 0, 0.5, 1]
            
            U = np.random.choice(vals, size=(n * m, target_rank))
            V = np.random.choice(vals, size=(m * p, target_rank))
            W = np.random.choice(vals, size=(p * n, target_rank))
            
            if _verify_fast(U, V, W, n, m, p):
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = target_rank
                return best_result
    
    return best_result

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using einsum."""
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)

def _verify(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is exact.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    r = U.shape[1]
    
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _try_strassen_based(n, m, p, time_limit):
    """
    Try to find a decomposition using Strassen-like techniques.
    Placeholder.
    """
    pass

# EVOLVE-BLOCK-END
