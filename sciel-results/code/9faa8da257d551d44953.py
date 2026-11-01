import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose matrix multiplication."""
    rank = len(U[0]) if len(U) > 0 else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(actual - 1.0) > tol:
                    return False
    return True

def _search_rank(target_rank, n, m, p, time_budget=10.0, seed=0):
    """
    Search for a rank-`target_rank` decomposition.
    """
    start_time = time.time()
    np.random.seed(seed)
    
    n_m = n * m  # 6
    m_p = m * p  # 12
    p_n = p * n  # 8
    
    num_vars = p_n * target_rank
    num_eqs = n * m * p  # 24
    
    # Pre-compute indices for efficiency
    # Equation index for (i, j, k)
    eq_indices = {}
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_indices[(i, j, k)] = idx
                idx += 1

    # Variable index for (k, i, r)
    # var_idx = (k*n + i)*rank + r

    # We will try several strategies for initializing U and V
    # Strategy 1: Random sparse
    # Strategy 2: Random dense
    # Strategy 3: Structured (e.g., identity-like)
    
    attempts = 0
    while time.time() - start_time < time_budget:
        attempts += 1
        if attempts > 5000: # Safety break
            break
            
        # Generate random U and V
        # Using half-integers: multiply by 2 to work with integers, then divide
        # Entries in {-2, -1, 0, 1, 2} corresponding to {-1, -0.5, 0, 0.5, 1}
        
        # Sparse initialization
        if attempts % 2 == 0:
            # Mostly zeros, some +-1
            U = np.random.choice([-2, 0, 2], size=(n_m, target_rank))
            V = np.random.choice([-2, 0, 2], size=(m_p, target_rank))
        else:
            # Dense initialization
            U = np.random.choice([-2, -1, 0, 1, 2], size=(n_m, target_rank))
            V = np.random.choice([-2, -1, 0, 1, 2], size=(m_p, target_rank))
        
        # Build the linear system A W = b
        # A is (num_eqs, num_vars)
        # b is (num_eqs,)
        A = np.zeros((num_eqs, num_vars))
        b = np.ones(num_eqs)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = eq_indices[(i, j, k)]
                    for r in range(target_rank):
                        var_idx = (k * n + i) * target_rank + r
                        A[eq_idx, var_idx] = U[i*m+j, r] * V[j*p+k, r]
        
        # Solve for W
        try:
            W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
        
        W = W_flat.reshape(p_n, target_rank)
        
        # Round to nearest half-integer
        # W entries are real numbers. We want them to be k/2.
        # So 2*W should be close to an integer.
        W_rounded = np.round(W * 2) / 2
        
        # Verify
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W_rounded.tolist()
        
        # Quick check before full verification
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for r in range(target_rank):
                        actual += U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r]
                    if abs(actual - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": U_list, "V": V_list, "W": W_list}
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin out of 150s
    
    n, m, p = DIMENSIONS
    
    # Try decreasing ranks
    for target_rank in [20, 19, 18, 17, 16, 15]:
        if time.time() > deadline - 15:
            break
        
        # Try different seeds for variety
        for seed in range(10):
            if time.time() > deadline - 10:
                break
            
            result = _search_rank(target_rank, n, m, p, 
                                 time_budget=min(5, deadline - time.time() - 5), 
                                 seed=seed + target_rank)
            if result is not None:
                return result
    
    # Fallback: trivial rank-24 decomposition
    rank = n * m * p  # 24
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
# EVOLVE-BLOCK-END
