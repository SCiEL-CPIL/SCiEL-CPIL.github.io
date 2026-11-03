"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    """Verify that U, V, W form a valid tensor decomposition."""
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    rank = len(U[0])
    
    # Build the multiplication tensor
    # T[i, j, k] = 1 if we're computing C[i, k] += A[i, j] * B[j, k]
    # The tensor is T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # Actually, the standard decomposition is:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # So the tensor has T[i, j, k] = 1 for the (i, j, k) term
    # In the factorization: sum_r U[a, r] * V[b, r] * W[c, r]
    # where a indexes A, b indexes B, c indexes C
    
    # Let's verify by checking a few random cases
    import numpy as np
    
    # Check with specific values
    for trial in range(10):
        A = np.random.randint(-3, 4, size=(n, m))
        B = np.random.randint(-3, 4, size=(m, p))
        C_expected = A @ B
        
        # Compute C from decomposition
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_col = np.array(U, dtype=float)[:, r].reshape(n, m)
            v_col = np.array(V, dtype=float)[:, r].reshape(m, p)
            w_col = np.array(W, dtype=float)[:, r].reshape(p, n)
            # The term is outer product structure
            # C += (U[:,r] as (n,m)) * (V[:,r] as (m,p)) * (W[:,r] as (p,n))
            # Actually the decomposition is:
            # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
        # Let's compute properly
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                total = 0.0
                for j in range(m):
                    for r in range(rank):
                        total += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                C_computed[i, k] = total
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def _try_random_search(max_rank, time_budget=10.0):
    """Try to find a decomposition with rank < max_rank using random search."""
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    start_time = time.time()
    
    # For each rank from max_rank-1 down to some minimum, try random initializations
    for rank in range(max_rank - 1, max(10, max_rank - 20), -1):
        if time.time() - start_time > time_budget:
            break
        
        for attempt in range(50):
            if time.time() - start_time > time_budget:
                break
            
            # Random initialization with small integers
            U = [[random.choice([-1, 0, 0, 1]) for _ in range(rank)] for _ in range(nm)]
            V = [[random.choice([-1, 0, 0, 1]) for _ in range(rank)] for _ in range(mp)]
            W = [[random.choice([-1, 0, 0, 1]) for _ in range(rank)] for _ in range(pn)]
            
            if _verify(U, V, W):
                return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    # For 3x5x5, the schoolbook rank is 75.
    # Let's first try to find a sub-schoolbook decomposition.
    # If we can't find one quickly, return the schoolbook.
    
    schoolbook = _schoolbook()
    
    # Try a brief random search for rank < 75
    # This is unlikely to succeed but let's try
    result = _try_random_search(75, time_budget=5.0)
    
    if result is not None:
        return result
    
    return schoolbook
# EVOLVE-BLOCK-END
