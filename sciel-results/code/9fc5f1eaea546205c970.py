"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n=3, m=3, p=5, atol=1e-7):
    """
    Verify that the tensor decomposition exactly matches matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The multiplication tensor T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    for all i in [0,n), j in [0,m), k in [0,p).
    """
    if isinstance(U, list):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
    else:
        U_np = U
        V_np = V
        W_np = W
    
    rank = U_np.shape[1]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
    
    target = np.ones((n, m, p))
    return np.allclose(T, target, atol=atol)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _try_decomposition(n, m, p, rank, timeout=10.0):
    """
    Try to find a decomposition of rank 'rank' for n x m x p matrix multiplication.
    Uses a structured approach based on known lower bounds and random search.
    """
    start_time = time.time()
    
    # For 3x3x3, the minimal rank is 21 (Strassen-like or similar)
    # For 3x3x5, we can try to extend the 3x3x3 decomposition
    
    # Strategy 1: If p >= m, try to use a decomposition for n x m x m and extend
    if p >= m:
        # First get a good decomposition for n x m x m
        core_rank = n * m * m  # Start with schoolbook for core
        # Try to find a better core decomposition
        if n == 3 and m == 3:
            # Known minimal rank for 3x3x3 is 21
            core_rank = 21
        
        # Try to find core decomposition
        U_core, V_core, W_core = _schoolbook(n, m, m)
        
        # If we have a better core decomposition, use it
        if core_rank < n * m * m:
            # For now, we'll use schoolbook core and extend
            pass
        
        # Extend from n x m x m to n x m x p
        # The extra dimensions are k = m, ..., p-1
        extra_dims = p - m
        
        # For each extra dimension k, we need to add terms
        # T[i,j,k] = 1 for all i,j
        # We can use terms that are 1 for specific (i,j) and the extra k
        
        # Actually, a simpler extension: for each extra k, we need rank n*m terms
        # But we can share some structure
        
        # Let's use the full schoolbook for simplicity first, then try to optimize
        # Total rank = core_rank + extra_dims * n * m
        # But this might not be minimal
        
        # Better: use the structure of matrix multiplication
        # C_ik = sum_j A_ij B_jk
        # For k < m: use core decomposition
        # For k >= m: we need separate terms
        
        # Let's construct a decomposition
        total_rank = core_rank + extra_dims * n * m
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        # Copy core decomposition
        U[:, :core_rank] = U_core
        V[:, :core_rank] = V_core
        W[:, :core_rank] = W_core
        
        # Add terms for extra dimensions
        col = core_rank
        for k_extra in range(extra_dims):
            k = m + k_extra
            for i in range(n):
                for j in range(m):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        
        if _verify(U, V, W, n, m, p):
            return U, V, W
        
    # Fallback: schoolbook
    U, V, W = _schoolbook(n, m, p)
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U_school, V_school, W_school)
    
    # Try to find a better decomposition
    # For 3x3x5, try ranks from 30 up to best_rank
    for rank in range(30, best_rank + 1):
        if time.time() > deadline - 5:
            break
        
        result = _try_decomposition(n, m, p, rank, timeout=5.0)
        if result is not None:
            U_try, V_try, W_try = result
            if _verify(U_try, V_try, W_try, n, m, p):
                if rank < best_rank:
                    best_rank = rank
                    best_result = (U_try, V_try, W_try)
                    break  # Found better, stop
    
    U_best, V_best, W_best = best_result
    
    # Convert to list format if needed
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
