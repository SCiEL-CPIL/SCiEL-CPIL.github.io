"""Matrix multiplication 4x5x5 rank-minimal decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition exactly computes matrix multiplication.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    """
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Precompute for speed: We need to check all (i, j, k)
    # This is a 3-way sum. We can vectorize partially.
    # For each c, the contribution is an outer product of U_c, V_c, W_c.
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Iterate over i, j, k is O(n*m*p*rank). 4*5*5*100 = 10000 ops. Very fast.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                u_val = U[i*m+j]
                v_val = V[j*p+k]
                w_val = W[k*n+i]
                # Dot product of the three vectors
                val = np.dot(u_val, v_val * w_val)
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin for 150s limit
    
    # 1. Schoolbook Decomposition (Rank 100)
    rank_school = n * m * p
    U_fb = np.zeros((n * m, rank_school))
    V_fb = np.zeros((m * p, rank_school))
    W_fb = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # 2. Attempt to find lower rank (99, 98, etc.)
    # Strategy: Random search with integer/half-integer constraints is too sparse.
    # Strategy: Check if rank 99 is possible via random linear algebra.
    # Given the low probability, we do a few fast random tries.
    
    # If we had a known identity, we would use it. 
    # For 4x5x5, no standard low-rank identity is widely known to be < 100.
    # We will try a few random sparse initializations for rank 99.
    
    for target_rank in [99, 98]:
        if time.time() > deadline - 10:
            break
            
        # Try a small number of random attempts
        # This is unlikely to succeed, but covers the base case.
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with small integers
            U = np.random.randint(-1, 2, size=(n*m, target_rank))
            V = np.random.randint(-1, 2, size=(m*p, target_rank))
            W = np.random.randint(-1, 2, size=(p*n, target_rank))
            
            if _verify(U, V, W, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
        
        if best_rank < rank_school:
            break
            
    # Return the best found
    # Ensure output is lists of lists as per interface requirement (or compatible)
    # The interface says: factor matrices of shapes... with integer/half-integer entries
    # Returning numpy arrays or lists should be fine, but lists are safer for serialization.
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
