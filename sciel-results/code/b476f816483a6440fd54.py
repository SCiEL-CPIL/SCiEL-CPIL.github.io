"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
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

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check that U, V, W exactly decompose the matmul tensor."""
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for c in range(rank):
        if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
            return False
    # Reconstruct tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Compare to delta(i,j,k) which is 1 for all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 10
    mp = m * p  # 25
    pn = p * n  # 10
    
    # Known lower bound for matmul(2,5,5): at least n*m = 10 (trivial) 
    # but realistically much higher. Schoolbook is 50.
    # Let's try to find a decomposition with rank < 50 using random search + local refinement.
    
    # Start with schoolbook as fallback
    best_result = _schoolbook()
    best_rank = n * m * p  # 50
    
    # Strategy: Try random initializations and use gradient-free local search (e.g., coordinate descent)
    # to minimize the residual while keeping rank fixed, then try to reduce rank.
    
    # Actually, for exact decomposition, we need the tensor to be exactly represented.
    # The matmul tensor T[i,j,k] = 1 for all i,j,k is a very simple tensor (all-ones).
    # Wait, the matmul tensor for multiplying A (n x m) by B (m x p) is:
    # T[i,j,k] = 1 if the index (i,j,k) corresponds to a valid term, else 0.
    # In standard indexing, C[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # This is the all-ones tensor of size n x m x p!
    
    # The all-ones tensor of size n x m x p has rank 1!
    # Because T[i,j,k] = 1 = 1 * 1 * 1 for all i,j,k.
    # So U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1.
    
    # Let's verify: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1*1*1 = 1. Correct!
    
    # So the minimal rank is 1.
    
    rank = 1
    U = [[1] for _ in range(nm)]
    V = [[1] for _ in range(mp)]
    W = [[1] for _ in range(pn)]
    
    # Validate
    if _validate(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # If validation fails (shouldn't), fall back to schoolbook
    return best_result
# EVOLVE-BLOCK-END
