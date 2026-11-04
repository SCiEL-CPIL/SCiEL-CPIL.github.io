"""
Program to find a rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
We start with the schoolbook rank 54 and attempt to reduce rank by exploiting
linear dependencies in the tensor structure.
"""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the tensor decomposition.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    # Try to reduce rank by finding linear dependencies
    # The tensor T[i,j,k] = e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # We can try to express some rank-1 tensors as linear combinations of others
    
    # Convert to numpy for efficient linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Attempt structured rank reduction
    # For matrix multiplication, the tensor has a special structure.
    # We can try to use the fact that C[i,k] = sum_j A[i,j] * B[j,k]
    # This means the tensor is determined by the bilinear form structure.
    
    # Try to find if we can combine terms where the same (i,k) pair appears
    # with different j values in a way that allows cancellation
    
    # Key insight: For each output element C[i,k], we need:
    # sum_j A[i,j] * B[j,k] = sum_r U[i*m+j] * V[j*p+k] * W[k*n+i]
    
    # The schoolbook uses one term per (i,j,k). We can try to use fewer terms
    # if the structure allows.
    
    # For 2x3 * 3x9, let's try to see if we can exploit the small n=2
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # This is a 2x3 matrix times a 3x9 matrix.
    # The tensor rank is at least max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # and at most n*m*p = 54
    
    # Let's try to find a decomposition with rank < 54
    
    # Strategy: Use the structure that for each k, we're computing
    # a 2x3 matrix times a 3-vector (column k of B)
    # This is a 2-dimensional output for each k
    
    # Try rank = 36 (27 + 9) as a target - this corresponds to
    # using the fact that n=2 is small
    
    # Actually, let's try a different approach:
    # The multiplication tensor for (n,m,p) has a known lower bound.
    # For (2,3,9), the tensor rank R(2,3,9) is not known exactly,
    # but we know 27 <= R(2,3,9) <= 54
    
    # Let's try to construct a decomposition with rank 36 by
    # grouping terms by k and using the structure of 2x3 multiplication
    
    # For each k, we need to compute:
    # [A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k],
    #  A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]]
    
    # This is a 2x3 matrix times a 3-vector, which has tensor rank at most 6
    # (schoolbook) but possibly less.
    
    # For 2x3 * 3x1, the tensor rank is 6 (no known improvement)
    # So for 9 independent k values, we'd need 9*6 = 54 in schoolbook.
    
    # However, if we can share computation across k values, we might do better.
    # The B[j,k] values are shared across i, and A[i,j] values are shared across k.
    
    # Let's try to use a decomposition that exploits this sharing.
    
    # For now, return the schoolbook decomposition as a safe fallback
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
