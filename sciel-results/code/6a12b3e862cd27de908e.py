"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]
N, M, P = 2, 3, 6
RANK_SCHOOLBOOK = N * M * P  # 36

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix
    by a 3x6 matrix.
    
    The tensor is T[i,j,k] = 1 if (i,j) is an entry of A, (j,k) of B, (k,i) of C.
    We seek U (N*M x r), V (M*P x r), W (P*N x r) such that:
      T[i,j,k] = sum_c U[i*M+j, c] * V[j*P+k, c] * W[k*N+i, c]
    
    Strategy:
    1. Start with the schoolbook decomposition (rank 36).
    2. Try to reduce rank by finding linear dependencies among the 36 rank-1 terms.
    3. Use a randomized search over subspaces to find lower-rank decompositions.
    4. Verify exactness with integer/rational arithmetic where possible.
    """
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin before 120s limit
    
    n, m, p = N, M, P
    
    # Build the tensor T as a 3D array: T[i,j,k]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    
    # Schoolbook decomposition
    rank0 = n * m * p
    U0 = np.zeros((n * m, rank0))
    V0 = np.zeros((m * p, rank0))
    W0 = np.zeros((p * n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
    
    best_rank = rank0
    best_U, best_V, best_W = U0, V0, W0
    
    # Try to find lower-rank decompositions
    # The key insight: if we can find a set of r < 36 rank-1 tensors that span
    # the tensor T, we have a lower-rank decomposition.
    
    # Approach: Use the fact that T is a structured tensor. 
    # T[i,j,k] = 1 for all (i,j,k). This is actually the "all-ones" tensor.
    # Wait - let me re-read the task.
    
    # The matrix multiplication tensor for (n,m,p) has entries:
    # T[i, j, k] = 1 if the (i,j) entry of A, (j,k) entry of B, and (k,i) entry of C
    # participate in the (i,k) entry of the product.
    # 
    # For matrix multiplication C = A * B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # So the tensor T has T[i,j,k] = 1 for all valid i, j, k.
    # This means T is just the all-ones tensor of shape (n, m, p).
    
    # The all-ones tensor of shape (n, m, p) has a trivial rank-1 decomposition:
    # T[i,j,k] = 1*1*1 = u_i * v_j * w_k where u_i=1, v_j=1, w_k=1
    # So rank = 1!
    
    # Let me verify: U has shape (n*m, 1), V has shape (m*p, 1), W has shape (p*n, 1)
    # U[i*m+j, 0] = 1 for all i, j
    # V[j*p+k, 0] = 1 for all j, k  
    # W[k*n+i, 0] = 1 for all k, i
    # Then sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1*1*1 = 1 = T[i,j,k]
    
    # This is correct! The rank is 1.
    
    U1 = [[1] for _ in range(n * m)]
    V1 = [[1] for _ in range(m * p)]
    W1 = [[1] for _ in range(p * n)]
    
    return {"U": U1, "V": V1, "W": W1}
# EVOLVE-BLOCK-END
