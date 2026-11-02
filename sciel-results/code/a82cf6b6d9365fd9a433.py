"""Tensor decomposition for 3x5 * 5x5 matrix multiplication."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Strategy: Use Kronecker product structure
    # Express 3x5 as 3x1 block vector of 1x5 rows
    # Express 5x5 as 5x1 block vector of 1x5 rows
    # The outer product structure is separable
    # 
    # For a 3x5 * 5x5 multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We can decompose this as:
    # - 3 copies of 1x5 * 5x5 (each with rank 5)
    # Total rank: 3 * 5 = 15
    #
    # But we can do better using the full structure.
    # Let's use a rank-15 decomposition based on tensor products.
    
    # Actually, let's think about this more carefully.
    # The matrix multiplication tensor for n x m * m x p has rank at most n*m*p
    # but can be much less.
    
    # For 3x5 * 5x5, we can use:
    # rank = n * p = 3 * 5 = 15
    # This comes from: C = A * B where each row of A is a 1x5 vector
    # and each column of B is a 5x1 vector
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # A rank-r decomposition has:
    # U of shape (n*m, r) = (15, r)
    # V of shape (m*p, r) = (25, r)  
    # W of shape (p*n, r) = (15, r)
    
    # The tensor T[i,j,k] = 1 if we're multiplying A[i,j] by B[j,k]
    # In our indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # So T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    
    # For the identity multiplication tensor, T[a,b,c] = 1 iff a's j-index matches b's j-index
    # and c's indices are consistent.
    
    # Let me try a rank-15 decomposition:
    # For each i in range(3), k in range(5):
    #   We need to compute C[i,k] = sum_j A[i,j] B[j,k]
    #   This requires 5 multiplications (one for each j)
    #   But if we share across k for fixed i, we get rank 3*5 = 15
    
    # Actually, let's just try rank = n*p = 15 first
    rank = n * p  # 15
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # For rank n*p decomposition:
    # Index c = i*p + k
    # U[i*m + j, c] should be A[i,j] coefficient
    # V[j*p + k, c] should be B[j,k] coefficient
    # W[k*n + i, c] should be C[k,i]... wait, W[k*n+i, c]
    
    # Let me reconsider the indexing.
    # U has shape (n*m, rank), indexed by (i, j) -> i*m + j
    # V has shape (m*p, rank), indexed by (j, k) -> j*p + k
    # W has shape (p*n, rank), indexed by (k, i) -> k*n + i
    
    # The product sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] should give
    # the coefficient for A[i,j] * B[j,k] contributing to C[i,k]
    
    # For the multiplication tensor, we want:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all valid (i,j,k)
    # where the tensor T[i,j,k] = 1 always (it's the full multiplication tensor)
    
    # Wait, no. The tensor T[a,b,c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a,b,c] = 1 if the j-index in a matches the j-index in b
    # Otherwise T[a,b,c] = 0
    
    # So we need sum_c U[a,c] V[b,c] W[c_index,c] = T[a,b,c_index]
    
    # Let me use a different approach. Start with schoolbook and try to reduce.
    
    # Schoolbook: rank = n*m*p = 75
    # For each (i,j,k): U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    
    # Try rank = n*p = 15:
    # c = i*p + k
    # U[i*m + j, c] = 1 for all j? No, that would sum over j.
    # Actually, for fixed (i,k), we need sum_j U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    # If we set U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # Then sum_j 1*1*1 = 5, not 1.
    
    # We need the sum to be 1. So we could use 1/5:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1/5, W[k*n+i, c] = 1
    # But entries should be integer or half-integer.
    
    # Let me try: U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # This gives sum = 5 for each (i,k), which is wrong.
    
    # The issue is that the tensor T[i,j,k] = 1 for ALL (i,j,k), not just when indices match.
    # Wait, I'm confusing myself. Let me re-read the task.
    
    # "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k
    # because C[i,k] = sum_j A[i,j] B[j,k]
    
    # So we need: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i,j,k
    
    # With rank = n*p = 15, c = i*p + k:
    # We need sum_j U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    # If U[i*m+j, c] = 1 for all j, V[j*p+k, c] = 1 for all j, W[k*n+i, c] = 1
    # Then sum = 5, not 1.
    
    # We need to divide by 5, but 1/5 is not integer or half-integer.
    
    # Let me try rank = n*m*p = 75 (schoolbook) and see if we can do better.
    
    # Actually, let's just implement a simple approach: use schoolbook rank and return it.
    # The score is 75, and we're trying to beat that.
    
    # Let me try a different rank. The known lower bound for 3x5 * 5x5 is at least 3*5 = 15.
    # Upper bound is 75.
    
    # Let me try to find a rank-15 or rank-20 decomposition.
    
    # For now, let's just return the schoolbook decomposition with rank 75.
    rank = n * m * p
    U_list = [[0] * rank for _ in range(n * m)]
    V_list = [[0] * rank for _ in range(m * p)]
    W_list = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_list[i * m + j][column] = 1
                V_list[j * p + k][column] = 1
                W_list[k * n + i][column] = 1
                column += 1
    
    return {"U": U_list, "V": V_list, "W": W_list}
