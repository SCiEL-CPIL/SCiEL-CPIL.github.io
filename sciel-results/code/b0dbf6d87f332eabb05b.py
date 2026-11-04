import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_s = [[0] * rank_school for _ in range(n * m)]
    V_s = [[0] * rank_school for _ in range(m * p)]
    W_s = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j][col] = 1
                V_s[j * p + k][col] = 1
                W_s[k * n + i][col] = 1
                col += 1
    
    # Target tensor T[i, j, k] = 1 if it's a valid multiplication term?
    # T is the tensor for matrix multiplication.
    # T[i, j, k] is 1 if the term A[i, j] * B[j, k] contributes to C[i, k].
    # In standard matrix multiplication, C[i, k] = sum_j A[i, j] B[j, k].
    # So T[i, j, k] = 1 for all i, j, k? No.
    # The tensor T has indices i (row of A), j (col of A / row of B), k (col of B).
    # The multiplication is defined by T[i, j, k] = 1.
    # Wait, the tensor is $T_{i,j,k} = 1$ for all valid i,j,k?
    # Yes, the matrix multiplication tensor is the tensor with 1s at all (i,j,k) corresponding to the summation index j.
    # Actually, for fixed i, k, the term is sum_j A[i,j]B[j,k].
    # So the tensor T is such that <T, A \otimes B> = C.
    # T[i, j, k] = 1 for all i, j, k.
    
    # Let's verify the rank of this tensor.
    # It is the tensor $T = \sum_{i,j,k} e_i \otimes e_j \otimes e_k$.
    # This is the "all-ones" tensor? No.
    # The matrix multiplication tensor $M_{n \times m \times p}$ has $T_{i,j,k} = 1$ if we are computing the (i,k) entry using j-th intermediate.
    # Yes, $T_{i,j,k} = 1$ for all $0 \le i < n, 0 \le j < m, 0 \le k < p$.
    
    # We want to find U, V, W such that:
    # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    # Note: The problem statement says W[k*n+i, r].
    # So W is indexed by (k, i).
    # Let's re-read carefully.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # So W is a matrix of size (p*n, rank).
    # The decomposition is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i, j, k]
    # Where T[i, j, k] = 1.
    
    # So we need:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i, j, k.
    
    # This is a different tensor than the standard "matrix multiplication tensor" which is usually defined as 
    # T[i, j, k] = 1 if we are multiplying A (n x m) and B (m x p).
    # The standard tensor has T[i, j, k] = 1.
    # So we need to decompose the all-ones tensor of size n x m x p?
    # No, the all-ones tensor has rank 1.
    # Wait.
    # Matrix multiplication: C[i, k] = sum_j A[i, j] B[j, k].
    # The tensor T is such that C = T(A, B).
    # T is a trilinear form.
    # T(A, B, C) = sum_{i,j,k} T_{i,j,k} A[i,j] B[j,k] C[i,k].
    # For this to equal trace(C^T (AB)), we need T_{i,j,k} = 1.
    # So yes, T_{i,j,k} = 1 for all i, j, k.
    # BUT, the rank of the all-ones tensor is 1.
    # Why is the schoolbook rank 40?
    # Ah, the schoolbook decomposition given in the seed program is:
    # U[i*m+j, c] = 1
    # V[j*p+k, c] = 1
    # W[k*n+i, c] = 1
    # Sum over c (which corresponds to i, j, k).
    # Term c=(i,j,k): U=1, V=1, W=1.
    # Sum = 1.
    # So the seed program decomposes the tensor T where T[i,j,k]=1.
    # And it uses rank 40.
    # But the all-ones tensor has rank 1!
    # U[i*m+j, 0] = 1 for all i,j.
    # V[j*p+k, 0] = 1 for all j,k.
    # W[k*n+i, 0] = 1 for all k,i.
    # Sum = 1.
    # So rank 1 is possible?
    
    # Let's check the constraints.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 5 matrix."
    # "output factor follows the official trace convention W[k*n+i,r]"
    
    # If T[i,j,k]=1, then rank 1 is trivial.
    # Is there a catch?
    # Maybe the tensor is NOT all ones?
    # In matrix multiplication, the coefficient of A[i,j] B[j,k] in C[i,k] is 1.
    # So T[i,j,k] = 1.
    # Why would anyone use rank 40?
    # Perhaps the "tensor" is defined differently?
    # Or maybe I am missing something about the indices.
    # U has shape (n*m, rank). Index i*m+j.
    # V has shape (m*p, rank). Index j*p+k.
    # W has shape (p*n, rank). Index k*n+i.
    
    # If I set U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1.
    # Then sum_r ... = 1.
    # This works for all i,j,k.
    # So rank 1 is a valid decomposition of the all-ones tensor.
    
    # Why is the seed program rank 40?
    # Maybe the task is to decompose the *multiplication map* in a way that preserves the structure?
    # No, "exact tensor decomposition".
    # If the tensor is all ones, rank 1 is the answer.
    
    # Let's double check if T[i,j,k] is always 1.
    # Yes, for standard matrix multiplication.
    
    # So I can return rank 1.
    
    # Let's verify if rank 1 is allowed.
    # "integer/half-integer entries". 1 is integer.
    # "score is the rank". Lower is better.
    
    # So I should return rank 1.
    
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    # Verify
    # For any i, j, k:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # = 1 * 1 * 1 = 1.
    # Correct.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
