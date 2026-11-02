"""Matrix multiplication 2,2,8 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix
    by a 2x8 matrix.
    
    The matrix multiplication tensor T is defined such that:
    (A * B)[i,k] = sum_j A[i,j] * B[j,k]
    
    Where A is n x m (2x2), B is m x p (2x8), and the result is n x p (2x8).
    
    The tensor T has indices (i, j, k) with T[i,j,k] = 1 if and only if
    the product A[i,j] * B[j,k] contributes to the output.
    
    For matrix multiplication, the tensor is:
    T[i,j,k] = 1 for all valid i,j,k (since each A[i,j]*B[j,k] term appears)
    
    Wait, let me reconsider. The standard tensor for matrix multiplication:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    This is a trilinear form. The tensor T has entries T[i,j,k] where:
    - i indexes rows of A (n = 2)
    - j indexes columns of A / rows of B (m = 2)  
    - k indexes columns of B (p = 8)
    
    And T[i,j,k] = 1 for all i, j, k in the standard decomposition.
    
    The schoolbook decomposition has rank n*m*p = 2*2*8 = 32.
    
    But we can do better! The key insight is that for matrix multiplication,
    we can use the fact that the multiplication tensor has structure.
    
    Actually, for multiplying an n x m matrix by an m x p matrix, the minimum
    rank is known to be at most n*m*p, but can be lower.
    
    For 2x2 * 2x8, let's think about this differently.
    
    Let A be 2x2 and B be 2x8. The product C = A*B is 2x8.
    
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    ...
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] for k=0..7
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] for k=0..7
    
    So we have 16 output elements, each being a sum of 2 products.
    
    The schoolbook approach uses 2*2*8 = 32 rank-1 terms.
    
    Can we do better? Let's think about the tensor structure.
    
    The tensor T is in V_i ⊗ V_j ⊗ V_k where dim V_i = 2, dim V_j = 2, dim V_k = 8.
    
    T = sum_{i=0}^{1} sum_{j=0}^{1} sum_{k=0}^{7} e_i ⊗ e_j ⊗ e_k
    
    This is a sum of 32 rank-1 tensors. But can we express it with fewer?
    
    Note that T can be written as:
    T = (e_0 ⊗ e_0) ⊗ (sum_k e_k) + (e_0 ⊗ e_1) ⊗ (sum_k e_k) + 
        (e_1 ⊗ e_0) ⊗ (sum_k e_k) + (e_1 ⊗ e_1) ⊗ (sum_k e_k)
    
    = [(e_0 ⊗ e_0) + (e_0 ⊗ e_1) + (e_1 ⊗ e_0) + (e_1 ⊗ e_1)] ⊗ (sum_k e_k)
    
    Wait, that's not quite right because the tensor product structure matters.
    
    Let me be more careful. The tensor T ∈ R^{2} ⊗ R^{2} ⊗ R^{8} where:
    T[i,j,k] = 1 for all i∈{0,1}, j∈{0,1}, k∈{0,...,7}
    
    So T = sum_{i=0}^1 sum_{j=0}^1 sum_{k=0}^7 e_i ⊗ e_j ⊗ e_k
    
    We can factor this as:
    T = (sum_{i=0}^1 sum_{j=0}^1 e_i ⊗ e_j) ⊗ (sum_{k=0}^7 e_k)
    
    But this is a single term in a different tensor product space, not a sum of rank-1 terms in the original space.
    
    For the rank-1 decomposition, we need:
    T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    
    where u_r ∈ R^2, v_r ∈ R^2, w_r ∈ R^8.
    
    The question is: what is the minimum R?
    
    Let's think about it. If we set u_r = e_{i_r}, v_r = e_{j_r}, w_r = e_{k_r}, then we get the schoolbook decomposition with R = 32.
    
    But we can use linear combinations. For example:
    
    Consider u_1 = e_0 + e_1, v_1 = e_0, w_1 = sum_k e_k
    u_2 = e_0 + e_1, v_2 = e_1, w_2 = sum_k e_k
    
    Then u_1 ⊗ v_1 ⊗ w_1 + u_2 ⊗ v_2 ⊗ w_2 = (e_0 + e_1) ⊗ e_0 ⊗ (sum_k e_k) + (e_0 + e_1) ⊗ e_1 ⊗ (sum_k e_k)
    = (e_0 ⊗ e_0 + e_1 ⊗ e_0) ⊗ (sum_k e_k) + (e_0 ⊗ e_1 + e_1 ⊗ e_1) ⊗ (sum_k e_k)
    = (e_0 ⊗ e_0 + e_1 ⊗ e_0 + e_0 ⊗ e_1 + e_1 ⊗ e_1) ⊗ (sum_k e_k)
    = T
    
    So we can do it with just 2 rank-1 terms!
    
    Let me verify:
    Term 1: u_1 = [1,1], v_1 = [1,0], w_1 = [1,1,1,1,1,1,1,1]
    Term 2: u_2 = [1,1], v_2 = [0,1], w_2 = [1,1,1,1,1,1,1,1]
    
    Sum: For each (i,j,k):
    Contribution = u_1[i]*v_1[j]*w_1[k] + u_2[i]*v_2[j]*w_2[k]
    
    For i=0,j=0,k=any: 1*1*1 + 1*0*1 = 1 ✓
    For i=0,j=1,k=any: 1*0*1 + 1*1*1 = 1 ✓
    For i=1,j=0,k=any: 1*1*1 + 1*0*1 = 1 ✓
    For i=1,j=1,k=any: 1*0*1 + 1*1*1 = 1 ✓
    
    Yes! Rank 2 is sufficient.
    
    But wait, can we do rank 1? That would require T = u ⊗ v ⊗ w for some u ∈ R^2, v ∈ R^2, w ∈ R^8.
    Then T[i,j,k] = u[i]*v[j]*w[k]. For this to equal 1 for all i,j,k, we'd need u[i]*v[j]*w[k] = 1 for all i,j,k.
    This means u[0]*v[0]*w[0] = 1, u[0]*v[0]*w[1] = 1, so w[0] = w[1]. Similarly all w[k] are equal.
    Also u[0]*v[0] = u[0]*v[1] so v[0] = v[1] (assuming u[0] ≠ 0).
    And u[0]*v[0] = u[1]*v[0] so u[0] = u[1].
    So u = [a,a], v = [b,b], w = [c,c,...,c] with a*b*c = 1.
    Then T[i,j,k] = a*b*c = 1 for all i,j,k. Yes!
    
    So rank 1 works! Let u = [1,1], v = [1,1], w = [1,1,1,1,1,1,1,1].
    
    Verify: T[i,j,k] = u[i]*v[j]*w[k] = 1*1*1 = 1 for all i,j,k. ✓
    
    So the minimum rank is 1!
    
    Now let's set up the factor matrices according to the interface:
    U has shape (n*m, rank) = (4, 1)
    V has shape (m*p, rank) = (16, 1)
    W has shape (p*n, rank) = (16, 1)
    
    The decomposition is:
    (A*B)[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, I need to be careful about the index conventions. Let me re-read the task.
    
    "The output factor follows the official trace convention W[k*n+i,r]."
    
    So W is indexed as W[k*n + i, r] where k is the column index of B (and result) and i is the row index of A (and result).
    
    The standard matrix multiplication tensor decomposition:
    (A*B)[i,k] = sum_{j,r} A[i,j] * B[j,k] * delta_{j,j}
    
    In tensor form, we want:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Where T[i,j,k] = 1 for the matrix multiplication tensor.
    
    With our rank-1 decomposition:
    U[i*m+j, 0] = u[i] for all j
    V[j*p+k, 0] = v[j] for all k
    W[k*n+i, 0] = w[k] for all i
    
    With u = [1,1], v = [1,1], w = [1,1,1,1,1,1,1,1]:
    
    U[0,0] = u[0] = 1 (i=0, j=0)
    U[1,0] = u[0] = 1 (i=0, j=1)
    U[2,0] = u[1] = 1 (i=1, j=0)
    U[3,0] = u[1] = 1 (i=1, j=1)
    
    V[0,0] = v[0] = 1 (j=0, k=0)
    V[1,0] = v[0] = 1 (j=0, k=1)
    ...
    V[7,0] = v[0] = 1 (j=0, k=7)
    V[8,0] = v[1] = 1 (j=1, k=0)
    ...
    V[15,0] = v[1] = 1 (j=1, k=7)
    
    W[0,0] = w[0] = 1 (k=0, i=0)
    W[1,0] = w[0] = 1 (k=0, i=1)
    W[2,0] = w[1] = 1 (k=1, i=0)
    W[3,0] = w[1] = 1 (k=1, i=1)
    ...
    W[14,0] = w[7] = 1 (k=7, i=0)
    W[15,0] = w[7] = 1 (k=7, i=1)
    
    So all entries are 1. Let me verify this gives the correct tensor:
    
    For T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    = U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0]
    = u[i] * v[j] * w[k]
    = 1 * 1 * 1 = 1 ✓
    
    So the rank-1 decomposition works with all entries equal to 1.
    """
    n, m, p = DIMENSIONS
    rank = 1
    
    # U has shape (n*m, rank)
    # U[i*m + j, r] = u[i]
    U = [[1] * rank for _ in range(n * m)]
    
    # V has shape (m*p, rank)
    # V[j*p + k, r] = v[j]
    V = [[1] * rank for _ in range(m * p)]
    
    # W has shape (p*n, rank)
    # W[k*n + i, r] = w[k]
    W = [[1] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
