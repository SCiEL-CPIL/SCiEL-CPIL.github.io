"""Program for rank-minimal tensor decomposition of 4x5 * 5x5 matrix multiplication."""
DIMENSIONS = [4, 5, 5]
import numpy as np
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Start with schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=int)
    V_school = np.zeros((m * p, rank_school), dtype=int)
    W_school = np.zeros((p * n, rank_school), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try WFTA-based decomposition
    # Partition into 2x2 blocks where possible
    # 4x5 * 5x5: 
    #   A is 4x5: 2 rows of 2x2 blocks + 1 row of 2x1
    #   B is 5x5: 2 cols of 2x2 blocks + 1 col of 2x1, etc.
    
    # For 2x2 block multiplication using WFTA:
    # We need 2x2 * 2x2 which normally takes 8 multiplications
    # WFTA can reduce this to fewer operations
    
    # Actually, let's think about this more carefully.
    # The standard approach for 2x2 matrix multiplication:
    # C = A * B where A, B are 2x2
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # This is 8 scalar multiplications.
    
    # Winograd's algorithm for 2x2 can reduce this.
    # The key insight is to use a transform approach.
    
    # Let's try a different approach: use the fact that we can combine
    # some of the terms.
    
    # For now, let's try to find a decomposition with rank < 100
    # by looking for linear dependencies in the tensor.
    
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 if and only if
    # the (i,j) entry of A times the (j,k) entry of B contributes to the (i,k) entry of C.
    
    # Let's build the tensor and try to find its rank using numerical methods.
    
    # Build the tensor as a 3D array
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    # Reshape to a matrix for SVD
    # T[i,j,k] can be seen as a matrix of shape (n*m, p) or (n, m*p) etc.
    # Let's use (n*m, p)
    T_mat = T.reshape(n * m, p)
    
    # Compute SVD to get numerical rank estimate
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(T_mat, full_matrices=False)
        # Count significant singular values
        tol = 1e-10
        num_significant = np.sum(s_vals > tol)
        print(f"Numerical rank estimate from SVD: {num_significant}")
    except Exception as e:
        print(f"SVD failed: {e}")
        num_significant = rank_school
    
    # The numerical rank gives a lower bound on the tensor rank
    # But we need to find an actual decomposition
    
    # Let's try to use the SVD components to build a decomposition
    # T_mat ≈ sum_r u_r * s_r * v_r^T
    # This gives us a decomposition where:
    # U[:, r] = u_r (scaled), V[:, r] = v_r (scaled), W[:, r] = 1/s_r
    
    # But this only works if the tensor has a special structure where
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    
    # For matrix multiplication, T[i,j,k] = 1 for all i,j,k
    # So T is actually the all-ones tensor!
    
    # Wait, that's not right. Let me reconsider.
    # The matrix multiplication tensor is defined such that:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 if and only if the product A[i,j]*B[j,k]
    # contributes to C[i,k].
    
    # Actually, the standard definition is that T[i,j,k] = 1 for all i,j,k
    # because every combination (i,j,k) contributes to the product.
    
    # So T is indeed the all-ones tensor of shape (n,m,p).
    # The rank of the all-ones tensor is 1!
    
    # But wait, that can't be right for matrix multiplication.
    # Let me reconsider the definition.
    
    # The matrix multiplication tensor is:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * δ_{i,k} (no, that's not right either)
    
    # Actually, the correct definition is:
    # T[i,j,k] = 1 if and only if the (i,j) element of A and the (j,k) element of B
    # multiply to contribute to the (i,k) element of C.
    
    # For standard matrix multiplication, T[i,j,k] = 1 for ALL i,j,k.
    # So T is the all-ones tensor.
    
    # But the rank of the all-ones tensor is 1, which means we can write:
    # T = 1 ⊗ 1 ⊗ 1 (outer product of three all-ones vectors)
    
    # This would give a rank-1 decomposition, which is clearly wrong for matrix multiplication.
    
    # I think I'm confusing the tensor definition. Let me look at this more carefully.
    
    # The matrix multiplication tensor is actually:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k
    # where e_i, e_j, e_k are basis vectors.
    
    # No wait, that's still the all-ones tensor.
    
    # I think the issue is that the tensor should be:
    # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C
    # which is always true, so T[i,j,k] = 1 for all i,j,k.
    
    # But then the rank is 1, which contradicts known results.
    
    # Let me reconsider. The tensor decomposition for matrix multiplication is:
    # (A*B)[i,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] (summed over r)
    
    # For schoolbook:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So we have U[i,j,r] = A[i,j] when r=(i,j,k) for a specific k, but that doesn't make sense.
    
    # I think the correct formulation is:
    # We want to find U, V, W such that:
    # (A*B)[i,k] = sum_{j,r} A[i,j] * B[j,k] * (some combination)
    
    # Actually, the standard formulation is:
    # Define the tensor T such that T[i,j,k] = 1 if and only if the scalar product
    # A[i,j] * B[j,k] contributes to C[i,k].
    
    # For matrix multiplication, this is always true, so T[i,j,k] = 1 for all i,j,k.
    
    # The tensor decomposition is:
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r ∈ R^{n*m}, v_r ∈ R^{m*p}, w_r ∈ R^{p*n}
    
    # And the condition is that:
    # (A*B)[i,k] = sum_r (sum_j A[i,j] * U[i,j,r]) * (sum_k B[j,k] * V[j,k,r]) * W[i,k,r]
    
    # No, that's not right either.
    
    # Let me use the correct definition from the literature:
    # The matrix multiplication tensor is:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k
    # where e_i is the i-th standard basis vector in R^n, etc.
    
    # Wait, that's still the all-ones tensor.
    
    # I think I need to be more careful. The tensor for matrix multiplication is:
    # T ∈ R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}
    # with T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise.
    
    # Yes! That's the key. The tensor is NOT the all-ones tensor.
    # It has 1's only at positions where the indices are consistent.
    
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k.
    
    # This is a sparse tensor with n*m*p non-zero entries.
    
    # Now the schoolbook decomposition is:
    # For each (i,j,k), define:
    # U[:, (i,j,k)] has 1 at position i*m+j
    # V[:, (i,j,k)] has 1 at position j*p+k
    # W[:, (i,j,k)] has 1 at position k*n+i
    
    # And T = sum_{i,j,k} U[:,(i,j,k)] ⊗ V[:,(i,j,k)] ⊗ W[:,(i,j,k)]
    
    # This is the schoolbook decomposition with rank n*m*p.
    
    # Now we want to find a decomposition with lower rank.
    
    # Let's build the actual tensor and try to find its rank.
    
    T_actual = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_actual[i * m + j, j * p + k, k * n + i] = 1.0
    
    # The rank of this tensor is what we're looking for.
    # We can estimate it using various methods.
    
    # One approach: use the fact that the tensor can be viewed as a matrix
    # by flattening two dimensions.
    
    # View as (n*m, m*p * p*n) matrix
    T_mat1 = T_actual.reshape(n * m, m * p * p * n)
    U1, s1, Vt1 = np.linalg.svd(T_mat1, full_matrices=False)
    rank_est1 = np.sum(s1 > 1e-10)
    
    # View as (n*m * m*p, p*n) matrix
    T_mat2 = T_actual.reshape(n * m * m * p, p * n)
    U2, s2, Vt2 = np.linalg.svd(T_mat2, full_matrices=False)
    rank_est2 = np.sum(s2 > 1e-10)
    
    print(f"Rank estimates: {rank_est1}, {rank_est2}")
    
    # The minimum of these gives a lower bound on the tensor rank.
    lower_bound = min(rank_est1, rank_est2)
    print(f"Lower bound on tensor rank: {lower_bound}")
    
    # For now, return the schoolbook decomposition as it's guaranteed correct.
    # A proper search for lower rank would require more sophisticated methods.
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
