"""Program for 2x2 by 2x10 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    # Output C is n x p = 2 x 10, index W[k*n+i, r] for k in 0..p-1, i in 0..n-1
    # Input A is n x m = 2 x 2, index U[i*m+j, r] for i in 0..n-1, j in 0..m-1
    # Input B is m x p = 2 x 10, index V[j*p+k, r] for j in 0..m-1, k in 0..p-1
    
    # We use a Strassen-like approach. For 2x2 * 2x10:
    # Standard schoolbook rank is 2*2*10 = 40.
    # 
    # Let's think about this differently. The tensor is:
    # T = sum_{i,j,k} e_i e_j^T (A part) * e_j e_k^T (B part) * (e_k e_i^T) (C part)
    # Actually, C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # For fixed k, the 2x2 matrix A times column k of B gives column k of C.
    # So we have 10 independent 2x2 * 2x1 multiplications.
    #
    # Each 2x2 * 2x1 multiplication: C_col = A * B_col where A is 2x2, B_col is 2x1.
    # C_col[0] = A[0,0]*B_col[0] + A[0,1]*B_col[1]
    # C_col[1] = A[1,0]*B_col[0] + A[1,1]*B_col[1]
    #
    # This is a bilinear map from (2x2) x (2x1) -> (2x1), rank at most 2*2*1=4.
    # Can we do better? The tensor for this is:
    # sum_{i,j} e_i (A part, row i) * e_j (B part) * e_i (C part) * (something for j)
    # More precisely: C[i] = sum_j A[i,j] * B[j]
    # Tensor: sum_{i,j} a_{i,j} * b_j * c_i where c_i is the i-th component of output.
    # This is a 2x2x2 tensor (A has 4 entries, B has 2 entries, C has 2 entries... wait)
    # Actually A is 2x2 (4 entries), B is 2x1 (2 entries), C is 2x1 (2 entries).
    # The tensor is in (R^4) * (R^2) * (R^2), and the multiplication is:
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    #
    # This is just the standard matrix-vector product for 2x2 * 2x1.
    # Rank is 4 (schoolbook), and it's known that rank of 2x2 matrix-vector is 4.
    # Actually, let's verify: the tensor is sum_{i=0}^1 sum_{j=0}^1 e_{i,j}^{(A)} ⊗ e_j^{(B)} ⊗ e_i^{(C)}
    # where e_{i,j}^{(A)} is the (i,j)-th basis vector in R^4.
    # This tensor has rank 4 (it's a direct sum of two rank-2 tensors, one for each i).
    # So for each column k, we need rank 4.
    #
    # But wait - the columns are not independent! The same A multiplies all columns.
    # So we can't just sum ranks.
    #
    # Total tensor: sum_{k=0}^{9} sum_{i=0}^{1} sum_{j=0}^{1} e_{i,j}^{(A)} ⊗ e_{j,k}^{(B)} ⊗ e_{i,k}^{(C)}
    # where e_{j,k}^{(B)} is the (j,k)-th basis in R^{2x10} and e_{i,k}^{(C)} is (i,k)-th in R^{2x10}.
    #
    # Can we factor this? Let's group by j:
    # = sum_{j=0}^{1} sum_{i=0}^{1} sum_{k=0}^{9} e_{i,j}^{(A)} ⊗ e_{j,k}^{(B)} ⊗ e_{i,k}^{(C)}
    #
    # For fixed j, this is: sum_{i=0}^{1} sum_{k=0}^{9} e_{i,j}^{(A)} ⊗ e_{j,k}^{(B)} ⊗ e_{i,k}^{(C)}
    # = sum_{i=0}^{1} e_{i,j}^{(A)} ⊗ (sum_{k=0}^{9} e_{j,k}^{(B)} ⊗ e_{i,k}^{(C)})
    #
    # The inner term sum_{k} e_{j,k}^{(B)} ⊗ e_{i,k}^{(C)} is a 2-term... no, 10-term rank-10 tensor?
    # No, sum_{k=0}^{9} e_{j,k}^{(B)} ⊗ e_{i,k}^{(C)} has rank 10 (each k gives a different pair).
    #
    # So for fixed j: e_{i,j}^{(A)} ⊗ (rank-10 tensor) for each i.
    # The rank of the whole thing for fixed j is 2 * 10 = 20? No, that's not right either.
    #
    # Let me reconsider. The total rank is at most 40 (schoolbook).
    # 
    # Key insight: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # For each i, C[i,:] is a linear combination of rows 0 and 1 of B.
    # 
    # Define: for each i, let's define new "B-like" vectors.
    # Actually, let's use the following decomposition:
    # 
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #
    # We can write this as 4 products for each k, but the A entries are shared across k.
    # 
    # Alternative: use 4 "intermediate" vectors for each of the 4 A entries:
    # For A[0,0]: contributes A[0,0]*B[0,k] to C[0,k] for all k
    # For A[0,1]: contributes A[0,1]*B[1,k] to C[0,k] for all k
    # For A[1,0]: contributes A[1,0]*B[0,k] to C[1,k] for all k
    # For A[1,1]: contributes A[1,1]*B[1,k] to C[1,k] for all k
    #
    # So we have 4 "groups", each group has 10 products (one per k).
    # Total = 4 * 10 = 40. That's schoolbook.
    #
    # Can we do better? The structure is:
    # C[0,:] = A[0,0] * B[0,:] + A[0,1] * B[1,:]
    # C[1,:] = A[1,0] * B[0,:] + A[1,1] * B[1,:]
    #
    # This is essentially two independent matrix-vector products (one for row 0, one for row 1),
    # each of which is a 1x2 matrix times a 2x10 matrix.
    #
    # The tensor for row 0: sum_{j,k} e_j^{(A_row0)} ⊗ e_{j,k}^{(B)} ⊗ e_k^{(C_row0)}
    # This is a (2) x (2*10) x (10) tensor. Rank is 2*10 = 20.
    # Similarly for row 1: rank 20.
    # Total: 40.
    #
    # But the A entries are in the same A matrix, so there might be sharing.
    # A[0,0], A[0,1], A[1,0], A[1,1] are all in the same A.
    #
    # Let's try to use Strassen-like ideas. For the 2x2 part, Strassen uses 7 products
    # instead of 4. But here the "B" part is 2x10, not 2x2.
    #
    # Strassen for 2x2 * 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # In our case, B is 2x10, so B11, B12, etc. are 1x10 vectors.
    # The 7 "products" M1..M7 would each be a product of a scalar (from A) and a 1x10 vector (from B).
    # Each such product has rank 10 (since it's scalar * vector, giving a 1x10 result).
    # So total rank would be 7 * 10 = 70, which is worse than 40!
    #
    # So Strassen doesn't help here because the "B" part is too wide.
    #
    # Let's think differently. The problem is that each "product" in Strassen involves
    # a scalar from A times a row of B (1x10 vector), which has rank 10.
    #
    # Alternative approach: since the inner dimension is only 2, and the output dimension is 10,
    # maybe we can use the fact that for each output position k, we only need 2 products.
    #
    # Actually, I think the minimum rank might be 40. Let's verify this is optimal.
    #
    # The tensor is T = sum_{i,j,k} a_{ij} b_{jk} c_{ik}
    # with i in {0,1}, j in {0,1}, k in {0,...,9}.
    #
    # This can be written as: for each (i,j), the term is a_{ij} * (sum_k b_{jk} c_{ik})
    # The inner sum_k b_{jk} c_{ik} is a rank-10 tensor (outer product of row j of B and row i of C,
    # but indexed as b_{jk} and c_{ik}).
    #
    # Wait, b_{jk} is the (j,k) entry of B, and c_{ik} is the (i,k) entry of C.
    # sum_k b_{jk} c_{ik} is not a standard outer product.
    #
    # Let me redefine. The tensor is:
    # T = sum_{i=0}^1 sum_{j=0}^1 sum_{k=0}^9 a_{ij} ⊗ b_{jk} ⊗ c_{ik}
    #
    # For fixed i,j: sum_{k=0}^9 a_{ij} ⊗ b_{jk} ⊗ c_{ik}
    # = a_{ij} ⊗ (sum_{k=0}^9 b_{jk} ⊗ c_{ik})
    #
    # The term sum_{k=0}^9 b_{jk} ⊗ c_{ik} is a rank-10 tensor in (B_space ⊗ C_space).
    # 
    # So T = sum_{i=0}^1 sum_{j=0}^1 a_{ij} ⊗ T_{ij} where T_{ij} has rank 10.
    #
    # The total rank is at most 4 * 10 = 40.
    #
    # Can we do better? The question is whether the T_{ij} for different (i,j) can be shared.
    # T_{00} = sum_k b_{0k} ⊗ c_{0k}
    # T_{01} = sum_k b_{1k} ⊗ c_{0k}
    # T_{10} = sum_k b_{0k} ⊗ c_{1k}
    # T_{11} = sum_k b_{1k} ⊗ c_{1k}
    #
    # Notice: T_{00} + T_{11} = sum_k (b_{0k}⊗c_{0k} + b_{1k}⊗c_{1k})
    # T_{01} + T_{10} = sum_k (b_{1k}⊗c_{0k} + b_{0k}⊗c_{1k})
    #
    # These are different tensors. It's not clear that we can reduce the rank.
    #
    # I'll go with the schoolbook decomposition of rank 40, which is likely optimal for this case.
    #
    rank = n * m * p  # 2 * 2 * 10 = 40
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
# EVOLVE-BLOCK-END
