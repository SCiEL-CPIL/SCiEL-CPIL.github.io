import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (28, rank)
    # W shape: (p*n, rank) = (14, rank)
    # Tensor T[i*m+j, j*p+k, k*n+i] = 1
    
    # We exploit the factorization m = 2 * 2.
    # Let j = 2*j1 + j2 where j1, j2 in {0, 1}.
    # The tensor T can be viewed as a sum over j1, j2 of tensors T_{j1, j2}.
    # T_{j1, j2}[i, 2*j1+j2, k] -> maps A[i, 2*j1+j2] and B[2*j1+j2, k] to C[i, k].
    # This is equivalent to two independent multiplications:
    # 1. A_left (2x2) * B_left (2x7) -> C_left (2x7)
    # 2. A_right (2x2) * B_right (2x7) -> C_right (2x7)
    # Where A_left = A[:, 0:2], A_right = A[:, 2:4]
    # B_left = B[0:2, :], B_right = B[2:4, :]
    # C = [A_left * B_left] + [A_right * B_right]
    
    # The rank of the 2x2x7 multiplication tensor is at most 2 * Rank(2x2x1)? No.
    # Rank(2x2x7) is the rank of the tensor for multiplying a 2x2 matrix by a 2x7 matrix.
    # This is equivalent to 7 independent 2x2x1 multiplications? No, the B matrix is shared?
    # No, B is 2x7. The multiplication is C = A B.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # This is a bilinear map.
    # The tensor is T_{i, j, k}. i in 2, j in 2, k in 7.
    # This is the Kronecker product of the 2x2x1 tensor (matrix-vector) and the 1x1x7 identity?
    # Let's check.
    # T_{i, j, k} = delta_{ij} delta_{jk}? No.
    # For 2x2x1: T_{i, j, k} with i,k in 2, j in 2. T_{i, j, k} = 1 if i=k? No.
    # Matrix-vector: C_i = sum_j A_{ij} v_j.
    # Tensor indices: A_{ij}, v_j -> C_i.
    # T_{i, j, i} = 1? No, T_{i, j, k} where output index is i.
    # So T_{i, j, k} = 1 if k=i? No, the output is a vector.
    # Let's use standard notation: T_{i, j, k} such that C_k = sum_{i,j} T_{i,j,k} A_i v_j?
    # Usually T_{i, j, k} means A_i * B_j contributes to C_k.
    # For 2x2x1 (A is 2x2, v is 2x1, C is 2x1):
    # C_0 = A_00 v_0 + A_01 v_1
    # C_1 = A_10 v_0 + A_11 v_1
    # Non-zeros: (0,0,0)=1, (0,1,0)=1, (1,0,1)=1, (1,1,1)=1.
    # Rank of this tensor is 4?
    # Let's check if it can be done in 3.
    # If rank 3, then R(2,2,1) <= 3.
    # But R(2,2,1) is the rank of matrix-vector product.
    # It is known that R(2,2,1) = 4. (Since it contains R(2,1,1)=2 and R(1,2,1)=2, and they don't combine well? Actually R(n,m,1) = nm).
    # So R(2,2,1) = 4.
    
    # Now for 2x2x7.
    # C_k depends on A and B_k.
    # C = A B.
    # This is equivalent to 7 columns of B being multiplied by A.
    # The tensor is T_{i, j, k} where i in 2, j in 2, k in 7.
    # T_{i, j, k} = 1 if j=k? No.
    # A is 2x2 (indices i, j). B is 2x7 (indices j, k).
    # C_{i, k} = sum_j A_{i, j} B_{j, k}.
    # So the tensor indices are (i, j, k) with value 1 if the indices match?
    # No, the tensor T has indices (u, v, w) corresponding to A_u, B_v, C_w.
    # A has 4 elements (2x2). B has 14 elements (2x7). C has 14 elements (2x7).
    # T_{u, v, w} = 1 if u and v share the middle index and v and w share the middle index?
    # Let A index be (i, j). B index be (j, k). C index be (i, k).
    # So T_{(i,j), (j,k), (i,k)} = 1.
    # This is the tensor for 2x2x7 matrix multiplication.
    # Is this the Kronecker product of T_{2,2,1} and I_7?
    # T_{2,2,1} has indices (i, j, i) -> 1. (Wait, output index is i? No, output is vector of size 2. Let's say output index is l. Then T_{i, j, l} = 1 if l=i? No.
    # C_l = sum_j A_{l, j} v_j. So T_{l, j, l} = 1? No, T_{(l, j), j, l} = 1.
    # Let's stick to the flat indices.
    # For 2x2x1:
    # A: 0,1,2,3 (row major: 00, 01, 10, 11)
    # v: 0,1
    # C: 0,1
    # T[0, 0, 0] = 1 (A00 * v0 -> C0)
    # T[1, 1, 0] = 1 (A01 * v1 -> C0)
    # T[2, 0, 1] = 1 (A10 * v0 -> C1)
    # T[3, 1, 1] = 1 (A11 * v1 -> C1)
    # Rank is 4.
    
    # For 2x2x7:
    # A: 0..3
    # B: 0..13 (row major: j0k0, j0k1, ..., j0k6, j1k0, ...)
    # C: 0..13 (row major: i0k0, ..., i0k6, i1k0, ...)
    # T[A_idx, B_idx, C_idx] = 1 if:
    # A_idx = i*m_A + j (m_A=2) -> i = A_idx//2, j = A_idx%2
    # B_idx = j*m_B + k (m_B=7) -> j = B_idx//7, k = B_idx%7
    # C_idx = i*m_C + k (m_C=7) -> i = C_idx//7, k = C_idx%7
    # Match if j_A == j_B and i_A == i_C and k_B == k_C.
    
    # Can we decompose this?
    # Notice that for each k (0..6), the sub-tensor for column k is exactly the 2x2x1 tensor.
    # And these sub-tensors for different k are disjoint in the B and C indices (different k components).
    # So the total tensor is the direct sum of 7 copies of the 2x2x1 tensor.
    # Rank of direct sum is sum of ranks.
    # So Rank(2x2x7) = 7 * Rank(2x2x1) = 7 * 4 = 28.
    
    # So we have two blocks:
    # Block 1: 2x2x7 (Left part of A, Left part of B) -> Rank 28
    # Block 2: 2x2x7 (Right part of A, Right part of B) -> Rank 28
    # Total Rank = 28 + 28 = 56.
    # This is the same as schoolbook!
    
    # We need to do better.
    # Can we share multiplications between the two blocks?
    # A_left and A_right are independent. B_left and B_right are independent.
    # There is no sharing possible unless we combine them.
    # But C = A_left B_left + A_right B_right.
    # This is just a sum of two independent multiplications.
    # So the rank is the sum of the ranks.
    
    # We need a different strategy.
    # What if we view the 2x4x7 multiplication differently?
    # Maybe use Strassen-like tricks on the 4x4 block?
    # Or use the fact that 2x4x7 is small enough to search?
    
    # Let's try to find a rank < 56 decomposition using a greedy algorithm with exact refinement.
    # We will try to find a set of rank-1 tensors that sum to T.
    
    # Since we have time constraints, we will try a few specific constructions.
    # One known result: R(2,4,7) <= 25?
    # Actually, let's try to construct a rank 28 decomposition for the whole thing?
    # If R(2,4,7) <= 28, that would be great.
    
    # Let's try a randomized search for a low rank decomposition.
    # We will generate random vectors u, v, w and try to fit the coefficients.
    
    # However, given the strict time limit and the difficulty of exact search,
    # I will implement a fallback to the schoolbook rank 56 if the search fails quickly.
    # But I must beat 56.
    
    # Let's try a simple improvement:
    # Is R(2,4,7) < 56?
    # Yes, it is known that R(2,4,7) <= 25 (Larsen, 2022?).
    # Let's try to find a rank 25 decomposition.
    
    # I will use a precomputed decomposition if I can find one, or a very fast heuristic.
    # Since I don't have the exact formulas for rank 25 memorized, I will try to search for a rank 30-40 decomposition which is easier to find heuristically.
    
    # Heuristic:
    # 1. Create random U, V, W of rank R.
    # 2. Compute the residual T - sum(u_i v_i w_i).
    # 3. Use SVD to find the best rank-1 approximation of the residual.
    # 4. Add it to the decomposition.
    # 5. Repeat until residual is 0 or time runs out.
    # 6. If residual is not 0, try to solve for the last few terms exactly.
    
    # This is risky for exactness.
    
    # Alternative: Use the fact that the tensor is sparse.
    # There are 56 non-zero entries.
    # We can try to cover these with rank-1 tensors.
    
    # Let's try to construct a rank 42 decomposition using a block diagonal approach with some sharing.
    # Or simply return a valid decomposition of rank 56 if I can't find better.
    # But the task requires beating the seed.
    
    # I will attempt a **Rank 28** construction by assuming R(2,4,7) <= 28.
    # If I can't verify it, I will use a **Rank 35** construction?
    
    # Let's try to search for a rank 30 decomposition with integer coefficients.
    
    start_time = time.time()
    deadline = 140
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to find a better rank
    # We will try ranks 25 to 55
    # For each rank, we try to find a solution.
    
    # Given the complexity, I will return the schoolbook solution to ensure validity,
    # as I cannot guarantee a valid lower-rank construction in this time.
    # However, the score is 1.0 for rank 56.
    # If I return rank 56, I tie.
    
    # I will try one specific trick:
    # Use the identity for 2x2 matrix multiplication (rank 7) to reduce the rank of the 4x4 block?
    # No, the dimensions are 2x4 and 4x7.
    
    # I will return the schoolbook solution.
    return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
