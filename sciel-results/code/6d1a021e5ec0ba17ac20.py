import time

DIMENSIONS = [3, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    # Use the rank-19 decomposition for 3x3 matrix multiplication
    # to build a rank-57 decomposition for 3x5 * 5x5 via block partitioning.
    # Partition: A (3x5) into columns (2,3), B (5x5) into (5x2, 5x3)
    # C = A1*B12 + A2*B23
    # Use the 3x3 rank-19 decomposition where:
    #   X (3x3), Y (3x3), Z = X*Y
    #   Map: X[i,j] -> A[i, j] for j<3, and we handle the 5th column separately
    #
    # Actually, let's use a cleaner approach:
    # Partition into two blocks along the middle dimension:
    #   Block 1: A[:, 0:2] (3x2) * B[0:2, :] (2x5) -> 3x5
    #   Block 2: A[:, 2:5] (3x3) * B[2:5, :] (3x5) -> 3x5
    #
    # For Block 1 (3x2 * 2x5): rank <= 3*2*5 = 30, but we can do better
    #   Actually 3x2*2x5 has rank at most 3*5 = 15 if we use the structure
    #   But the tensor rank of (3,2,5) matmul is at most 3*2*5=30
    #   We can use: for each of 3 rows and 5 cols, sum over 2 middle -> 30 terms
    #   But we can group: rank is at most min(3*2, 2*5, 3*5) = 10? No.
    #   The rank of the (3,2,5) matrix multiplication tensor is exactly 3*2*5 / max(3,2,5)... 
    #   Actually it's known that rank of (n,m,p) matmul is at most n*m*p and at least max(n*m, m*p, n*p)
    #   For (3,2,5): at least max(6,10,15)=15, at most 30.
    #   Let's just use schoolbook for this block: 3*2*5 = 30 terms... that's too many.
    #
    # Better approach: use the full 3x3 rank-19 decomposition for the 3x3 part,
    # and handle the remaining 3x2 * 2x5 part.
    #
    # 3x2 * 2x5: We can write this as sum over k=0,1 of A[:,k] * B[k,:]
    # Each A[:,k] is 3x1, B[k,:] is 1x5. The outer product has rank 1.
    # So 3x2*2x5 = sum_{k=0}^{1} (A[:,k]) * (B[k,:]) which is 2 terms if we treat
    # A[:,k] as a 3-vector and B[k,:] as a 5-vector. But in tensor form:
    # C[i, j] = sum_k A[i,k] * B[k,j]
    # This is a (3, 2, 5) tensor. Its rank is at most 3*5 = 15 (fix i, j, sum over k)
    # Actually rank <= 2*15 = 30 but we can do 15 by: for each (i,j), C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    # That's 15 rank-1 terms (one for each i,j pair with k summed).
    # Wait no: each (i,j) needs 2 terms, so 3*5*2 = 30. 
    # But we can factor: C = A1 * B1 where A1 is 3x2 and B1 is 2x5.
    # The rank-1 decomposition: C[i,j] = sum_k A[i,k] B[k,j]
    # = sum_{i,j,k} A[i,k] B[k,j] e_{i,j}
    # This is already in rank-1 form with 3*2*5 = 30 terms.
    # Can we do better? The tensor rank of (3,2,5) matmul... 
    # For rectangular, rank(n,m,p) >= max(nm, mp, np) and <= nmp
    # rank(3,2,5) >= max(6, 10, 15) = 15, <= 30
    # We can achieve 15: group by k. For each k, we have a (3,1,5) tensor which is 
    # a rank-5 tensor (sum of 5 rank-1 terms). So 2*5 = 10? No.
    # For fixed k, C_k[i,j] = A[i,k]*B[k,j]. This is an outer product of A[:,k] (3x1) 
    # and B[k,:] (1x5), which is a 3x5 matrix of rank 1. As a tensor in (i,j,k) space,
    # fixing k gives a (3,5) matrix of rank 1, which is 5 rank-1 terms in (i,j).
    # So total: 2 * 5 = 10 rank-1 terms? Let me verify:
    # C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    # Term for k=0: A[i,0]*B[0,j] = sum over i,j of (A[i,0])(B[0,j]) * 1
    # This is rank-1 in the (i,j) space for each k, so 2 rank-1 terms total? 
    # No wait - in the tensor decomposition C[i,j] = sum_r U[i,r]*V[r,j]*W[r,...] 
    # Hmm, let me think in the (U,V,W) format.
    # C[i,j] = sum_k sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]... 
    # No, the convention is W[k*n+i, r] for the output.
    # Actually for matmul C = A*B where A is n x m, B is m x p, C is n x p:
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # In tensor form: T[i, k, j] = A[i,k]*B[k,j]
    # Decomposition: T[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # 
    # For (3,2,5): n=3, m=2, p=5
    # We can use rank 10: for each of the 5 output columns j, we have a 3x2 matmul with 
    # a 2-vector, which is rank 2 per column, so 5*2 = 10.
    # 
    # For (3,3,5): n=3, m=3, p=5
    # We can use the 3x3 rank-19 for each... no, that doesn't directly work.
    # rank(3,3,5) >= max(9, 15, 15) = 15, <= 45
    # We can do: for each of 5 output columns, 3x3 matmul with rank 19? No.
    # For fixed j, C[:,j] = A * B[:,j] which is 3x3 * 3x1 = 3-vector.
    # This is a (3,3,1) tensor which has rank at most 9.
    # So 5 * 9 = 45 total, or better: 
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # = sum_{k,j} A[i,k]*B[k,j]
    # Group by i: for each i, C[i,:] = A[i,:] * B which is 1x3 * 3x5 = 1x5
    # This is rank 5 per row, so 3*5 = 15 total rank-1 terms!
    # Let me verify: C[i,j] = sum_k A[i,k]*B[k,j]
    # For each (i,j): one term per k. Total 3*3*5 = 45 schoolbook.
    # But we can factor: C[i,j] = (A[i,0]*B[0,j] + A[i,1]*B[1,j] + A[i,2]*B[2,j])
    # The tensor C[i,j] as a function of i and j for fixed k-sum...
    # Actually the rank-1 decomposition in the (U,V,W) format:
    # C[i,j] = sum_r U[i*m+k, r]*V[k*p+j, r]*W[j*n+i, r]
    # The sum is over both k (implicit in U, V indices) and r.
    # So rank r means r terms, each term picks specific (i,k) for U, (k,j) for V, (j,i) for W.
    # For the (3,3,5) block: we need rank such that we can express all 45 C[i,j] entries.
    # Minimum rank is at least 15 (from max(n*p, m*p) = max(15,15) = 15).
    # We can achieve 15: for each pair (i,j), we need sum_k A[i,k]*B[k,j].
    # If we use 15 terms, each term must cover multiple (i,j) pairs.
    # Term r: U picks (i_r, k_r), V picks (k_r, j_r), W picks (j_r, i_r).
    # This gives C[i_r, j_r] += U*V*W.
    # To get C[i,j] = sum_k A[i,k]*B[k,j], we need for each (i,j) the sum over k.
    # With 15 terms we can't do this in general unless there's structure.
    # Actually 15 is a lower bound, not necessarily achievable.
    # Let's just use schoolbook for simplicity: 3*3*5 = 45 terms for this block.
    # Total: 10 + 45 = 55. That's better than 75!
    #
    # Wait, let me reconsider. The whole 3x5*5x5:
    # Schoolbook: 3*5*5 = 75.
    # Block approach:
    #   Block 1: A[:, 0:2] (3x2) * B[0:2, :] (2x5) -> rank 10
    #   Block 2: A[:, 2:5] (3x3) * B[2:5, :] (3x5) -> rank ?
    #
    # For Block 2 (3,3,5): Let me use rank 25 (5 output columns, each needing 3x3*3x1 = rank 9? No)
    # Actually let me just carefully count.
    # C2[i,j] = sum_{k=0}^{2} A2[i,k] * B2[k,j]
    # Schoolbook for this: 3*3*5 = 45 terms.
    # Total: 10 + 45 = 55.
    #
    # Can we do better on Block 2? Use the fact that it's 3x3*3x5.
    # For each of the 5 columns j of B2, we compute A2 * B2[:,j] (3x3 * 3x1).
    # This is a (3,3,1) tensor with rank at most 9.
    # So 5*9 = 45. Same as schoolbook.
    # 
    # What if we use rank-19 for 3x3? That applies when both are 3x3.
    # Here B2 is 3x5, not 3x3.
    #
    # Alternative: partition B into (3x3, 3x2):
    #   Block A: A (3x5) * B1 (5x3) -> 3x3, use rank 19
    #   Block B: A (3x5) * B2 (5x2) -> 3x2
    # For Block B (3,5,2): rank >= max(15, 10, 6) = 15, <= 30
    # Schoolbook: 3*5*2 = 30. Can we do 15?
    # C[i,j] = sum_k A[i,k]*B[k,j] for k=0..4, j=0..1
    # For each j: 3x5 * 5x1 = 3-vector, rank at most 15 per j, so 2*15=30.
    # Or: for each i: 1x5 * 5x2 = 1x2, rank 2 per i, so 3*2 = 6? 
    # No, that's not how it works. The rank of the (3,5,2) tensor...
    # Lower bound is 15, upper is 30. Let's use 30 (schoolbook).
    # Total: 19 + 30 = 49! That's much better.
    #
    # Let me verify this partition:
    # B is 5x5. Split into B_left (5x3) and B_right (5x2).
    # C = A * B = A * [B_left | B_right] = [A*B_left | A*B_right]
    # A*B_left: 3x5 * 5x3 = 3x3. Use rank-19 decomposition.
    # A*B_right: 3x5 * 5x2 = 3x2. Use schoolbook: 3*5*2 = 30 terms.
    # Total rank: 19 + 30 = 49.
    #
    # Now I need the rank-19 decomposition of 3x3 matmul.
    # The standard rank-19 algorithm (Bini, Capovani, Lotti, Romani 1980):
    # 
    # M1  = (A11 + A22) * (B11 + B22)
    # M2  = (A11 - A22) * (B11 - B22)
    # M3  = (A12 + A22) * (B21 - B22)
    # M4  = (A11 + A12) * (B11 - B12)
    # M5  = (A21 + A11) * (B12 - B11)
    # M6  = (A22 + A21) * (B21 - B11)
    # M7  = (A11 + A12 + A21 + A22) * (B11 - B12 - B21 + B22)
    # M8  = (A12 + A21) * (B11 + B22)
    # M9  = (A12 - A21) * (B12 + B21)
    # M10 = A11 * B12
    # M11 = A12 * B22
    # M12 = A21 * B11
    # M13 = A22 * B21
    # M14 = (A11 - A22) * (B12 - B21)
    # M15 = (A11 - A12) * (B21 - B11)
    # M16 = (A11 - A21) * (B11 - B12)
    # M17 = (A22 - A12) * (B21 - B22)
    # M18 = (A22 - A21) * (B11 - B22)
    # M19 = (A11 + A12 - A21 - A22) * (B11 - B12 + B21 - B22)
    #
    # Wait, I need to get the exact formulas right. Let me use a known correct rank-19.
    # 
    # Actually, let me use a simpler approach. I'll verify with a known rank-19 decomposition.
    # Here's one from the literature:
    #
    # c11 = M1 + M2 + M3 - M4 + M5 + M6 + M7 + M8 + M9 + M10 - M11 + M12 - M13 + M14 + M15 + M16 - M17 + M18 + M19
    # ... this is getting complex. Let me just hardcode a verified rank-19.

    # Let me use a different, simpler strategy that I can verify:
    # Partition 3x5*5x5 as:
    #   Use 5 blocks of 3x1*1x5 (each rank 1): total 5*3*5 = 75 (schoolbook)
    # 
    # Or: 3 blocks of 1x5*5x5 (each is 1x5*5x5 = 1x5, rank 5): total 3*5 = 15? No.
    # For fixed i: C[i,:] = A[i,:] * B, which is (1x5)*(5x5) = 1x5.
    # This is a (1,5,5) tensor with rank 5.
    # So total rank = 3 * 5 = 15!
    #
    # Let me verify: C[i,j] = sum_k A[i,k]*B[k,j]
    # For fixed i, C[i,:] is a 1x5 row vector = A[i,:] * B
    # A[i,:] is 1x5, B is 5x5. The product is 1x5.
    # As a tensor T[i, k, j] = A[i,k]*B[k,j], for fixed i this is rank 5 
    # (since it's an outer product of a 5-vector and a 5x5 matrix... no).
    # Actually T[i,k,j] for fixed i: sum over k of A[i,k]*B[k,j].
    # This is a bilinear form in (k,j). The matrix is diag(A[i,:]) * B.
    # The rank of this as a (k,j) matrix is at most 5.
    # So for each i, we need 5 rank-1 terms in the (k,j) space.
    # But in the full tensor (i,k,j), each such term also fixes i.
    # So total: 3 * 5 = 15 rank-1 terms.
    #
    # Let me construct this:
    # For each i (0,1,2) and each "rank-1 component" r (0..4):
    #   We need to express A[i,:] * B as a sum of 5 rank-1 terms.
    #   A[i,:] * B = sum_{k=0}^{4} A[i,k] * B[k,:]
    #   Each term A[i,k] * B[k,:] is a scalar times a row of B.
    #   In the tensor decomposition: 
    #     U[i*m+k, r_total] = A[i,k]... but we need constants, not variable values.
    #
    # Wait, the tensor decomposition uses constant coefficients. The variables A and B are "plugged in".
    # The tensor T[i,k,j] = delta_{i,k,j} is the identity tensor for matmul.
    # T[i,k,j] = 1 if we're computing C[i,j] = sum_k A[i,k]*B[k,j], the coefficient is 1 for each (i,k,j).
    # So the rank-1 decomposition is: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # where U, V, W are constant matrices.
    #
    # For the "fix i" approach:
    # For each i and each k, we have one term: U[i*m+k, r] = 1, V[k*p+j, r] = 1 for all j, W[j*n+i, r] = 1 for all j.
    # But that's 3*5 = 15 terms, each with V having 5 non-zero entries and W having 5 non-zero entries.
    # Wait no - in a rank-1 term, U has ONE non-zero entry, V has ONE non-zero entry, W has ONE non-zero entry.
    # So each rank-1 term corresponds to a specific (i,k,j) triple.
    # That gives us 3*5*5 = 75 terms (schoolbook).
    #
    # To get fewer terms, we need U, V, W to have multiple non-zero entries in each rank-1 term.
    # E.g., if U[i*m+k1, r] = 1 and U[i*m+k2, r] = 1, then the term covers two k values.
    #
    # So the "fix i, sum over k" idea: for fixed i, C[i,j] = sum_k A[i,k]*B[k,j]
    # = A[i,0]*B[0,j] + A[i,1]*B[1,j] + ... + A[i,4]*B[4,j]
    # This is 5 rank-1 terms in the (k,j) subspace, but each term has U fixed to row i.
    # In the full tensor: term r (for fixed i, component c of the 5):
    #   U: non-zero at (i, k_c) for some k_c... 
    #   Actually, the 5 terms for fixed i are: for each k, U[i*m+k, r] = 1, V[k*p+j, r] = 1, W[j*n+i, r] = 1.
    #   But V[k*p+j, r] = 1 for ALL j means V has 5 non-zero entries in column r.
    #   Similarly W[j*n+i, r] = 1 for all j means W has 5 non-zero entries.
    #   So this is NOT a rank-1 term in the standard sense.
    #
    # I'm confusing myself. Let me restart with the clear definition.
    #
    # The matmul tensor T is in R^{(n*m) x (m*p) x (p*n)}.
    # T[a, b, c] = 1 if a = i*m+k, b = k*p+j, c = j*n+i for some i,j,k; else 0.
    # A rank-1 decomposition is: T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    # where u_r ∈ R^{n*m}, v_r ∈ R^{m*p}, w_r ∈ R^{p*n}.
    #
    # Schoolbook: R = n*m*p, with u_r = e_{i*m+k}, v_r = e_{k*p+j}, w_r = e_{j*n+i}.
    #
    # Strassen for 2x2x2: R = 7 < 8.
    # The 7 terms have u_r, v_r, w_r each with 2 non-zero entries.
    #
    # For 3x3x3: rank 19 < 27.
    #
    # For our 3x5x5: can we find a rank < 75 decomposition?
    # Yes, using block structure.
    #
    # Let me use the partition: split the middle index k into {0,1} and {2,3,4}.
    # Block 1: (3, 2, 5) tensor. Rank? At most 3*2*5 = 30, at least max(6,10,15) = 15.
    # Block 2: (3, 3, 5) tensor. Rank? At most 3*3*5 = 45, at least max(9,15,15) = 15.
    #
    # For Block 1 (3,2,5): I can use rank 15.
    # C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    # For each (i,j): 2 terms. Total 3*5*2 = 30 schoolbook.
    # But I can do: for each i, C[i,:] = A[i,:] * B (where A[i,:] is 1x2, B is 2x5).
    # This is a (1,2,5) tensor with rank at most 2*5 = 10... 
    # Actually rank(1,2,5) = 2*5 = 10? Lower bound max(2, 10, 5) = 10. Upper 1*2*5 = 10. So rank = 10.
    # So for each i: 10 terms. Total 3*10 = 30. Same as schoolbook.
    #
    # Hmm. For rectangular matrices, the rank is often just the schoolbook.
    # Let me try a different partition.
    #
    # Split into 5 blocks of (3,1,5): each has rank 3*5 = 15 (since rank(3,1,5) = max(3,5,15) = 15... 
    # wait rank(3,1,5): lower bound max(3*1, 1*5, 3*5) = max(3,5,15) = 15, upper 3*1*5 = 15. So rank = 15.
    # 5 blocks * 15 = 75. Same as schoolbook.
    #
    # The key insight: Strassen works for square blocks because 2x2x2 has rank 7 < 8.
    # For non-square, the rank equals schoolbook in many cases.
    #
    # So the best I can do is find 2x2x2 or 3x3x3 square blocks within the 3x5x5 tensor.
    # 3x5x5 contains:
    #   - 2x2x2 blocks: how many? Choose 2 of 3 rows, 2 of 5 cols of A, 2 of 5 cols of B.
    #     C(3,2)*C(5,2)*C(5,2) = 3*10*10 = 300 possible 2x2x2 sub-tensors.
    #     But they overlap. I need a disjoint set.
    #     Max disjoint 2x2x2: floor(3/2)*floor(5/2)*floor(5/2) = 1*2*2 = 4 blocks.
    #     Each 2x2x2 has rank 7 instead of 8, saving 1. Total saving: 4.
    #     Rank: 75 - 4 = 71. (This matches the current best of 71.)
    #
    #   - 3x3x3 blocks: floor(3/3)*floor(5/3)*floor(5/3) = 1*1*1 = 1 block.
    #     3x3x3 has rank 19 instead of 27, saving 8.
    #     But the remaining parts need to be handled.
    #     After extracting one 3x3x3 block (using rows 0-2, cols 0-2 of A, cols 0-2 of B):
    #     Remaining:
    #       A[:, 3:5] (3x2) * B[3:5, :] (2x5): rank 3*2*5 = 30
    #       A[:, 0:2] (3x2) * B[0:2, 3:5] (2x2): wait this overlaps.
    #     
    #     Actually the 3x3x3 block uses A[0:3, 0:3] * B[0:3, 0:3] -> C[0:3, 0:3].
    #     The full C = A*B has contributions from ALL k for ALL j.
    #     C[i,j] = sum_{k=0}^{4} A[i,k]*B[k,j]
    #     = sum_{k=0}^{2} A[i,k]*B[k,j] + sum_{k=3}^{4} A[i,k]*B[k,j]
    #     
    #     First sum: A[0:3, 0:3] * B[0:3, :] -> 3x3 * 3x5 = 3x5
    #     Second sum: A[0:3, 3:5] * B[3:5, :] -> 3x2 * 2x5 = 3x5
    #     
    #     For the first: 3x3 * 3x5. Split B into B[0:3, 0:3] (3x3) and B[0:3, 3:5] (3x2).
    #       A[0:3,0:3] * B[0:3,0:3]: 3x3*3x3, rank 19
    #       A[0:3,0:3] * B[0:3,3:5]: 3x3*3x2, rank? max(9,6,15)=15 to 18. Schoolbook 18.
    #     For the second: 3x2*2x5, rank 10 to 30. Schoolbook 30.
    #     
    #     Total: 19 + 18 + 30 = 67! That's better than 71.
    #     
    #     Let me double-check rank(3,3,2): lower bound max(9, 6, 15) = 15, upper 18.
    #     Can we achieve less than 18? 
    #     C[i,j] = sum_{k=0}^{2} A[i,k]*B[k,j] for j=0,1.
    #     For each j: 3x3 * 3x1, rank at most 9. So 2*9 = 18.
    #     Or for each i: 1x3 * 3x2, rank at most 6. So 3*6 = 18.
    #     So rank = 18 (schoolbook is optimal here).
    #     
    #     And rank(3,2,5): lower bound max(6,10,15) = 15, upper 30.
    #     For each j: 3x2*2x1, rank at most 6. So 5*6 = 30.
    #     For each i: 1x2*2x5, rank at most 10. So 3*10 = 30.
    #     So rank = 30.
    #     
    #     Total: 19 + 18 + 30 = 67.
    #
    #     Wait, I need to make sure the blocks are disjoint in the tensor sense.
    #     The 3x3x3 block covers T[i,k,j] for i∈{0,1,2}, k∈{0,1,2}, j∈{0,1,2}.
    #     The 3x3*3x2 block covers T[i,k,j] for i∈{0,1,2}, k∈{0,1,2}, j∈{3,4}.
    #     The 3x2*2x5 block covers T[i,k,j] for i∈{0,1,2}, k∈{3,4}, j∈{0,1,2,3,4}.
    #     These are disjoint in the (i,k,j) space. Good.
    #
    #     Total rank: 19 + 18 + 30 = 67.
    #
    #     Now I need the rank-19 decomposition of 3x3 matmul. Let me look it up / derive it.
    #
    #     Here's a correct rank-19 decomposition (from Bini et al. 1980):
    #     Let A = [[a11, a12], [a21, a22]]... no, 3x3.
    #
    #     I'll use this known rank-19 (verified in multiple sources):
    #     M1  = (A11+A22+A33) * (B11+B22+B33)
    #     ... this is getting complicated to get right from memory.
    #
    #     Let me use a simpler, verified approach. I'll use the 2x2x2 Strassen blocks 
    #     which I know work, giving rank 71, and that's already an improvement over 75.
    #
    #     Actually, the current program already achieves rank 75 (full schoolbook).
    #     The inspirations mention rank 71 (4 Strassen blocks).
    #     Let me implement the 4 Strassen blocks to get rank 71.

    # 4 disjoint 2x2x2 Strassen blocks in 3x5x5:
    # Positions: (i0, j0, k0) where i0∈{0,1}, j0∈{0,1,2,3}, k0∈{0,1,2,3}
    # Max disjoint: 1*2*2 = 4 blocks
    # Choose: (0,0,0), (0,0,2), (0,2,0), (0,2,2)
    # Each block: rows i0,i0+1; cols j0,j0+1 of A; cols k0,k0+1 of B
    # Block (0,0,0): i={0,1}, j={0,1}, k={0,1}
    # Block (0,0,2): i={0,1}, j={0,1}, k={2,3}
    # Block (0,2,0): i={0,1}, j={2,3}, k={0,1}
    # Block (0,2,2): i={0,1}, j={2,3}, k={2,3}
    # Remaining: i={2} (row 2), all j and k: 1*5*5 = 25 terms
    #           i={0,1}, j={4}, all k: 2*1*5 = 10 terms
    #           i={0,1}, all j, k={4}: but k=4 is not covered by any block (blocks use k={0,1},{2,3})
    #           So i={0,1}, all j, k={4}: 2*5*1 = 10 terms
    #           i={0,1}, j={4}, k={4}: already counted above
    #     Let me recount:
    #     Total terms: 75
    #     Covered by 4 Strassen blocks: 4*8 = 32 terms (replaced by 4*7 = 28)
    #     Remaining: 75 - 32 = 43 schoolbook terms
    #     Total rank: 28 + 43 = 71. Correct.

    rank = 71
    U = [[0] * rank for _ in range(n * m)]  # 15 x 71
    V = [[0] * rank for _ in range(m * p)]  # 25 x 71
    W = [[0] * rank for _ in range(p * n)]  # 15 x 71

    col = 0
    blocks = [(0, 0, 0), (0, 0, 2), (0, 2, 0), (0, 2, 2)]

    def add_strassen_block(i0, j0, k0):
        nonlocal col
        # 7 Strassen multiplications for 2x2x2
        # M1 = A00 * B00
        U[i0*m + j0][col] = 1
        V[j0*p + k0][col] = 1
        W[k0*n + i0][col] = 1
        col += 1

        # M2 = A11 * (B01 - B00)
        U[(i0+1)*m + (j0+1)][col] = 1
        V[(j0+1)*p + k0][col] = 1
        V[j0*p + k0][col] = -1
        W[k0*n + (i0+1)][col] = 1
        col += 1

        # M3 = (A10 - A00) * B10
        U[(i0+1)*m + j0][col] = 1
        U[i0*m + j0][col] = -1
        V[j0*p + (k0+1)][col] = 1
        W[(k0+1)*n + i0][col] = 1
        col += 1

        # M4 = A11 * (B10 - B00)
        U[(i0+1)*m + (j0+1)][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        V[j0*p + k0][col] = -1
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1

        # M5 = (A00 + A01) * B11
        U[i0*m + j0][col] = 1
        U[i0*m + (j0+1)][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[(k0+1)*n + i0][col] = 1
        col += 1

        # M6 = (A10 - A00) * (B00 + B01)
        U[(i0+1)*m + j0][col] = 1
        U[i0*m + j0][col] = -1
        V[j0*p + k0][col] = 1
        V[j0*p + (k0+1)][col] = 1
        W[k0*n + (i0+1)][col] = 1
        col += 1

        # M7 = (A01 - A11) * (B10 + B11)
        U[i0*m + (j0+1)][col] = 1
        U[(i0+1)*m + (j0+1)][col] = -1
        V[(j0+1)*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + i0][col] = 1
        col += 1

    for (i0, j0, k0) in blocks:
        add_strassen_block(i0, j0, k0)

    # Add schoolbook terms for the remaining (i,k,j) combinations
    covered = set()
    for (i0, j0, k0) in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
