import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    _start_time = time.time()
    _deadline = _start_time + 130  # safety margin before 150s limit
    n, m, p = DIMENSIONS  # 4, 5, 5

    # Strategy: Use a 2+2+1 partition for all three dimensions.
    # A: rows split as 2+2, cols split as 2+2+1
    # B: rows split as 2+2+1, cols split as 2+2+1
    # C: rows split as 2+2+1, cols split as 2+2

    # The full tensor M_{4,5,5} can be decomposed by considering all block interactions.
    # We'll use the following approach:
    # - 2x2x2 blocks: use Strassen (rank 7 each)
    # - Other blocks: use direct multiplication (rank = product of sizes)
    # Then try to optimize.

    # Actually, let me try a cleaner approach: use the known result that
    # R(4,5,5) <= R(4,4,4) + R(4,4,1) + R(4,1,4) + R(4,1,1) cross terms
    # But a better split: 5 = 2+3

    # Let's try: A is 4x5 = 4x(2+3), B is 5x5 = (2+3)x5, C is 5x4 = (2+3)x4
    # Blocks:
    # C_{1,1} = A_{1,:} * B_{:,1} where A_{1,:} is 4x5, B_{:,1} is 5x2
    # C_{1,2} = A_{1,:} * B_{:,2} where B_{:,2} is 5x3
    # C_{2,1} = A_{2,:} * B_{:,1} where A_{2,:} is... wait, let me be more careful.

    # Let me define:
    # A = [A1; A2] where A1 is 4x2, A2 is 4x3 (split columns)
    # No, that's not right for matrix multiplication.

    # C = A * B where A is 4x5, B is 5x5, C is 4x5.
    # Split: A = [A_L | A_R] where A_L is 4x2, A_R is 4x3
    # B = [B_T; B_B] where B_T is 2x5, B_B is 3x5
    # C = A_L * B_T + A_R * B_B (4x5)

    # Split C columns: C = [C_L | C_R] where C_L is 4x2, C_R is 4x3
    # C_L = A_L * B_T[:,0:2] + A_R * B_B[:,0:2] (4x2)
    # C_R = A_L * B_T[:,2:5] + A_R * B_B[:,2:5] (4x3)

    # So we have 4 sub-problems:
    # 1. (A_L, B_T[:,0:2]) -> contributes to C_L: 4x2 times 2x2 = 4x2
    # 2. (A_R, B_B[:,0:2]) -> contributes to C_L: 4x3 times 3x2 = 4x2
    # 3. (A_L, B_T[:,2:5]) -> contributes to C_R: 4x2 times 2x3 = 4x3
    # 4. (A_R, B_B[:,2:5]) -> contributes to C_R: 4x3 times 3x3 = 4x3

    # Ranks:
    # 1. M_{4,2,2}: rank <= 4*2 = 8 (direct) or use Strassen on 2x2 part: 4*7=28? No.
    #    Actually M_{4,2,2}: for each of 4 output rows, we compute a 2x1 vector from 4x2 * 2x2.
    #    Direct: 4*2*2 = 16 mults. Can we do better?
    #    M_{4,2,2} = M_{4,2,1} + M_{4,2,1} (shifted). R(M_{4,2,1}) <= 4*2=8? Actually R(M_{n,m,1}) = n*m.
    #    So R(M_{4,2,2}) <= 2*8 = 16? Or can we share?
    #    Actually C_{i,k} = sum_j A_{i,j} B_{j,k} for k=0,1.
    #    = (sum_j A_{i,j} B_{j,0}, sum_j A_{i,j} B_{j,1})
    #    For each i, this is a 2D output from 4x2 inputs. Rank of M_{4,2,2}...
    #    Let's just use direct: 4*2*2 = 16 terms.

    # 2. M_{4,3,2}: direct = 4*3*2 = 24 terms.
    #    Can we do better? Maybe use a 3x3 structure partially?
    #    Actually, let me think of this differently.

    # Let me try the 2+2+1 split for the middle dimension only:
    # A: 4x(2+2+1), B: (2+2+1)x(2+2+1), C: (2+2+1)x4... no, C is 4x5.

    # OK let me just try to get a rank below 85 by using a smarter combination.

    # Alternative: Use the fact that R(4,5,5) <= 7*R(2,2,2) for the 2x2 blocks + direct for rest
    # with 2+2+1 split on all dimensions where applicable.

    # Let me try: split indices as 2+2+1 for the 5-dimensions, 2+2 for the 4-dimensions.
    # A: rows 2+2, cols 2+2+1
    # B: rows 2+2+1, cols 2+2+1
    # C: rows 2+2+1, cols 2+2

    # Block (i,j,k) means A[i,:], B[:,j], C[k,:] contributions... this is getting complex.

    # Let me just implement the 2+2+1 split carefully.

    # For 4x5x5, use partition:
    # n=4: [2, 2]
    # m=5: [2, 2, 1]
    # p=5: [2, 2, 1]

    # C is 4x5, so C partition: rows [2,2], cols [2,2,1] -> wait C is 4 rows x 5 cols
    # A is 4x5: rows [2,2], cols [2,2,1]
    # B is 5x5: rows [2,2,1], cols [2,2,1]
    # C is 4x5: rows [2,2], cols [2,2,1]

    # The tensor M_{4,5,5} decomposes into blocks:
    # For each (row_block_A, col_block_B, row_block_C, col_block_C) where
    # row_block_A in [0,1], col_block_B in [0,1,2], row_block_C in [0,1], col_block_C in [0,1,2]
    # The sub-tensor is M_{|A_rb|, |B_cb|, |C_rb| x |C_cc|}
    # But C has separate row and col partitions, so actually:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # i in C row partition, j in shared (A col = B row) partition, k in C col partition.

    # So for each (a_row, m_shared, c_row, c_col) block:
    # A block: size a_row x m_shared
    # B block: size m_shared x c_col
    # C block: size c_row x c_col

    # Sub-tensor rank contributes to the total.

    # Partitions:
    # a_rows: [2, 2]
    # m: [2, 2, 1]  (shared between A cols and B rows)
    # c_rows: [2, 2]
    # c_cols: [2, 2, 1]

    # For each combination:
    # a_row_size, m_size, c_row_size, c_col_size -> M_{a_row, m, c_row x c_col}
    # which is a tensor of shape (a_row * c_row, m, c_row * c_col)... no.
    # Actually it's M_{a_row, m, c_col} for each c_row index independently? No.
    # C_{i,k} = sum_j A_{i,j} B_{j,k} where i is in c_row block, j in m block, k in c_col block.
    # So for fixed (a_row_block, m_block, c_row_block, c_col_block):
    # We compute a sub-matrix of C of size (c_row_block_size) x (c_col_block_size)
    # from A sub-matrix of size (a_row_block_size) x (m_block_size)
    # and B sub-matrix of size (m_block_size) x (c_col_block_size)
    # But wait, the a_row_block determines which rows of A, and c_row_block determines which rows of C.
    # If a_row_block == c_row_block (same index), then A rows correspond to C rows.
    # Since both are partitioned as [2,2], a_row_block i corresponds to c_row_block i.

    # So for each (r in 0..1, m_b in 0..2, c_c in 0..2):
    # A block: rows [2r, 2r+2), cols [m_start[m_b], m_start[m_b]+m_size[m_b])
    # B block: rows [m_start[m_b], m_start[m_b]+m_size[m_b]), cols [c_start[c_c], c_start[c_c]+c_size[c_c])
    # C block: rows [2r, 2r+2), cols [c_start[c_c], c_start[c_c]+c_size[c_c])

    # Sub-problem: multiply (2 x m_size) by (m_size x c_size) to get (2 x c_size)
    # This is M_{2, m_size, c_size}

    a_row_sizes = [2, 2]
    m_sizes = [2, 2, 1]
    c_col_sizes = [2, 2, 1]

    m_starts = [0, 2, 4]
    c_starts = [0, 2, 4]

    # Ranks for M_{2, a, b}:
    # M_{2,1,1}: rank 1 (one multiplication)
    # M_{2,1,2}: rank 2 (C_{0,0}=A0*B0, C_{0,1}=A0*B1, C_{1,0}=A1*B0, C_{1,1}=A1*B1) -> 4 mults? No, 2*1*2=4.
    #   Actually rank = 4 for M_{2,1,2}? C_{i,k} = A_i * B_k for i=0,1 and k=0,1. That's 4 independent products. Rank 4.
    #   Wait no: C_{0,0}=A0*B0, C_{0,1}=A0*B1, C_{1,0}=A1*B0, C_{1,1}=A1*B1. These are 4 independent bilinear terms. Rank 4.
    # M_{2,2,1}: rank 4 (similarly, C_{i,0}=A_{i,0}*B_{0,0}+A_{i,1}*B_{1,0} for i=0,1 -> 4 mults)
    # M_{2,2,2}: rank 7 (Strassen)
    # M_{2,1,3}: rank 6 (2*1*3=6 direct, likely optimal)
    # M_{2,2,3}: rank ?  (2*2*3=12 direct. Can we do better? Maybe 10 or 11?)
    # M_{2,3,1}: rank 6
    # M_{2,3,2}: rank ? (2*3*2=12 direct)
    # M_{2,3,3}: rank ? (2*3*3=18 direct. Winograd for 3x3 is rank 27, but this is 2x3x3.)

    # For now, use direct multiplication for all sub-blocks and see what rank we get.
    # Then optimize specific blocks.

    def get_block_rank(a_size, m_size, c_size):
        """Get rank for M_{a_size, m_size, c_size} using best known method."""
        if a_size == 0 or m_size == 0 or c_size == 0:
            return 0
        if a_size == 2 and m_size == 2 and c_size == 2:
            return 7  # Strassen
        # Direct multiplication
        return a_size * m_size * c_size

    total_rank = 0
    for r in range(2):
        for m_b in range(3):
            for c_c in range(3):
                a_sz = a_row_sizes[r]
                m_sz = m_sizes[m_b]
                c_sz = c_col_sizes[c_c]
                total_rank += get_block_rank(a_sz, m_sz, c_sz)

    # Let's compute:
    # r=0 or 1 (2 of these):
    #   m_b=0 (m_sz=2), m_b=1 (m_sz=2), m_b=2 (m_sz=1)
    #   c_c=0 (c_sz=2), c_c=1 (c_sz=2), c_c=2 (c_sz=1)
    #
    # For a_sz=2:
    #   (m=2,c=2): 7
    #   (m=2,c=2): 7
    #   (m=2,c=1): 4
    #   (m=2,c=2): 7
    #   (m=2,c=2): 7
    #   (m=2,c=1): 4
    #   (m=1,c=2): 4
    #   (m=1,c=2): 4
    #   (m=1,c=1): 1
    # Per r: 7+7+4+7+7+4+4+4+1 = 45
    # Total for 2 rows: 90
    # That's worse than 85!

    # The issue is that the 2+2+1 split creates many small blocks that don't benefit from Strassen.
    # Let me try a different approach.

    # Go back to the 4x4x4 + strips approach but try to optimize.
    # Current: 49 (Strassen 4x4x4) + 16 + 16 + 4 = 85
    #
    # What if we use a different 4x4x4 decomposition? The optimal rank for 4x4x4 is 49 (Strassen-based).
    # But maybe a non-Strassen 4x4x4 with rank < 49 exists? No, 49 is proven optimal.
    #
    # What about a different partition of the 5x5x4 problem?
    # Instead of 4+1, try 3+2:
    # A: 4x(3+2), B: (3+2)x(3+2), C: (3+2)x4... C is 4x5 not 5x4.
    #
    # Let me try: A: 4x5, B: 5x5, C: 4x5
    # Split the 5 as 3+2 in the shared dimension:
    # A = [A1 | A2] where A1 is 4x3, A2 is 4x2
    # B = [B1; B2] where B1 is 3x5, B2 is 2x5
    # C = A1*B1 + A2*B2 (4x5)
    #
    # Now split C: C = [C_left | C_right] where C_left is 4x3, C_right is 4x2
    # C_left = A1*B1[:,0:3] + A2*B2[:,0:3] (4x3)
    # C_right = A1*B1[:,3:5] + A2*B2[:,3:5] (4x2)
    #
    # Sub-problems:
    # 1. A1 (4x3) * B1_left (3x3) -> C_left_part (4x3): M_{4,3,3}
    # 2. A2 (4x2) * B2_left (2x3) -> C_left_part (4x3): M_{4,2,3}
    # 3. A1 (4x3) * B1_right (3x2) -> C_right_part (4x2): M_{4,3,2}
    # 4. A2 (4x2) * B2_right (2x2) -> C_right_part (4x2): M_{4,2,2}
    #
    # Ranks:
    # M_{4,3,3}: direct = 36. Can we do better?
    #   R(3,3,3) = 23 (Winograd). But this is M_{4,3,3} not M_{3,3,3}.
    #   M_{4,3,3}: C_{i,k} = sum_j A_{i,j} B_{j,k} for i=0..3, j=0..2, k=0..2.
    #   For each i, it's M_{1,3,3} which has rank 9 (direct) or less?
    #   R(M_{1,3,3}) = R(M_{3,3}) for a single row = 9? Actually C_k = sum_j A_j * B_{j,k}.
    #   This is a 3x3 matrix product where A is a row vector. Rank = 3*3 = 9? Or can we do 8?
    #   Actually for a fixed row of A, C = a * B where a is 1x3, B is 3x3.
    #   C_{k} = sum_j a_j B_{j,k}. This requires 9 multiplications in general (each a_j * B_{j,k}).
    #   But we can precompute... no, a_j varies. So rank 9 per output row, 4 rows -> 36.
    #   However, we can share computations across rows! If two rows of A share structure...
    #   But in a bilinear decomposition, we can't assume structure. So M_{4,3,3} rank = 4*9 = 36?
    #   Actually no. The tensor M_{4,3,3} has entries T_{i,j,k} = delta_{i,j}*delta... no.
    #   T_{(i,k),j} = 1 if... wait, the matrix multiplication tensor M_{n,m,p} has
    #   T_{(i,k),j} = 1 for the term A_{i,j}*B_{j,k}.
    #   So M_{4,3,3} has 4*3*3 = 36 non-zero entries, one for each (i,j,k) triple.
    #   The rank is the minimum number of rank-1 tensors to express it.
    #   For M_{4,3,3}, I believe rank = 36 (direct) unless there's a clever decomposition.
    #   Actually, R(M_{1,m,p}) = m*p (each output is a separate product sum).
    #   So R(M_{4,3,3}) >= 4 * R(M_{1,3,3}) = 4*9 = 36? Not necessarily, because
    #   rank is subadditive in a different way. Actually R(M_{a,b,c}) <= a*R(M_{1,b,c}) is trivial.
    #   And R(M_{a,b,c}) >= max(a*R(M_{1,b,c}), ...) by slicing.
    #   Slicing on the first index: M_{4,3,3} contains 4 copies of M_{1,3,3}, so rank >= 4*R(M_{1,3,3}).
    #   R(M_{1,3,3}) = 9 (it's just 9 independent products: C_k = sum_j A_j B_{j,k}, 9 terms).
    #   Actually wait: C_k = A_0*B_{0,k} + A_1*B_{1,k} + A_2*B_{2,k}. That's 3 products per k, 3 k's = 9 products.
    #   Can we do it in 8? The rank of M_{1,3,3} is the rank of a 9x9 matrix (the "flattened" tensor).
    #   M_{1,3,3}: T_{(k,j)} = 1. This is just the 9-dimensional identity-like structure.
    #   Actually T_{k,j} where the output index is k and input indices are (A_j for j=0,1,2 and B_{j,k} for j=0,1,2).
    #   The tensor is T_{k, (j, l)} = delta_{j,l} * 1 (for the j-th A and l-th B contributing to k-th C).
    #   Hmm, let me think of it as: output C_k = sum_j A_j * B_{j,k}.
    #   Bilinear form: for fixed k, it's a bilinear map from (A_0,A_1,A_2) x (B_{0,k},B_{1,k},B_{2,k}) to C_k.
    #   The matrix of this bilinear form is I_3 (identity), so rank 3 per k.
    #   But the 3 k's share the same A variables. So the full tensor has 9 output variables (C_0,C_1,C_2)
    #   and 6 input variables (A_0,A_1,A_2,B_{0,0},B_{0,1},B_{0,2},B_{1,0},B_{1,1},B_{1,2},B_{2,0},B_{2,1},B_{2,2}).
    #   Wait that's 3 A's and 9 B's = 12 inputs, 3 outputs.
    #   The tensor T has shape (3, 3, 9) -> T_{k, a_idx, b_idx}.
    #   T_{k, j, l} = 1 if a_idx=j and b_idx corresponds to (j,k), else 0.
    #   So T_{k, j, (j', k')} = delta_{j,j'} * delta_{k,k'}.
    #   This is a 3x3x9 tensor. Its rank is... 9? Because each of the 9 "B slots" is paired with exactly one A slot.
    #   Yes, R(M_{1,3,3}) = 9.
    #   So R(M_{4,3,3}) >= 4*9 = 36? No, slicing gives rank >= R(M_{1,3,3}) = 9, not 4*9.
    #   Actually, if we fix the first A-index to a specific value, we get M_{1,3,3} as a "slice".
    #   So rank(M_{4,3,3}) >= rank(M_{1,3,3}) = 9. That's a weak lower bound.
    #   The trivial upper bound is 36 (direct).
    #   I don't know the exact rank of M_{4,3,3}. Let's assume 36 for now.

    # This approach seems to give: 36 + 24 + 24 + 8 = 92. Worse.

    # Let me go back to the original approach and just make sure it's correct and fast.
    # The current rank 85 comes from:
    # - 7 Strassen terms for the top-left 4x4x4 block (rank 49)
    # - Cross terms for the 5th column/row

    # Actually, I realize I should just verify the current code works and try a small optimization:
    # Can we reduce the cross-term rank?

    # The current cross terms:
    # 1. 16 terms for A[:,0:4] * B[4,:] -> C[:,4]  (M_{4,4,1} type, but really it's the 4x4x1 block)
    #    Wait: C_{i,4} = sum_{j=0}^{3} A_{i,j} B_{j,4} for i=0..3. That's 16 mults.
    #    Plus C_{i,4} += A_{i,4} B_{4,4}. That's 1 more.
    #    Total for column 4 of C: 16+1 = 17? But the code has 16+16+4 = 36 cross terms.

    # Let me re-examine the code structure:
    # - 49 terms for 4x4x4 Strassen
    # - 16 terms: for i in 0..3, j in 0..3: U[i*5+j], V[j*5+4], W[4*4+i]
    #   This computes: contribution to C[i,4] from A[i,j]*B[j,4] for i,j in 0..3. That's 16 terms.
    #   But wait, W[4*4+i] = W[16+i] which is C row 4, col i? No, W index is k*4+i where k is C col, i is C row.
    #   W[16+i] means k=4, i=i. So C[i,4]. Yes, 16 terms for the 4x4 part of C[:,4].
    # - 16 terms: for i in 0..3, k in 0..3: U[i*5+4], V[4*5+k], W[k*4+i]
    #   U[i*5+4] = A[i,4], V[4*5+k] = B[4,k], W[k*4+i] = C[i,k].
    #   This computes C[i,k] += A[i,4]*B[4,k] for i,k in 0..3. That's 16 terms for C[0:4,0:4] from row 4 of A and col 4 of B.
    # - 4 terms: for i in 0..3: U[i*5+4], V[4*5+4], W[4*4+i]
    #   A[i,4]*B[4,4] -> C[i,4]. 4 terms.

    # Total: 49 + 16 + 16 + 4 = 85. Correct.

    # Can we reduce any of these?
    # The 16+16+4 = 36 cross terms handle:
    # - C[i,4] for i=0..3 from j=0..3: 16 terms (M_{4,4,1} essentially, but shared across i)
    #   Actually C[i,4] = sum_{j=0}^3 A[i,j]*B[j,4]. For each i, this is 4 products. 4 i's = 16.
    #   Can we do better? R(M_{4,4,1}) where the "1" is a single column of C.
    #   C[i,4] = sum_j A[i,j] B[j,4]. This is like multiplying a 4x4 matrix by a 4x1 vector.
    #   Rank = 16 (each output is an inner product, 16 independent terms).
    #   Actually, can we share? C[0,4] = A[0,0]B[0,4]+A[0,1]B[1,4]+A[0,2]B[2,4]+A[0,3]B[3,4]
    #   C[1,4] = A[1,0]B[0,4]+A[1,1]B[1,4]+A[1,2]B[2,4]+A[1,3]B[3,4]
    #   These share B[j,4] factors. Can we use a Strassen-like trick?
    #   The tensor here is M_{4,4,1} (4 rows of A, 4 cols shared, 1 col of C).
    #   This is equivalent to computing y = A * x where A is 4x4, x is 4x1.
    #   The rank of this bilinear map is 16 (it's just 16 independent products in the standard basis).
    #   I don't think we can reduce below 16 for this specific structure.

    # Similarly for the other 16 terms.

    # So 85 seems hard to beat with this structure. Let me just ensure the code is correct and fast.

    # Let me rewrite cleanly with the known rank-85 decomposition.
    rank = 85
    U = [[0] * rank for _ in range(n * m)]  # 20 x 85
    V = [[0] * rank for _ in range(m * p)]  # 25 x 85
    W = [[0] * rank for _ in range(p * n)]  # 20 x 85

    # Strassen 2x2x2 factors
    # u[i][k], v[j][k], w[l][k] for 2x2x2
    su = [[0]*7 for _ in range(4)]
    sv = [[0]*7 for _ in range(4)]
    sw = [[0]*7 for _ in range(4)]

    # Term 0: (a00+a11)(b00+b11) -> c00 += , c11 +=
    su[0][0]=1; su[3][0]=1
    sv[0][0]=1; sv[3][0]=1
    sw[0][0]=1; sw[3][0]=1

    # Term 1: (a10+a11)*b00 -> c10 += , c11 -=
    su[2][1]=1; su[3][1]=1
    sv[0][1]=1
    sw[2][1]=1; sw[3][1]=-1

    # Term 2: a00*(b01-b11) -> c01 += , c11 +=
    su[0][2]=1
    sv[1][2]=1; sv[3][2]=-1
    sw[1][2]=1; sw[3][2]=1

    # Term 3: a11*(b10-b00) -> c00 += , c10 +=
    su[3][3]=1
    sv[2][3]=1; sv[0][3]=-1
    sw[0][3]=1; sw[2][3]=1

    # Term 4: (a00+a01)*b11 -> c00 -= , c01 +=
    su[0][4]=1; su[1][4]=1
    sv[3][4]=1
    sw[0][4]=-1; sw[1][4]=1

    # Term 5: (a10-a00)*(b00+b01) -> c11 +=
    su[0][5]=-1; su[2][5]=1
    sv[0][5]=1; sv[1][5]=1
    sw[3][5]=1

    # Term 6: (a01-a11)*(b10+b11) -> c00 +=
    su[1][6]=1; su[3][6]=-1
    sv[2][6]=1; sv[3][6]=1
    sw[0][6]=1

    # Build 4x4x4 from 2x2x2 Strassen
    # The 4x4x4 Strassen has rank 49 = 7 * 7.
    # It's built from 7 "Strassen terms" each involving a 2x2x2 sub-decomposition.

    # The standard 4x4x4 decomposition uses the following 7 terms:
    # Each term t (0..6) involves:
    # - A linear combination of 2x2 blocks of A
    # - A linear combination of 2x2 blocks of B
    # - A linear combination of 2x2 blocks of C
    # Applied with the 2x2x2 Strassen pattern.

    # The 4x4 blocks: A has 4 blocks (2x2 each), B has 4 blocks, C has 4 blocks.
    # Block (i,j) of a 4x4 matrix is rows 2i:2i+2, cols 2j:2j+2.

    # The 7 terms for 4x4x4:
    terms = [
        # (A_blocks, B_blocks, C_blocks)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    for t_idx, (a_coeff, b_coeff, c_coeff) in enumerate(terms):
        if time.time() > _deadline:
            return {"U": U, "V": V, "W": W}
        start_col = t_idx * 7
        for inner_k in range(7):
            col_idx = start_col + inner_k
            # A contribution
            for (br, bc), coeff in a_coeff.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr * 2 + lc
                        u_val = su[li][inner_k]
                        if u_val != 0:
                            gi = 2 * br + lr
                            gj = 2 * bc + lc
                            U[gi * 5 + gj][col_idx] += coeff * u_val
            # B contribution
            for (br, bc), coeff in b_coeff.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr * 2 + lc
                        v_val = sv[li][inner_k]
                        if v_val != 0:
                            gj = 2 * br + lr
                            gk = 2 * bc + lc
                            V[gj * 5 + gk][col_idx] += coeff * v_val
            # C contribution
            for (br, bc), coeff in c_coeff.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr * 2 + lc
                        w_val = sw[li][inner_k]
                        if w_val != 0:
                            gi = 2 * br + lr
                            gk = 2 * bc + lc
                            W[gk * 4 + gi][col_idx] += coeff * w_val

    col = 49
    # Cross term 1: A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4]
    for i in range(4):
        if time.time() > _deadline:
            return {"U": U, "V": V, "W": W}
        for j in range(4):
            U[i*5 + j][col] = 1
            V[j*5 + 4][col] = 1
            W[4*4 + i][col] = 1  # C[i,4]
            col += 1

    # Cross term 2: A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4]
    for i in range(4):
        if time.time() > _deadline:
            return {"U": U, "V": V, "W": W}
        for k in range(4):
            U[i*5 + 4][col] = 1
            V[4*5 + k][col] = 1
            W[k*4 + i][col] = 1  # C[i,k]
            col += 1

    # Cross term 3: A[0:4, 4] * B[4, 4] -> C[0:4, 4]
    for i in range(4):
        if time.time() > _deadline:
            return {"U": U, "V": V, "W": W}
        U[i*5 + 4][col] = 1
        V[4*5 + 4][col] = 1
        W[4*4 + i][col] = 1  # C[i,4]
        col += 1

    assert col == 85, f"Expected 85, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
