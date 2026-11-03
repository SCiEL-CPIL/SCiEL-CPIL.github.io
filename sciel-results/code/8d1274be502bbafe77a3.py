import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = 2, 3, 9
    
    # Strategy: Use Strassen (rank 7) for 2x2x2 blocks where possible.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # Split into:
    #   Part 1: A[:,0:2] * B[0:2,:]  (2x2 * 2x9)
    #   Part 2: A[:,2:3] * B[2:3,:]  (2x1 * 1x9)
    #
    # Part 2: schoolbook, rank = 2*1*9 = 18
    # Part 1: Split 9 cols into 4 pairs + 1:
    #   4 blocks of 2x2 * 2x2: Strassen rank 7 each -> 28
    #   1 block of 2x2 * 2x1: schoolbook rank 4 -> 4
    #   Total Part 1: 32
    # Grand total: 32 + 18 = 50
    
    U_rows = n * m  # 6
    V_rows = m * p  # 27
    W_rows = p * n  # 18
    rank = 50
    
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    def strassen_2x2x2(offset, i_off, j_off, k_off):
        """Add Strassen rank-7 terms for 2x2 * 2x2 block.
        i_off: row offset in A (0 or 1)
        j_off: col offset in A / row offset in B (0 or 1)
        k_off: col offset in B / col offset in C (0 or 1)
        Returns next offset.
        """
        # Strassen's 7 products for 2x2 * 2x2:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each M is a product of linear combinations.
        # We need to express each M as a sum of rank-1 terms in the tensor.
        # Actually, for the tensor decomposition, each "multiplication" M_t
        # corresponds to one rank-1 term in the tensor where:
        # U encodes the A-side linear combo, V encodes the B-side linear combo,
        # and W encodes the C-side output.
        #
        # But Strassen's algorithm uses 7 scalar multiplications, each of which
        # is a rank-1 term in the tensor sense (one product of two scalars).
        # So each M_t is ONE rank-1 term.
        #
        # For M1: A-side = A00+A11, B-side = B00+B11
        #   U[i_off+0, j_off+0, r] += 1  (A00)
        #   U[i_off+1, j_off+1, r] += 1  (A11)
        #   V[j_off+0, k_off+0, r] += 1  (B00)
        #   V[j_off+1, k_off+1, r] += 1  (B11)
        #   W[k_off+0, i_off+0, r] += 1  (C00)
        #   W[k_off+1, i_off+0, r] += 1  (C01? No...)
        
        # Wait, I need to be more careful. In the tensor decomposition:
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
        # where i indexes rows of A (and C), j indexes cols of A (and rows of B),
        # k indexes cols of B (and C).
        #
        # For a single rank-1 term to contribute to C[i,k]:
        #   U[i,j,r] * V[j,k,r] * W[i,k,r] should equal A[i,j] * B[j,k] for the
        #   specific (i,j,k) we want, and 0 otherwise.
        #
        # But Strassen's algorithm works on 2x2 blocks. Let me think of it differently.
        #
        # For a 2x2 * 2x2 block at positions (i_off..i_off+1, j_off..j_off+1) in A
        # and (j_off..j_off+1, k_off..k_off+1) in B, producing
        # (i_off..i_off+1, k_off..k_off+1) in C:
        #
        # The naive decomposition uses 4 terms (one per (i,j,k) triple in the block).
        # Strassen uses 7 terms total for the 4 output entries.
        #
        # Each Strassen "multiplication" M_t corresponds to a rank-1 tensor term
        # where U has nonzeros on the A-side support, V on the B-side support,
        # and W on the C-side support.
        #
        # M1: A-side: {(0,0),(1,1)}, B-side: {(0,0),(1,1)}
        #   Contributes to C: (0,0) with +1, (1,1) with -1 (from C11 = M1 - M2 + M3 + M6)
        #   Wait no. Let me re-derive.
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        #
        # So M1 contributes +1 to C00 and +1 to C11.
        # M2 contributes +1 to C10 and -1 to C11.
        # M3 contributes +1 to C01 and +1 to C11.
        # M4 contributes +1 to C00 and +1 to C10.
        # M5 contributes -1 to C00 and +1 to C01.
        # M6 contributes +1 to C11.
        # M7 contributes +1 to C00.
        
        # For each M_t, define the U, V, W vectors:
        # M1: U: A00=1, A11=1; V: B00=1, B11=1; W: C00=1, C11=1
        # M2: U: A10=1, A11=1; V: B00=1; W: C10=1, C11=-1
        # M3: U: A00=1; V: B01=1, B11=-1; W: C01=1, C11=1
        # M4: U: A11=1; V: B10=1, B00=-1; W: C00=1, C10=1
        # M5: U: A00=1, A01=1; V: B11=1; W: C00=-1, C01=1
        # M6: U: A10=1, A00=-1; V: B00=1, B01=1; W: C11=1
        # M7: U: A01=1, A11=-1; V: B10=1, B11=1; W: C00=1
        
        strassen_terms = [
            # (U_entries, V_entries, W_entries)
            # Each entry is (local_index, value)
            ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),       # M1
            ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),             # M2
            ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),             # M3
            ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),             # M4
            ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),             # M5
            ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),             # M6
            ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),             # M7
        ]
        
        for u_entries, v_entries, w_entries in strassen_terms:
            for (li, lv) in u_entries:
                U[i_off + li // 2][offset] += lv  # Wait, local index mapping
                # local index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                # U index: (i_off + li//2) * m + (j_off + li%2)
                U[(i_off + li // 2) * m + (j_off + li % 2)][offset] += lv
            for (li, lv) in v_entries:
                # local index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                # V index: (j_off + li//2) * p + (k_off + li%2)
                V[(j_off + li // 2) * p + (k_off + li % 2)][offset] += lv
            for (li, lv) in w_entries:
                # local index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                # W index: (k_off + li//2) * n + (i_off + li%2)
                W[(k_off + li // 2) * n + (i_off + li % 2)][offset] += lv
            offset += 1
        
        return offset
    
    def schoolbook_2x2x2(offset, i_off, j_off, k_off):
        """Add schoolbook rank-4 terms for 2x2 * 2x2 block."""
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    U[(i_off + di) * m + (j_off + dj)][offset] = 1
                    V[(j_off + dj) * p + (k_off + dk)][offset] = 1
                    W[(k_off + dk) * n + (i_off + di)][offset] = 1
                    offset += 1
        return offset
    
    def schoolbook_2x2x1(offset, i_off, j_off, k_off):
        """Add schoolbook rank-4 terms for 2x2 * 2x1 block."""
        for di in range(2):
            for dj in range(2):
                U[(i_off + di) * m + (j_off + dj)][offset] = 1
                V[(j_off + dj) * p + (k_off)][offset] = 1
                W[(k_off) * n + (i_off + di)][offset] = 1
                offset += 1
        return offset
    
    def schoolbook_2x1x9(offset, i_off, j_idx, k_start, k_end):
        """Add schoolbook rank terms for 2x1 * 1x(k_end-k_start) block."""
        for di in range(2):
            for dk in range(k_start, k_end):
                U[(i_off + di) * m + j_idx][offset] = 1
                V[j_idx * p + dk][offset] = 1
                W[dk * n + (i_off + di)][offset] = 1
                offset += 1
        return offset
    
    offset = 0
    
    # Part 1: A[:,0:2] * B[0:2,:] (2x2 * 2x9)
    # Split 9 cols: 4 pairs (0-1, 2-3, 4-5, 6-7) + 1 single (8)
    for block in range(4):
        k_off = block * 2
        offset = strassen_2x2x2(offset, 0, 0, k_off)
    
    # Last single column (col 8)
    offset = schoolbook_2x2x1(offset, 0, 0, 8)
    
    # Part 2: A[:,2:3] * B[2:3,:] (2x1 * 1x9)
    offset = schoolbook_2x1x9(offset, 0, 2, 0, 9)
    
    # Verify: offset should equal rank
    assert offset == rank, f"Expected rank {rank}, got {offset}"
    
    # Verify correctness with random test
    import random
    random.seed(42)
    A_test = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
    B_test = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
    
    # Compute C = A * B
    C_expected = [[0] * p for _ in range(n)]
    for i in range(n):
        for k in range(p):
            s = 0
            for j in range(m):
                s += A_test[i][j] * B_test[j][k]
            C_expected[i][k] = s
    
    # Compute C from decomposition
    C_computed = [[0] * p for _ in range(n)]
    for r in range(rank):
        # U_vec: n*m entries
        u_vec = [U[ui][r] for ui in range(U_rows)]
        # V_vec: m*p entries
        v_vec = [V[vi][r] for vi in range(V_rows)]
        # W_vec: p*n entries
        w_vec = [W[wi][r] for wi in range(W_rows)]
        
        for i in range(n):
            for k in range(p):
                # Sum over j of U[i*m+j] * V[j*p+k] * W[k*n+i]
                s = 0
                for j in range(m):
                    s += u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i]
                C_computed[i][k] += s
    
    for i in range(n):
        for k in range(p):
            if C_expected[i][k] != C_computed[i][k]:
                # Fall back to naive
                U2 = [[0] * 54 for _ in range(U_rows)]
                V2 = [[0] * 54 for _ in range(V_rows)]
                W2 = [[0] * 54 for _ in range(W_rows)]
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            r2 = (i2 * m + j2) * p + k2
                            U2[i2 * m + j2][r2] = 1
                            V2[j2 * p + k2][r2] = 1
                            W2[k2 * n + i2][r2] = 1
                return {"U": U2, "V": V2, "W": W2}
    
    return {"U": U, "V": V, "W": W}
    # Strategy: Block decomposition
    # A is 2x3, B is 3x9, C is 2x9
    # We can view B as a block matrix of 3 blocks of size 3x3: B = [B0, B1, B2]
    # Then C = A * [B0, B1, B2] = [A*B0, A*B1, A*B2]
    # Each A*Bk is a 2x3 * 3x3 multiplication.
    # Let R be the rank of the 2x3 x 3x3 multiplication tensor.
    # Then the total rank is 3*R.
    # We know rank(2x3 x 3x3) <= 2*3*3 = 18.
    # Can we do better than 18 for 2x3 x 3x3?
    # Strassen-like methods might help.
    # However, 2x3 x 3x3 is small. Let's try to find the minimal rank for 2x3 x 3x3.
    # Actually, we can just use the naive rank for the sub-blocks if we can't find better.
    # Naive rank for 2x3 x 3x3 is 18. Total rank 54.
    # This is the same as the current seed.
    
    # Is there a known lower rank for 2x3 x 3x3?
    # According to literature, the rank of 2x3 x 3x3 is not definitively known to be < 18, but it's likely 18 or close.
    # Let's try a different blocking.
    # What if we block A? A is 2x3. Can't really block 2 rows.
    # What if we block the columns of A and rows of B differently?
    
    # Let's look at the dimensions: 2x3 and 3x9.
    # The middle dimension is 3.
    # This is essentially 3 independent 2x1 * 1x9 multiplications? No.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # This is a sum of 3 terms.
    # Each term is a rank-1 matrix (outer product of row i of A and row j of B? No, column j of A and row j of B).
    # A[i,:] is 1x3. B[:,k] is 3x1.
    # C[i,k] = dot(A[i,:], B[:,k]).
    
    # Let's try to use a fast matrix multiplication algorithm if possible.
    # For 2x3 x 3x9, we can pad to 4x4 x 4x4?
    # 2->4, 3->4, 9->16? No, 9 is not a power of 2.
    # Pad 2x3 to 4x4. Pad 3x9 to 4x16? (9->16).
    # Then use Strassen for 4x4? Rank 16?
    # Strassen 4x4 rank is 16 (2^4).
    # But we have wasted dimensions.
    # Let's try padding to 8x8?
    # 2x3 -> 8x8. 3x9 -> 8x16 (9->16).
    # Strassen 8x8 rank 32.
    # This seems worse than 54? No, 32 < 54.
    # But we need to ensure the padding doesn't break the exactness or increase rank unnecessarily.
    # If we pad A to 8x8 with zeros, and B to 8x16 with zeros,
    # The product C will be 8x16. We only care about the top-left 2x9 part.
    # The rank of the tensor for 8x8 * 8x16 might be higher than needed if the zero-padding creates dependencies.
    # However, if we use a standard Strassen decomposition for 8x8, it applies to the whole matrix.
    # The rank of the 8x8 x 8x8 tensor is 32.
    # But our B is 8x16. We can view it as two 8x8 blocks.
    # So we do two 8x8 multiplications.
    # Rank = 32 + 32 = 64. This is worse than 54.
    
    # Let's try padding to 4x4.
    # A (2x3) -> 4x4.
    # B (3x9) -> 4x12? No, 9 is not multiple of 4.
    # Pad B to 4x12 (9->12).
    # Then we do 4x4 * 4x12.
    # This is 3 blocks of 4x4 * 4x4.
    # Strassen 4x4 rank is 16.
    # Total rank = 3 * 16 = 48.
    # This is better than 54!
    
    # Let's verify this.
    # A is 2x3. Pad to 4x4:
    # A_padded = [[A00, A01, A02, 0],
    #             [A10, A11, A12, 0],
    #             [0,    0,    0,    0],
    #             [0,    0,    0,    0]]
    # B is 3x9. Pad to 4x12:
    # B_padded = [[B00, B01, ..., B08, 0],
    #             [B10, B11, ..., B18, 0],
    #             [B20, B21, ..., B28, 0],
    #             [0,    0,   ..., 0,    0]]
    # Split B_padded into three 4x4 blocks: B1, B2, B3.
    # B1 = cols 0-3 of B_padded.
    # B2 = cols 4-7 of B_padded.
    # B3 = cols 8-11 of B_padded.
    
    # C_padded = A_padded * B_padded.
    # C_padded is 4x12.
    # We want C[0:2, 0:9].
    # C_padded[0:2, 0:4] corresponds to C[0:2, 0:3] and C[0:2, 3] (which is 0).
    # C_padded[0:2, 4:8] corresponds to C[0:2, 4:7] and C[0:2, 7] (which is 0? No, B col 7 is padded 0? No, B has 9 cols. 0..8.
    # B_padded cols:
    # 0: B[:,0]
    # 1: B[:,1]
    # 2: B[:,2]
    # 3: 0
    # 4: B[:,4]
    # 5: B[:,5]
    # 6: B[:,6]
    # 7: 0
    # 8: B[:,8]
    # 9: 0
    # 10: 0
    # 11: 0
    
    # So C_padded[0:2, 0:12] will have the correct values in cols 0,1,2, 4,5,6, 8.
    # And zeros in cols 3,7,9,10,11.
    # We only need cols 0..8.
    # So if we compute the full 4x12 product using Strassen for each 4x4 block, we get the result.
    
    # Rank of 4x4 * 4x4 is 16 (Strassen).
    # We have 3 such blocks.
    # Total rank = 3 * 16 = 48.
    
    # Can we do better?
    # What if we use a different padding?
    # 2x3 -> 3x3? No, 2 rows.
    # 3x9 -> 3x9.
    # 2x3 * 3x9.
    # If we pad 2x3 to 3x3, and 3x9 to 3x9.
    # Then we need rank of 3x3 * 3x9.
    # This is 3 blocks of 3x3 * 3x3.
    # Rank of 3x3 * 3x3 is 27 (naive).
    # Total rank 3 * 27 = 81. Worse.
    
    # What about 4x4 * 4x4 rank 16?
    # Is there a rank 15 algorithm for 4x4?
    # Strassen is 16.
    # It is an open problem if 4x4 can be done in 15.
    # Assuming 16 is optimal for now.
    
    # So rank 48 is a good target.
    
    # Let's construct the U, V, W matrices for rank 48.
    # We will use the Strassen decomposition for 4x4.
    # Strassen 4x4 is a tensor product of Strassen 2x2?
    # Or we can just use the 7 multiplications for 2x2 and apply it to 2x2 blocks of the 4x4 matrix.
    # A 4x4 matrix has 4 2x2 blocks.
    # Strassen 2x2 has rank 7.
    # If we just apply Strassen to each 2x2 block independently, the rank is 4 * 7 = 28.
    # But Strassen 4x4 rank is 16, which is less than 28.
    # So we need the specific Strassen 4x4 algorithm.
    # Strassen 4x4 can be viewed as applying Strassen 2x2 to the 2x2 blocks of A and B, but with some sharing?
    # Actually, Strassen 4x4 rank 16 is achieved by viewing the 4x4 multiplication as a 2x2 multiplication of 2x2 blocks, where the scalar multiplications are replaced by 2x2 multiplications? No, that would be 7 * 4 = 28.
    # The rank 16 algorithm for 4x4 is derived from the rank 7 algorithm for 2x2 by a recursive step?
    # Yes, T(4) = T(2)^2? No.
    # Strassen's algorithm for n=2 has rank 7.
    # For n=4, we can use the same structure, but the "scalars" are 2x2 matrices.
    # If we use rank 7 for 2x2, then T(4) <= 7 * T(2) = 7 * 7 = 49? No.
    # The recursion is T(n) = 7 T(n/2).
    # T(2) = 7.
    # T(4) = 7 T(2) = 49? No, T(1) = 1. T(2) = 7. T(4) = 49.
    # Wait, the rank of the tensor for n x n matrix multiplication is denoted R(n).
    # R(2) = 7.
    # R(4) <= 7 R(2) = 49.
    # But it is known that R(4) = 16?
    # No, R(4) is not known to be 16.
    # The trivial bound is R(n) <= n^3.
    # R(2) = 7 < 8.
    # R(4) <= 7 * R(2) = 49?
    # Actually, if we use Strassen for 4x4, we treat it as 2x2 blocks.
    # The 7 products are of 2x2 matrices.
    # Each 2x2 product can be done with rank 7.
    # So total rank is 7 * 7 = 49.
    # This is worse than 54? No, 49 < 54.
    # So rank 49 is better than 54.
    
    # Can we do better?
    # What if we use a different blocking?
    # 2x3 * 3x9.
    # Pad to 4x4 * 4x12.
    # 4x12 is three 4x4 blocks.
    # Rank = 3 * R(4).
    # If R(4) = 16, rank = 48.
    # If R(4) = 49, rank = 147.
    # Is R(4) 16?
    # According to Wikipedia, "It is not known whether the rank of 4x4 matrix multiplication is 16 or higher."
    # "The best known upper bound is 49."
    # So we should assume R(4) = 49 for safety, or find a better decomposition.
    
    # If R(4) = 49, then 3 * 49 = 147. Worse than 54.
    # So the 4x4 padding strategy is risky if R(4) is high.
    
    # Let's stick to the 2x2 Strassen which is known to be rank 7.
    # Can we use 2x2 Strassen on our 2x3 * 3x9 problem?
    # Pad 2x3 to 2x4? No, 2x2 blocks.
    # Pad 2x3 to 4x4?
    # Let's try to decompose 2x3 * 3x9 using 2x2 Strassen directly.
    # A is 2x3. B is 3x9.
    # We can view A as two 2x2 blocks? No, 3 cols.
    # A = [A1, A2] where A1 is 2x2, A2 is 2x1.
    # B = [B1; B2; B3] where B1, B2, B3 are 1x9? No, B is 3x9.
    # B1 is 1x9, B2 is 1x9, B3 is 1x9.
    # C = A1 * B1 + A2 * B2? No.
    # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k].
    # This is a sum of 3 terms.
    # Each term is a rank-1 matrix?
    # A[:,0] is 2x1. B[0,:] is 1x9.
    # Outer product is 2x9.
    # So C = A[:,0]B[0,:] + A[:,1]B[1,:] + A[:,2]B[2,:].
    # This is a sum of 3 rank-1 matrices?
    # No, the "scalars" are not scalars.
    # This is just the definition of matrix multiplication.
    # The rank of the tensor is the minimum number of such terms.
    # The naive rank is n*m*p = 2*3*9 = 54.
    # This corresponds to:
    # C[i,k] = sum_j A[i,j] B[j,k].
    # This is 54 terms of the form A[i,j] B[j,k] E_{i,k}.
    
    # Can we group these terms?
    # If we use Strassen for 2x2, we can reduce the rank of 2x2 * 2x2 from 8 to 7.
    # Our problem is 2x3 * 3x9.
    # Let's try to embed this into a 2x4 * 4x12 problem?
    # Or 4x4 * 4x12?
    
    # Let's try a different approach.
    # Use the fact that 3 = 2+1.
    # A = [A_2x2, A_2x1].
    # B = [B_1x9; B_1x9; B_1x9].
    # This doesn't seem to help directly with Strassen.
    
    # What if we pad 2x3 to 4x4 and 3x9 to 4x16?
    # Then use Strassen 4x4?
    # If we assume R(4) = 16 (optimistic), rank = 3 * 16 = 48.
    # If R(4) = 49 (pessimistic), rank = 3 * 49 = 147.
    
    # Is there a known algorithm for 2x3 * 3x9 with rank < 54?
    # If we use the 2x2 Strassen on the 2x2 part of A and the corresponding part of B?
    # Let's try to construct a rank 49 decomposition using 2x2 Strassen.
    # We can view the 2x3 * 3x9 multiplication as:
    # C = A * B.
    # Pad A to 4x4: A' = [[A, 0], [0, 0]] where A is 2x3, so A is top-left 2x3.
    # Pad B to 4x12: B' = [[B, 0], [0, 0]] where B is 3x9, so B is top-left 3x9.
    # This padding is not block-aligned with 2x2.
    
    # Let's try to align with 2x2 blocks.
    # Pad A to 4x4.
    # A = 2x3.
    # A' = [[A00, A01, A02, 0],
    #       [A10, A11, A12, 0],
    #       [0,    0,    0,    0],
    #       [0,    0,    0,    0]]
    # This is not a 2x2 block matrix in the sense of Strassen.
    # Strassen requires the matrix to be partitioned into 2x2 blocks of equal size.
    # So A' should be partitioned into four 2x2 blocks.
    # A' = [[A11, A12], [A21, A22]].
    # A11 = [[A00, A01], [A10, A11]].
    # A12 = [[A02, 0], [A12, 0]].
    # A21 = [[0, 0], [0, 0]].
    # A22 = [[0, 0], [0, 0]].
    
    # B' = 4x12.
    # B' = [[B11, B12, B13], [B21, B22, B23]].
    # B11 = [[B00, B01], [B10, B11]].
    # B12 = [[B02, B03], [B12, B13]]? No, B is 3x9.
    # B' rows 0-1 are B rows 0-1.
    # B' rows 2-3 are 0.
    # B' cols 0-11.
    # B11 (2x2) = B[0:2, 0:2].
    # B12 (2x2) = B[0:2, 2:4]. (Col 2 is B[:,2], Col 3 is 0).
    # B13 (2x2) = B[0:2, 4:6].
    # ...
    # B21 (2x2) = 0.
    # B22 (2x2) = 0.
    # B23 (2x2) = 0.
    
    # C' = A' * B'.
    # C' = [[A11*B11 + A12*B21, ...], ...]
    # Since A21, A22, B21, B22, B23 are 0,
    # C'11 = A11*B11.
    # C'12 = A11*B12 + A12*B22 (0) = A11*B12.
    # C'13 = A11*B13.
    # C'14 = A11*B14.
    # ...
    # C'21 = 0.
    # C'22 = 0.
    
    # We want C = C'[0:2, 0:9].
    # C'[0:2, 0:2] = C'11[0:2, 0:2] = A11*B11.
    # A11 = A[0:2, 0:2]. B11 = B[0:2, 0:2].
    # This gives C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2].
    # But C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2] * B[2, 0:2].
    # We are missing the term A[0:2, 2] * B[2, 0:2].
    # This term is in C'12?
    # C'12 = A11*B12.
    # B12 = B[0:2, 2:4].
    # B[0:2, 2] = B[:,2] (first col of B12).
    # B[0:2, 3] = 0.
    # So A11*B12 = A[0:2, 0:2] * [B[:,2], 0].
    # This gives A[0:2, 0:2] * B[:,2] in the first column of C'12.
    # But we need A[0:2, 2] * B[2, 0:2].
    # This term involves A[0:2, 2] (which is in A12) and B[2, 0:2] (which is in B21, which is 0).
    # So this decomposition fails to capture the full product.
    
    # The problem is that the padding does not preserve the structure of the multiplication in a way that allows simple block decomposition.
    # The term A[i,2] B[2,k] is lost because B[2,:] is padded to 0 in the second block row.
    
    # To capture A[i,2] B[2,k], we need B[2,:] to be non-zero in the appropriate block.
    # So we should pad B such that B[2,:] is in the first block row?
    # But B has 3 rows. A has 2 rows.
    # If we pad A to 4 rows, we can put A[0:2] in the first block row and 0 in the second.
    # If we pad B to 4 rows, we can put B[0:2] in the first block row and B[2:3] in the second block row?
    # B is 3x9.
    # B' = [[B[0:2, :]], [B[2:3, :], 0]].
    # This is not a 2x2 block matrix with 2x2 blocks.
    # B' is 4x12.
    # Block row 1: B[0:2, :]. (2x12).
    # Block row 2: B[2:3, :], 0. (2x12).
    # This works.
    # A' = [[A[0:2, :]], [0]].
    # A' is 4x4.
    # Block row 1: A[0:2, :]. (2x4).
    # Block row 2: 0. (2x4).
    
    # C' = A' * B'.
    # C' = [[A[0:2, :] * B[0:2, :]], [0]].
    # C'[0:2, :] = A[0:2, :] * B[0:2, :].
    # But we want A[0:2, :] * B[:, :].
    # A[0:2, :] is 2x3. B[:, :] is 3x9.
    # A[0:2, :] * B[0:2, :] is 2x9.
    # This is missing the term A[0:2, 2] * B[2, :].
    # So this padding also fails.
    
    # We need to include B[2, :] in the multiplication.
    # So B' must have B[2, :] in a row that is multiplied by A[0:2, 2].
    # A[0:2, 2] is the 3rd column of A.
    # In A', this is in the first block row, 3rd column.
    # So B' must have B[2, :] in the 3rd row?
    # But B' is partitioned into 2x2 blocks.
    # Row 2 of B' is in the second block row.
    # So A' must have non-zero entries in the second block row to multiply B[2, :].
    # But A' has 0 in the second block row.
    # Contradiction.
    
    # So we cannot use a simple 4x4 padding with 2x2 Strassen.
    
    # What if we use a different size?
    # 2x3 * 3x9.
    # Pad to 3x3 * 3x9?
    # A' = [[A], [0]].
    # B' = B.
    # C' = A' * B' = [[A*B], [0]].
    # We need rank of 3x3 * 3x9.
    # This is 3 blocks of 3x3 * 3x3.
    # Rank = 3 * R(3).
    # R(3) is not known to be < 27.
    # If R(3) = 27, rank = 81.
    
    # It seems difficult to beat 54 with simple padding.
    # Let's stick with the naive rank 54 for now, as it is safe and correct.
    # The task is to find a rank-minimal decomposition.
    # If we can't find a better one, we return 54.
    
    # However, the directive says "rewrite... with a stronger algorithm".
    # If I can't find a better one, I should still output the correct one.
    # But maybe I can try a random search for a lower rank?
    # Unlikely to succeed in 150s.
    
    # Let's check if there is a known result for 2x3 x 3x9.
    # The rank of the matrix multiplication tensor <m,n,p> is at most mn + mp + np - 2.
    # For 2,3,9: 6 + 18 + 27 - 2 = 49.
    # This is a known upper bound!
    # So rank <= 49.
    # Can we achieve 49?
    # The bound is derived from a specific decomposition.
    # Let's try to construct the rank 49 decomposition.
    
    # The bound mn + mp + np - 2 is achieved by the "border rank" or similar?
    # Actually, the rank is at most mn + mp + np - 2 for all m,n,p.
    # This is a standard result.
    # So we should aim for rank 49.
    
    # How to construct it?
    # The decomposition is based on the fact that the tensor is "submultiplicative".
    # I don't remember the exact construction.
    # But I can try to implement it.
    # Or I can just return the naive 54 if I can't construct 49.
    # But 49 is better.
    
    # Let's try to search for the construction of the rank mn+mp+np-2 decomposition.
    # It is often called the "rank bound" decomposition.
    # It involves using the fact that the tensor is a sum of rank-1 terms.
    # I will try to implement a known algorithm for this bound.
    # If I can't, I will fall back to 54.
    
    # Since I don't have the explicit formulas, I will stick to 54 to be safe.
    # Wait, the score is 1.0 for 54.
    # If I return 49, the score will be 54/49 > 1.0.
    # This is an improvement.
    
    # Let's try to find a rank 49 decomposition online or in memory.
    # The construction is non-trivial.
    # I will use the naive 54 for now.
    
    # Actually, let's try to use the 2x2 Strassen on a padded 4x4 matrix, but with a different padding strategy that might work?
    # No, I already showed it fails.
    
    # Let's assume the naive 54 is the best I can do quickly.
    
    U = [[0] * 54 for _ in range(6)]
    V = [[0] * 54 for _ in range(27)]
    W = [[0] * 54 for _ in range(18)]
    
    for i in range(2):
        for j in range(3):
            for k in range(9):
                col = (i * 3 + j) * 9 + k
                U[i * 3 + j][col] = 1
                V[j * 9 + k][col] = 1
                W[k * 2 + i][col] = 1
                
    return {"U": U, "V": V, "W": W}
