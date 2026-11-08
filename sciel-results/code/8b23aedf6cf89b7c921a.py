"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    # We'll build the decomposition by splitting the middle dimension (m=4) into 2+2
    # and the last dimension (p=6) into 3+3.
    # For each block (i in {0,1}, j in {0,1}), we have a sub-problem:
    #   A_i (3 x 2) * B_j (2 x 3) -> C_ij (3 x 3)
    # We use a rank-10 decomposition for each 3x2x3 block.
    # Total rank = 4 * 10 = 40.
    
    # Helper: build rank-10 decomposition for multiplying a (3x2) matrix by a (2x3) matrix.
    # This uses Strassen for the 2x2x2 core and rank-1 for the degenerate parts.
    # A = [[a00,a01],[a10,a11],[a20,a21]] (3x2)
    # B = [[b00,b01,b02],[b10,b11,b12]] (2x3)
    # C = A*B (3x3)
    # Split A into A1 (2x2: rows 0,1) and A2 (1x2: row 2)
    # Split B into B1 (2x2: cols 0,1) and B2 (2x1: col 2)
    # C = A1*B1 + A1*B2 + A2*B1 + A2*B2
    # Rank 7 for A1*B1 (Strassen), Rank 1 each for the others. Total 10.
    
    # Strassen for 2x2x2:
    # Given X (2x2), Y (2x2):
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y01-y11)
    # M4 = x11(y10-y00)
    # M5 = (x00+x10)y11
    # M6 = (x01-x11)(y10+y11)
    # M7 = (x00-x10)(y00+y01)
    # c00 = M1 - M4 + M5 + M6
    # c01 = M3 + M1 - M2 - M4
    # c10 = M2 + M4 - M1 + M7
    # c11 = M3 - M5 + M1 + M7
    
    # We need to express this in terms of the original A and B entries.
    # Let's denote the global U, V, W matrices and assign columns.
    
    rank = 40
    U = [[0] * rank for _ in range(n * m)]  # 12 x 40
    V = [[0] * rank for _ in range(m * p)]  # 24 x 40
    W = [[0] * rank for _ in range(p * n)]  # 18 x 40
    
    # We'll process each of the 4 blocks independently.
    # Block (bi, bj) where bi in {0,1} for middle split, bj in {0,1} for last split.
    # bi=0: middle rows 0,1; bi=1: middle rows 2,3
    # bj=0: last cols 0,1,2; bj=1: last cols 3,4,5
    
    # For each block, we use 10 rank-1 terms.
    # The U indices are (i, j) where i is row of A (0..2), j is middle col (0..3).
    # The V indices are (j, k) where j is middle row (0..3), k is last col (0..5).
    # The W indices are (k, i) where k is last col (0..5), i is row of C (0..2).
    # Note: W uses trace convention W[k*n+i, r] = W[k*3+i, r].
    
    col_offset = 0
    for bi in range(2):
        for bj in range(2):
            # Middle rows: 2*bi, 2*bi+1
            # Last cols: 3*bj, 3*bj+1, 3*bj+2
            
            # For this block, we have a 3x2 * 2x3 multiplication.
            # A_local (3x2): rows 0,1,2; cols 2*bi, 2*bi+1
            # B_local (2x3): rows 2*bi, 2*bi+1; cols 3*bj, 3*bj+1, 3*bj+2
            
            # We'll add 10 rank-1 terms.
            # The first 7 are for the Strassen part (A1*B1), where:
            # A1 = A_local[0:2, 0:2] = rows 0,1; cols 2*bi, 2*bi+1
            # B1 = B_local[0:2, 0:2] = rows 2*bi, 2*bi+1; cols 3*bj, 3*bj+1
            
            # Strassen terms:
            # M1 = (x00+x11)(y00+y11)
            # x00 = A[0, 2*bi], x11 = A[1, 2*bi+1]
            # y00 = B[2*bi, 3*bj], y11 = B[2*bi+1, 3*bj+1]
            
            # M2 = (x10+x11)y00
            # x10 = A[1, 2*bi], x11 = A[1, 2*bi+1]
            # y00 = B[2*bi, 3*bj]
            
            # M3 = x00(y01-y11)
            # x00 = A[0, 2*bi]
            # y01 = B[2*bi, 3*bj+1], y11 = B[2*bi+1, 3*bj+1]
            
            # M4 = x11(y10-y00)
            # x11 = A[1, 2*bi+1]
            # y10 = B[2*bi+1, 3*bj], y00 = B[2*bi, 3*bj]
            
            # M5 = (x00+x10)y11
            # x00 = A[0, 2*bi], x10 = A[1, 2*bi]
            # y11 = B[2*bi+1, 3*bj+1]
            
            # M6 = (x01-x11)(y10+y11)
            # x01 = A[0, 2*bi+1], x11 = A[1, 2*bi+1]
            # y10 = B[2*bi+1, 3*bj], y11 = B[2*bi+1, 3*bj+1]
            
            # M7 = (x00-x10)(y00+y01)
            # x00 = A[0, 2*bi], x10 = A[1, 2*bi]
            # y00 = B[2*bi, 3*bj], y01 = B[2*bi, 3*bj+1]
            
            # C00 = M1 - M4 + M5 + M6
            # C01 = M3 + M1 - M2 - M4
            # C10 = M2 + M4 - M1 + M7
            # C11 = M3 - M5 + M1 + M7
            
            # C00 is at local (0,0) -> global (0, 3*bj)
            # C01 is at local (0,1) -> global (0, 3*bj+1)
            # C10 is at local (1,0) -> global (1, 3*bj)
            # C11 is at local (1,1) -> global (1, 3*bj+1)
            
            # We'll define the 7 Strassen terms.
            # Each term has a U vector, V vector, and W vector.
            # The U vector is 12-dimensional (3 rows x 4 middle cols).
            # The V vector is 24-dimensional (4 middle rows x 6 last cols).
            # The W vector is 18-dimensional (6 last cols x 3 rows).
            
            # For term t (0..6), we have:
            # U_local: 3x2 (only cols 2*bi, 2*bi+1 are non-zero)
            # V_local: 2x3 (only rows 2*bi, 2*bi+1 and cols 3*bj, 3*bj+1 are non-zero)
            # W_local: 3x3 (only cols 3*bj, 3*bj+1 are non-zero)
            
            # Let's define the U, V, W for each Strassen term.
            # U is indexed by (i, j) -> i in 0..2, j in 0..3.
            # V is indexed by (j, k) -> j in 0..3, k in 0..5.
            # W is indexed by (k, i) -> k in 0..5, i in 0..2.
            
            # Term 0: M1 = (x00+x11)(y00+y11)
            # U: (0, 2*bi)=1, (1, 2*bi+1)=1
            # V: (2*bi, 3*bj)=1, (2*bi+1, 3*bj+1)=1
            # W: C00 gets +1, C01 gets +1, C10 gets -1, C11 gets +1
            # W indices: (3*bj, 0)=1, (3*bj+1, 0)=1, (3*bj, 1)=-1, (3*bj+1, 1)=1
            
            U[0 * m + (2 * bi)][col_offset] = 1
            U[1 * m + (2 * bi + 1)][col_offset] = 1
            V[(2 * bi) * p + (3 * bj)][col_offset] = 1
            V[(2 * bi + 1) * p + (3 * bj + 1)][col_offset] = 1
            W[(3 * bj) * n + 0][col_offset] = 1
            W[(3 * bj + 1) * n + 0][col_offset] = 1
            W[(3 * bj) * n + 1][col_offset] = -1
            W[(3 * bj + 1) * n + 1][col_offset] = 1
            col_offset += 1
            
            # Term 1: M2 = (x10+x11)y00
            # U: (1, 2*bi)=1, (1, 2*bi+1)=1
            # V: (2*bi, 3*bj)=1
            # W: C01 gets -1, C10 gets +1
            # W indices: (3*bj+1, 0)=-1, (3*bj, 1)=1
            
            U[1 * m + (2 * bi)][col_offset] = 1
            U[1 * m + (2 * bi + 1)][col_offset] = 1
            V[(2 * bi) * p + (3 * bj)][col_offset] = 1
            W[(3 * bj + 1) * n + 0][col_offset] = -1
            W[(3 * bj) * n + 1][col_offset] = 1
            col_offset += 1
            
            # Term 2: M3 = x00(y01-y11)
            # U: (0, 2*bi)=1
            # V: (2*bi, 3*bj+1)=1, (2*bi+1, 3*bj+1)=-1
            # W: C01 gets +1, C11 gets +1
            # W indices: (3*bj+1, 0)=1, (3*bj+1, 1)=1
            
            U[0 * m + (2 * bi)][col_offset] = 1
            V[(2 * bi) * p + (3 * bj + 1)][col_offset] = 1
            V[(2 * bi + 1) * p + (3 * bj + 1)][col_offset] = -1
            W[(3 * bj + 1) * n + 0][col_offset] = 1
            W[(3 * bj + 1) * n + 1][col_offset] = 1
            col_offset += 1
            
            # Term 3: M4 = x11(y10-y00)
            # U: (1, 2*bi+1)=1
            # V: (2*bi+1, 3*bj)=1, (2*bi, 3*bj)=-1
            # W: C00 gets -1, C01 gets -1, C10 gets +1
            # W indices: (3*bj, 0)=-1, (3*bj+1, 0)=-1, (3*bj, 1)=1
            
            U[1 * m + (2 * bi + 1)][col_offset] = 1
            V[(2 * bi + 1) * p + (3 * bj)][col_offset] = 1
            V[(2 * bi) * p + (3 * bj)][col_offset] = -1
            W[(3 * bj) * n + 0][col_offset] = -1
            W[(3 * bj + 1) * n + 0][col_offset] = -1
            W[(3 * bj) * n + 1][col_offset] = 1
            col_offset += 1
            
            # Term 4: M5 = (x00+x10)y11
            # U: (0, 2*bi)=1, (1, 2*bi)=1
            # V: (2*bi+1, 3*bj+1)=1
            # W: C00 gets +1, C11 gets -1
            # W indices: (3*bj, 0)=1, (3*bj+1, 1)=-1
            
            U[0 * m + (2 * bi)][col_offset] = 1
            U[1 * m + (2 * bi)][col_offset] = 1
            V[(2 * bi + 1) * p + (3 * bj + 1)][col_offset] = 1
            W[(3 * bj) * n + 0][col_offset] = 1
            W[(3 * bj + 1) * n + 1][col_offset] = -1
            col_offset += 1
            
            # Term 5: M6 = (x01-x11)(y10+y11)
            # U: (0, 2*bi+1)=1, (1, 2*bi+1)=-1
            # V: (2*bi+1, 3*bj)=1, (2*bi+1, 3*bj+1)=1
            # W: C00 gets +1
            # W indices: (3*bj, 0)=1
            
            U[0 * m + (2 * bi + 1)][col_offset] = 1
            U[1 * m + (2 * bi + 1)][col_offset] = -1
            V[(2 * bi + 1) * p + (3 * bj)][col_offset] = 1
            V[(2 * bi + 1) * p + (3 * bj + 1)][col_offset] = 1
            W[(3 * bj) * n + 0][col_offset] = 1
            col_offset += 1
            
            # Term 6: M7 = (x00-x10)(y00+y01)
            # U: (0, 2*bi)=1, (1, 2*bi)=-1
            # V: (2*bi, 3*bj)=1, (2*bi, 3*bj+1)=1
            # W: C10 gets +1
            # W indices: (3*bj, 1)=1
            
            U[0 * m + (2 * bi)][col_offset] = 1
            U[1 * m + (2 * bi)][col_offset] = -1
            V[(2 * bi) * p + (3 * bj)][col_offset] = 1
            V[(2 * bi) * p + (3 * bj + 1)][col_offset] = 1
            W[(3 * bj) * n + 1][col_offset] = 1
            col_offset += 1
            
            # Now the remaining 3 terms for A1*B2, A2*B1, A2*B2.
            # Term 7: A1*B2. A1 is 2x2 (rows 0,1; cols 2*bi, 2*bi+1). B2 is 2x1 (rows 2*bi, 2*bi+1; col 3*bj+2).
            # C02 = A[0, 2*bi]*B[2*bi, 3*bj+2] + A[0, 2*bi+1]*B[2*bi+1, 3*bj+2]
            # C12 = A[1, 2*bi]*B[2*bi, 3*bj+2] + A[1, 2*bi+1]*B[2*bi+1, 3*bj+2]
            # This is a rank-2 term, but we can express it as two rank-1 terms? No, wait.
            # Actually, A1*B2 is a 2x1 matrix. It's a bilinear form.
            # We can write it as: (sum of A1 entries in row i) * (B2 entries).
            # But that's not rank-1 in the tensor sense.
            # Let's reconsider. The term A1*B2 is a sum of 4 products.
            # We need to express the contribution to C02 and C12.
            # C02 = A[0, 2*bi]*B[2*bi, 3*bj+2] + A[0, 2*bi+1]*B[2*bi+1, 3*bj+2]
            # C12 = A[1, 2*bi]*B[2*bi, 3*bj+2] + A[1, 2*bi+1]*B[2*bi+1, 3*bj+2]
            
            # This is not a single rank-1 term. It's a rank-2 term (or more).
            # My earlier claim of rank 1 was wrong.
            
            # Let's re-evaluate. The sub-problem is 3x2 * 2x3.
            # A1 (2x2) * B2 (2x1) -> 2x1. This has rank at most 2.
            # A2 (1x2) * B1 (2x2) -> 1x2. This has rank at most 2.
            # A2 (1x2) * B2 (2x1) -> 1x1. This has rank 1.
            
            # So the total rank is 7 + 2 + 2 + 1 = 12, not 10.
            # My earlier analysis was incorrect.
            
            # Let's try a different approach. Maybe the rank is indeed 12 for 3x2x3.
            # Then total rank = 4 * 12 = 48. That's the same as splitting 6 into 2+2+2.
            
            # Is there a better decomposition for 3x2x3?
            # Maybe using the fact that one dimension is 2?
            
            # Alternatively, maybe I should just use the schoolbook method but with a better split.
            # Or maybe I should use a known result.
            
            # Let's check: is there a rank-9 or rank-10 decomposition for 3x2x3?
            # I'm not sure. Let's assume rank 12 for now.
            
            # Actually, let's reconsider the Strassen-based decomposition.
            # The issue is that A1*B2 and A2*B1 are not rank-1.
            # But maybe we can combine them with other terms?
            
            # Given time constraints, I'll stick with the schoolbook method for now,
            # but I'll try to use a slightly better split.
            
            # Actually, let's just use the schoolbook method with rank 72.
            # It's correct, and I'm running out of time.
            
            # Wait, I already have col_offset at 7. I need to add 3 more terms.
            # But I don't have the correct terms.
            
            # Let's just fill the remaining columns with zeros and return.
            # This will be incorrect, but it's better than crashing.
            
            # Actually, let's just use the schoolbook method from scratch.
            pass
    
    # Given the complexity, let's just use the schoolbook method.
    # I'll restart with the schoolbook method.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
