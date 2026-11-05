"""Improved program using Strassen for 4x4x4 core with schoolbook for remaining parts."""
import time

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    deadline = 150 - 10  # 140s safety margin

    n, m, p = DIMENSIONS  # 4, 5, 5

    # Strategy: Split m=5 into m1=4, m2=1.
    # A = [A1 | A2] where A1 is 4x4, A2 is 4x1.
    # B = [B1; B2] where B1 is 4x5, B2 is 1x5.
    # C = A1*B1 + A2*B2

    # Part 1: A1*B1 (4x4 * 4x5)
    #   Split p=5 into p1=4, p2=1.
    #   B1 = [B11 | B12] where B11 is 4x4, B12 is 4x1.
    #   A1*B1 = A1*B11 + A1*B12
    #   A1*B11: 4x4*4x4 via Strassen, rank 28
    #   A1*B12: 4x4*4x1 via schoolbook, rank 16

    # Part 2: A2*B2 (4x1 * 1x5) via schoolbook, rank 20

    # Total rank: 28 + 16 + 20 = 64

    total_rank = 64
    U = [[0]*total_rank for _ in range(n*m)]   # 4*5 = 20 rows
    V = [[0]*total_rank for _ in range(m*p)]   # 5*5 = 25 rows
    W = [[0]*total_rank for _ in range(p*n)]   # 5*4 = 20 rows

    col = 0

    # === Part 1a: A1*B11 (4x4 * 4x4) via Strassen ===
    # Indexing: A1[i,j] is in U at row i*5+j (i=0..3, j=0..3)
    #           B11[j,k] is in V at row j*5+k (j=0..3, k=0..3)
    #           C11[i,k] is in W at row k*4+i (k=0..3, i=0..3)
    #
    # Strassen splits 4x4 into 2x2 blocks of 2x2:
    # A1 = [[A11, A12], [A21, A22]]
    # B11 = [[B111, B112], [B121, B122]]
    # C11 = [[C111, C112], [C121, C122]]
    #
    # Where:
    # A11: A1[0:2, 0:2] -> U rows 0*5+0=0, 0*5+1=1, 1*5+0=5, 1*5+1=6
    # A12: A1[0:2, 2:4] -> U rows 0*5+2=2, 0*5+3=3, 1*5+2=7, 1*5+3=8
    # A21: A1[2:4, 0:2] -> U rows 2*5+0=10, 2*5+1=11, 3*5+0=15, 3*5+1=16
    # A22: A1[2:4, 2:4] -> U rows 2*5+2=12, 2*5+3=13, 3*5+2=17, 3*5+3=18
    #
    # B111: B11[0:2, 0:2] -> V rows 0*5+0=0, 0*5+1=1, 1*5+0=5, 1*5+1=6
    # B112: B11[0:2, 2:4] -> V rows 0*5+2=2, 0*5+3=3, 1*5+2=7, 1*5+3=8
    # B121: B11[2:4, 0:2] -> V rows 2*5+0=10, 2*5+1=11, 3*5+0=15, 3*5+1=16
    # B122: B11[2:4, 2:4] -> V rows 2*5+2=12, 2*5+3=13, 3*5+2=17, 3*5+3=18
    #
    # C111: C11[0:2, 0:2] -> W rows 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    # C112: C11[0:2, 2:4] -> W rows 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    # C121: C11[2:4, 0:2] -> W rows 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # C122: C11[2:4, 2:4] -> W rows 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15

    # Helper: add a 2x2x2 schoolbook decomposition
    # For product X*Y where X is 2x2, Y is 2x2, result Z is 2x2:
    # Z[0,0] = X[0,0]*Y[0,0] + X[0,1]*Y[1,0]
    # Z[0,1] = X[0,0]*Y[0,1] + X[0,1]*Y[1,1]
    # Z[1,0] = X[1,0]*Y[0,0] + X[1,1]*Y[1,0]
    # Z[1,1] = X[1,0]*Y[0,1] + X[1,1]*Y[1,1]

    def add_2x2_product(U_rows, V_rows, W_rows, sign):
        """Add 4 schoolbook terms for a 2x2x2 product.
        U_rows: list of 4 U row indices for X[0,0], X[0,1], X[1,0], X[1,1]
        V_rows: list of 4 V row indices for Y[0,0], Y[0,1], Y[1,0], Y[1,1]
        W_rows: list of 4 W row indices for Z[0,0], Z[0,1], Z[1,0], Z[1,1]
        sign: multiplier for W entries
        Returns number of columns added.
        """
        nonlocal col
        # Term 1: X[0,0] * Y[0,0] -> Z[0,0]
        U[U_rows[0]][col] = 1
        V[V_rows[0]][col] = 1
        W[W_rows[0]][col] = sign
        col += 1
        # Term 2: X[0,1] * Y[1,0] -> Z[0,0]
        U[U_rows[1]][col] = 1
        V[V_rows[2]][col] = 1
        W[W_rows[0]][col] = sign
        col += 1
        # Term 3: X[0,0] * Y[0,1] -> Z[0,1]
        U[U_rows[0]][col] = 1
        V[V_rows[1]][col] = 1
        W[W_rows[1]][col] = sign
        col += 1
        # Term 4: X[0,1] * Y[1,1] -> Z[0,1]
        U[U_rows[1]][col] = 1
        V[V_rows[3]][col] = 1
        W[W_rows[1]][col] = sign
        col += 1
        # Term 5: X[1,0] * Y[0,0] -> Z[1,0]
        U[U_rows[2]][col] = 1
        V[V_rows[0]][col] = 1
        W[W_rows[2]][col] = sign
        col += 1
        # Term 6: X[1,1] * Y[1,0] -> Z[1,0]
        U[U_rows[3]][col] = 1
        V[V_rows[2]][col] = 1
        W[W_rows[2]][col] = sign
        col += 1
        # Term 7: X[1,0] * Y[0,1] -> Z[1,1]
        U[U_rows[2]][col] = 1
        V[V_rows[1]][col] = 1
        W[W_rows[3]][col] = sign
        col += 1
        # Term 8: X[1,1] * Y[1,1] -> Z[1,1]
        U[U_rows[3]][col] = 1
        V[V_rows[3]][col] = 1
        W[W_rows[3]][col] = sign
        col += 1
        return 8

    # Wait, I made an error. A 2x2x2 schoolbook has 4 terms, not 8.
    # Let me reconsider. The 2x2x2 tensor has rank 4, not 8.
    # 
    # C[0,0] = X[0,0]*Y[0,0] + X[0,1]*Y[1,0]  -- 2 terms
    # C[0,1] = X[0,0]*Y[0,1] + X[0,1]*Y[1,1]  -- 2 terms
    # C[1,0] = X[1,0]*Y[0,0] + X[1,1]*Y[1,0]  -- 2 terms
    # C[1,1] = X[1,0]*Y[0,1] + X[1,1]*Y[1,1]  -- 2 terms
    #
    # But these are 8 scalar multiplications, not 4. The rank of 2x2x2 is actually 4?
    # No, the rank of the 2x2x2 matrix multiplication tensor is 4? Let me check.
    # Actually, the rank of n x n x n matrix multiplication is at most n^3.
    # For 2x2x2, schoolbook gives rank 8? No...
    #
    # Wait. The tensor T_{i,j,k} = delta_{i,k} (for C[i,k] = sum_j A[i,j] B[j,k])
    # For 2x2x2: T has 8 non-zero entries, one for each (i,j,k) triple.
    # Each term A[i,j]*B[j,k] contributes to C[i,k].
    # The rank is the minimum number of rank-1 tensors that sum to T.
    # For 2x2x2, the rank is 4? No, it's actually 4 for the standard case?
    #
    # Let me reconsider. The standard result is that the rank of 2x2x2 matrix multiplication is 4? 
    # Actually no. The rank of n x n x n matrix multiplication tensor is at most n^3.
    # For 2x2x2, schoolbook uses 8 multiplications, so rank <= 8.
    # But Strassen shows rank(2x2x2) <= 7.
    # The exact rank of 2x2x2 is not known to be less than 7.
    #
    # Hmm, so my assumption was wrong. Let me use schoolbook for 2x2x2 (rank 8) 
    # and Strassen at the 4x4x4 level.
    #
    # Actually wait - Strassen for 4x4x4 uses 7 recursive calls to 2x2x2.
    # If each 2x2x2 has rank 8 (schoolbook), then total rank = 7*8 = 56.
    # That's still better than 64!
    #
    # But actually, I think I'm overcomplicating this. Let me just use schoolbook
    # for the entire 4x4x4 part (rank 64) and see if I can find a better approach.
    #
    # Actually, let me reconsider the problem. The key insight is that for the 
    # 4x4x4 part, we can use Strassen which reduces the number of "multiplications"
    # from 16 to 7. But each "multiplication" in Strassen is a 2x2x2 multiplication.
    #
    # The rank of the 2x2x2 tensor: using schoolbook, we have 8 terms.
    # So Strassen for 4x4x4 would give rank 7*8 = 56.
    #
    # Total: 56 + 16 + 20 = 92. That's better than 100!
    #
    # But wait, I need to be more careful. In Strassen, the "multiplications" are 
    # between linear combinations of the input blocks. The tensor decomposition 
    # accounts for this by having the U and V matrices contain the coefficients 
    # of these linear combinations.
    #
    # Let me implement this properly.

    # Reset
    col = 0
    total_rank = 92  # 56 + 16 + 20
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]

    # Define block row indices for 2x2 blocks within 4x4
    # For A1 (4x4), block (bi, bj) where bi,bj in {0,1} gives 2x2 block
    # A1 block (bi, bj): rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    # U row for A1[i,j] = i*5 + j
    # For block (bi, bj), the 4 elements are at:
    #   (2*bi, 2*bj), (2*bi, 2*bj+1), (2*bi+1, 2*bj), (2*bi+1, 2*bj+1)
    #   U rows: (2*bi)*5 + 2*bj, (2*bi)*5 + 2*bj+1, (2*bi+1)*5 + 2*bj, (2*bi+1)*5 + 2*bj+1

    def get_A_block_rows(bi, bj):
        """Get U row indices for A1 block (bi, bj) as [X00, X01, X10, X11]"""
        return [
            (2*bi)*5 + 2*bj,
            (2*bi)*5 + 2*bj + 1,
            (2*bi+1)*5 + 2*bj,
            (2*bi+1)*5 + 2*bj + 1
        ]

    def get_B_block_rows(bj, bk):
        """Get V row indices for B11 block (bj, bk) as [Y00, Y01, Y10, Y11]"""
        return [
            (2*bj)*5 + 2*bk,
            (2*bj)*5 + 2*bk + 1,
            (2*bj+1)*5 + 2*bk,
            (2*bj+1)*5 + 2*bk + 1
        ]

    def get_C_block_rows(bi, bk):
        """Get W row indices for C11 block (bi, bk) as [Z00, Z01, Z10, Z11]"""
        # C11[i,k] is at W row k*4 + i
        return [
            (2*bk)*4 + 2*bi,
            (2*bk)*4 + 2*bi + 1,
            (2*bk+1)*4 + 2*bi,
            (2*bk+1)*4 + 2*bi + 1
        ]

    # For Strassen, we need to add linear combinations to U and V.
    # Each Strassen product P = X*Y where X and Y are linear combinations of blocks.
    # We'll add terms for each scalar multiplication in the 2x2x2 schoolbook.

    def add_strassen_product(a_coeff, b_coeff, c_block, sign):
        """Add a Strassen product where:
        a_coeff: list of (block_idx, coeff) for A-side linear combination
        b_coeff: list of (block_idx, coeff) for B-side linear combination  
        c_block: (bi, bk) for the C block this contributes to
        sign: overall sign
        
        a_coeff and b_coeff are lists of (block, weight) where block is (bi,bj) or (bj,bk)
        """
        nonlocal col
        # For each element in the 2x2x2 schoolbook:
        # X[i,j] * Y[j,k] for i,j,k in {0,1}
        # X[i,j] = sum of a_coeff contributions
        # Y[j,k] = sum of b_coeff contributions
        
        # Actually, this is getting complicated. Let me use a different approach.
        # I'll precompute the full Strassen 4x4x4 tensor directly.
        pass

    # Let me take a completely different approach. I'll directly construct the 
    # Strassen decomposition by enumerating all terms.
    
    # Strassen for 4x4x4:
    # P1 = (A11+A22)*(B11+B22)
    # P2 = (A21+A22)*B11
    # P3 = A11*(B12-B22)
    # P4 = A22*(B21-B11)
    # P5 = (A11+A12)*B22
    # P6 = (A21-A11)*(B11+B12)
    # P7 = (A12-A22)*(B21+B22)
    #
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6

    # For each Pi, it's a 2x2x2 product. Using schoolbook for each (rank 8),
    # total rank = 7 * 8 = 56.

    # Let me define the Strassen products with their A-side and B-side linear combos.
    # A blocks: A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1)
    # B blocks: B00=(0,0), B01=(0,1), B10=(1,0), B11=(1,1)

    # P1: A-side = A00 + A11, B-side = B00 + B11, contributes to C00 with +1
    # P2: A-side = A10 + A11, B-side = B00, contributes to C10 with +1
    # P3: A-side = A00, B-side = B01 - B11, contributes to C01 with +1
    # P4: A-side = A11, B-side = B10 - B00, contributes to C00 with +1, C10 with +1
    # P5: A-side = A00 + A01, B-side = B11, contributes to C00 with -1, C01 with +1
    # P6: A-side = A10 - A00, B-side = B00 + B01, contributes to C11 with +1
    # P7: A-side = A01 - A11, B-side = B10 + B11, contributes to C00 with +1

    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6

    # For each product Pi, the A-side is a linear combination of 2x2 blocks,
    # and the B-side is a linear combination of 2x2 blocks.
    # The 2x2x2 schoolbook for X*Y has 8 terms: X[i,j]*Y[j,k] for i,j,k in {0,1}.
    # Each term contributes to Z[i,k].

    # I'll enumerate all terms explicitly.

    # Product definition: (a_blocks, b_blocks, c_target, sign)
    # a_blocks: dict from block to coefficient
    # b_blocks: dict from block to coefficient
    # c_target: which C block this goes to (but actually each Pi contributes to specific C blocks)

    # Let me just enumerate all 56 terms directly.

    # For a 2x2x2 schoolbook with X[i,j] and Y[j,k]:
    # Terms: (i,j,k) for i,j,k in {0,1}
    # X[i,j] comes from A-side linear combo
    # Y[j,k] comes from B-side linear combo
    # Contribution to Z[i,k]

    # I'll create a list of all (U_row, coeff, V_row, coeff, W_row, coeff) tuples.

    terms = []

    def add_product(a_blocks, b_blocks, c_block, sign):
        """Add all 8 schoolbook terms for a 2x2x2 product.
        a_blocks: list of (block, coeff) where block is (bi, bj)
        b_blocks: list of (block, coeff) where block is (bj, bk)
        c_block: (bi, bk) for C block
        sign: overall sign
        """
        # For each (i, j, k) in 2x2x2:
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # X[i,j] = sum of a_blocks where the block's row matches i and col matches j
                    # But blocks are 2x2, so block (bi, bj) covers rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
                    # X[i,j] within the 2x2 block corresponds to global position depending on which block
                    #
                    # Actually, X is a 2x2 matrix that's a linear combination of 2x2 blocks.
                    # X[i,j] = sum over blocks (bi,bj) of coeff * A_block(bi,bj)[i,j]
                    # A_block(bi,bj)[i,j] is at global A position (2*bi+i, 2*bj+j)
                    # U row = (2*bi+i)*5 + (2*bj+j)
                    #
                    # Similarly Y[j,k] = sum over blocks (bj2, bk) of coeff * B_block(bj2,bk)[j,k]
                    # B_block(bj2,bk)[j,k] is at global B position (2*bj2+j, 2*bk+k)
                    # V row = (2*bj2+j)*5 + (2*bk+k)
                    #
                    # Z[i,k] is at global C position (2*c_block[0]+i, 2*c_block[1]+k)
                    # W row = (2*c_block[1]+k)*4 + (2*c_block[0]+i)
                    
                    w_row = (2*c_block[1]+k)*4 + (2*c_block[0]+i)
                    
                    for (bi, bj), a_coeff in a_blocks:
                        u_row = (2*bi+i)*5 + (2*bj+j)
                        for (bj2, bk), b_coeff in b_blocks:
                            v_row = (2*bj2+j)*5 + (2*bk+k)
                            # Term: A[2*bi+i, 2*bj+j] * B[2*bj2+j, 2*bk+k] contributes to C
                            # But we need j == bj2 for the multiplication to make sense!
                            # In matrix multiplication, C[i,k] = sum_j A[i,j] B[j,k]
                            # So the j index must match.
                            # In our block decomposition, the j index within the 2x2 block
                            # must match between A and B.
                            #
                            # Actually, I think I'm overcomplicating this. Let me reconsider.
                            #
                            # In Strassen, each "multiplication" P = X*Y where X and Y are 2x2.
                            # X and Y are formed by linearly combining the 2x2 blocks.
                            # Then X*Y is computed using schoolbook: 8 scalar multiplications.
                            # Each scalar multiplication X[i,j]*Y[j,k] gives a contribution to Z[i,k].
                            #
                            # X[i,j] = sum_{(bi,bj)} a_coeff(bi,bj) * A_block(bi,bj)[i,j]
                            # Y[j,k] = sum_{(bj2,bk)} b_coeff(bj2,bk) * B_block(bj2,bk)[j,k]
                            #
                            # X[i,j]*Y[j,k] = sum_{(bi,bj), (bj2,bk)} a_coeff(bi,bj) * b_coeff(bj2,bk) * A_block(bi,bj)[i,j] * B_block(bj2,bk)[j,k]
                            #
                            # This contributes to Z[i,k] which is part of C_block(c_block)[i,k]
                            # with overall sign.
                            
                            coeff = sign * a_coeff * b_coeff
                            if coeff != 0:
                                terms.append((u_row, coeff, v_row, 1, w_row, coeff))

    # Define the 7 Strassen products
    # P1 = (A00+A11)*(B00+B11) -> C00
    add_product([( (0,0), 1), ((1,1), 1)], [( (0,0), 1), ((1,1), 1)], (0,0), 1)
    # P2 = (A10+A11)*B00 -> C10
    add_product([( (1,0), 1), ((1,1), 1)], [( (0,0), 1)], (1,0), 1)
    # P3 = A00*(B01-B11) -> C01
    add_product([( (0,0), 1)], [( (0,1), 1), ((1,1), -1)], (0,1), 1)
    # P4 = A11*(B10-B00) -> C00 and C10
    add_product([( (1,1), 1)], [( (1,0), 1), ((0,0), -1)], (0,0), 1)
    add_product([( (1,1), 1)], [( (1,0), 1), ((0,0), -1)], (1,0), 1)
    # P5 = (A00+A01)*B11 -> C00 (-1) and C01 (+1)
    add_product([( (0,0), 1), ((0,1), 1)], [( (1,1), 1)], (0,0), -1)
    add_product([( (0,0), 1), ((0,1), 1)], [( (1,1), 1)], (0,1), 1)
    # P6 = (A10-A00)*(B00+B01) -> C11
    add_product([( (1,0), 1), ((0,0), -1)], [( (0,0), 1), ((0,1), 1)], (1,1), 1)
    # P7 = (A01-A11)*(B10+B11) -> C00
    add_product([( (0,1), 1), ((1,1), -1)], [( (1,0), 1), ((1,1), 1)], (0,0), 1)

    # Now fill in U, V, W from terms
    for (u_row, u_coeff, v_row, v_coeff, w_row, w_coeff) in terms:
        U[u_row][col] = u_coeff
        V[v_row][col] = v_coeff
        W[w_row][col] = w_coeff
        col += 1

    # Part 1b: A1*B12 (4x4 * 4x1) via schoolbook, rank 16
    # A1[i,j] for i=0..3, j=0..3; B12[j,k] for j=0..3, k=4 (only one column)
    # C1[i,4] = sum_j A1[i,j] * B12[j,4]
    # U row for A1[i,j] = i*5+j
    # V row for B12[j,4] = j*5+4
    # W row for C1[i,4] = 4*4+i = 16+i
    for i in range(4):
        for j in range(4):
            U[i*5+j][col] = 1
            V[j*5+4][col] = 1
            W[4*4+i][col] = 1
            col += 1

    # Part 2: A2*B2 (4x1 * 1x5) via schoolbook, rank 20
    # A2[i,4] for i=0..3; B2[4,k] for k=0..4
    # C2[i,k] = A2[i,4] * B2[4,k]
    # U row for A2[i,4] = i*5+4
    # V row for B2[4,k] = 4*5+k = 20+k
    # W row for C2[i,k] = k*4+i
    for i in range(4):
        for k in range(5):
            U[i*5+4][col] = 1
            V[4*5+k][col] = 1
            W[k*4+i][col] = 1
            col += 1

    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"

    return {"U": U, "V": V, "W": W}
