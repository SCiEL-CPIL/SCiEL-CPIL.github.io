"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    # Rank-36 decomposition via block splitting:
    # Split 2x4x5 into two 2x2x5 blocks.
    # Each 2x2x5 is decomposed as:
    #   Split B (2x5) into B1 (2x4) and B2 (2x1)
    #   2x2x4 uses Strassen-style rank-7 twice -> rank 14
    #   2x2x1 uses rank 4
    # Total per 2x2x5: 14 + 4 = 18
    # Total: 18 + 18 = 36
    #
    # Convention: C[i,k] = sum_j A[i,j] * B[j,k]
    # U has shape (n*m, rank) = (8, 36)
    # V has shape (m*p, rank) = (20, 36)
    # W has shape (p*n, rank) = (10, 36)
    # W index: W[k*n + i, c]
    n, m, p = 2, 4, 5
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 20 rows
    W = [[0] * rank for _ in range(p * n)]  # 10 rows

    # For each of the two 2x2x5 blocks (block b = 0, 1):
    #   A_block is rows of A for cols [2*b, 2*b+1]
    #   B_block is rows of B for rows [2*b, 2*b+1], all cols
    #
    # Within each 2x2x5 block, we split B into:
    #   B1: cols 0..3 (2x4) -> use Strassen for 2x2x4, rank 7 per 2x2x2 sub-block x2 = 14
    #   B2: col 4 (2x1) -> rank 4
    #
    # Let's index the columns of B in the 2x2x5 block:
    #   B1 has cols 0,1,2,3 (4 cols) -> two 2x2x2 Strassen blocks
    #   B2 has col 4 (1 col) -> rank 4
    #
    # For a 2x2x2 Strassen block with A (2x2), B (2x2), C (2x2):
    #   M1 = (A00+A11)*(B00+B11)
    #   M2 = (A10+A11)*B00
    #   M3 = A00*(B01-B11)
    #   M4 = A11*(B10-B00)
    #   M5 = (A00+A01)*B11
    #   M6 = (A10-A00)*(B00+B10)
    #   M7 = (A01-A11)*(B01+B11)
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    #
    # For a 2x2x1 block with A (2x2), B (2x1), C (2x1):
    #   C0 = A00*B0 + A01*B1
    #   C1 = A10*B0 + A11*B1
    #   Rank 4: just enumerate all 4 products
    #   P1 = A00*B0, P2 = A01*B1, P3 = A10*B0, P4 = A11*B1
    #   C0 = P1 + P2
    #   C1 = P3 + P4

    def add_strassen_2x2x2(U, V, W, col_offset, a_row_offset, b_row_offset, c_col_offset):
        """Add Strassen 2x2x2 block at given offsets.
        a_row_offset: row offset in A (0 or 1 for the 2x2 block)
        b_row_offset: row offset in B
        c_col_offset: col offset in C (which output column this contributes to)
        Actually, let me rethink the indexing.
        """
        pass

    # Let me be more explicit.
    # A is 2x4: A[i,j] for i in {0,1}, j in {0,1,2,3}
    # B is 4x5: B[j,k] for j in {0,1,2,3}, k in {0,1,2,3,4}
    # C is 2x5: C[i,k] for i in {0,1}, k in {0,1,2,3,4}
    #
    # Block 0: A cols 0,1; B rows 0,1
    # Block 1: A cols 2,3; B rows 2,3
    #
    # For block b (b=0,1):
    #   A_sub[i, jj] = A[i, 2*b + jj] for jj in {0,1}
    #   B_sub[jj, k] = B[2*b + jj, k] for jj in {0,1}, k in {0,1,2,3,4}
    #
    # Split B_sub into B1 (cols 0-3) and B2 (col 4)
    # B1: 2x4, split into two 2x2: B1a (cols 0,1) and B1b (cols 2,3)
    # B2: 2x1
    #
    # So for each block b, we have:
    #   Strassen for A_sub * B1a -> contributes to C[:, 2*b_col... ]
    #   Wait, B1a is cols 0,1 of the 5-col B_sub, which are global cols 0,1
    #   B1b is cols 2,3 of the 5-col B_sub, which are global cols 2,3
    #   B2 is col 4 of the 5-col B_sub, which is global col 4
    #
    # So:
    # Block 0:
    #   Strassen(A[A[:,0:2], :], B[0:2, 0:2]) -> C[:, 0:2]
    #   Strassen(A[A[:,0:2], :], B[0:2, 2:4]) -> C[:, 2:4]
    #   Rank4(A[A[:,0:2], :], B[0:2, 4:5]) -> C[:, 4:5]
    # Block 1:
    #   Strassen(A[A[:,2:4], :], B[2:4, 0:2]) -> C[:, 0:2]
    #   Strassen(A[A[:,2:4], :], B[2:4, 2:4]) -> C[:, 2:4]
    #   Rank4(A[A[:,2:4], :], B[2:4, 4:5]) -> C[:, 4:5]
    #
    # Total: 7+7+4+7+7+4 = 36. Correct!

    col = 0
    for b in range(2):  # block index
        a_col_base = 2 * b  # column offset in A
        b_row_base = 2 * b  # row offset in B
        for b1_split in range(2):  # B1a (cols 0,1) or B1b (cols 2,3)
            b_col_base = 2 * b1_split  # column offset in B for this 2x2
            # Add Strassen 2x2x2:
            # A indices: rows i in {0,1}, cols a_col_base + jj for jj in {0,1}
            # B indices: rows b_row_base + jj for jj in {0,1}, cols b_col_base + kk for kk in {0,1}
            # C indices: rows i in {0,1}, cols b_col_base + kk for kk in {0,1}
            #
            # Strassen:
            # M1 = (A00+A11)*(B00+B11)
            # M2 = (A10+A11)*B00
            # M3 = A00*(B01-B11)
            # M4 = A11*(B10-B00)
            # M5 = (A00+A01)*B11
            # M6 = (A10-A00)*(B00+B10)
            # M7 = (A01-A11)*(B01+B11)
            #
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            for s in range(7):
                c = col + s
                # U entries (A side)
                if s == 0:  # M1: A00 + A11
                    U[a_col_base + 0][c] += 1  # A[0, a_col_base+0]
                    U[a_col_base + 1][c] += 1  # A[1, a_col_base+1]
                elif s == 1:  # M2: A10 + A11
                    U[a_col_base + 0][c] += 1  # A[1, a_col_base+0]
                    U[a_col_base + 1][c] += 1  # A[1, a_col_base+1]
                elif s == 2:  # M3: A00
                    U[a_col_base + 0][c] += 1  # A[0, a_col_base+0]
                elif s == 3:  # M4: A11
                    U[a_col_base + 1][c] += 1  # A[1, a_col_base+1]
                elif s == 4:  # M5: A00 + A01
                    U[a_col_base + 0][c] += 1  # A[0, a_col_base+0]
                    U[a_col_base + 1][c] += 1  # A[0, a_col_base+1]
                elif s == 5:  # M6: A10 - A00
                    U[a_col_base + 0][c] += -1  # A[1, a_col_base+0]
                    U[a_col_base + 1][c] += 1   # A[0, a_col_base+0]
                elif s == 6:  # M7: A01 - A11
                    U[a_col_base + 1][c] += -1  # A[0, a_col_base+1]
                    U[a_col_base + 1][c] += 1   # A[1, a_col_base+1]
                
                # V entries (B side)
                if s == 0:  # M1: B00 + B11
                    V[b_row_base + 0][c] += 1  # B[b_row_base+0, b_col_base+0]
                    V[b_row_base + 1][c] += 1  # B[b_row_base+1, b_col_base+1]
                elif s == 1:  # M2: B00
                    V[b_row_base + 0][c] += 1  # B[b_row_base+0, b_col_base+0]
                elif s == 2:  # M3: B01 - B11
                    V[b_row_base + 0][c] += 1   # B[b_row_base+0, b_col_base+1]
                    V[b_row_base + 1][c] += -1  # B[b_row_base+1, b_col_base+1]
                elif s == 3:  # M4: B10 - B00
                    V[b_row_base + 1][c] += 1   # B[b_row_base+1, b_col_base+0]
                    V[b_row_base + 0][c] += -1  # B[b_row_base+0, b_col_base+0]
                elif s == 4:  # M5: B11
                    V[b_row_base + 1][c] += 1  # B[b_row_base+1, b_col_base+1]
                elif s == 5:  # M6: B00 + B10
                    V[b_row_base + 0][c] += 1   # B[b_row_base+0, b_col_base+0]
                    V[b_row_base + 1][c] += 1   # B[b_row_base+1, b_col_base+0]
                elif s == 6:  # M7: B01 + B11
                    V[b_row_base + 0][c] += 1   # B[b_row_base+0, b_col_base+1]
                    V[b_row_base + 1][c] += 1   # B[b_row_base+1, b_col_base+1]
                
                # W entries (C side): W[k*n + i, c]
                if s == 0:  # M1: C00 + C11
                    W[0 * n + 0][c] += 1  # C[0, b_col_base+0]
                    W[1 * n + 1][c] += 1  # C[1, b_col_base+1]
                elif s == 1:  # M2: C10
                    W[0 * n + 1][c] += 1  # C[1, b_col_base+0]
                elif s == 2:  # M3: C01
                    W[1 * n + 0][c] += 1  # C[0, b_col_base+1]
                elif s == 3:  # M4: C00 + C10
                    W[0 * n + 0][c] += 1  # C[0, b_col_base+0]
                    W[0 * n + 1][c] += 1  # C[1, b_col_base+0]
                elif s == 4:  # M5: -C00 + C01
                    W[0 * n + 0][c] += -1  # C[0, b_col_base+0]
                    W[1 * n + 0][c] += 1   # C[0, b_col_base+1]
                elif s == 5:  # M6: C11
                    W[1 * n + 1][c] += 1  # C[1, b_col_base+1]
                elif s == 6:  # M7: C00 - C11
                    W[0 * n + 0][c] += 1   # C[0, b_col_base+0]
                    W[1 * n + 1][c] += -1  # C[1, b_col_base+1]
                col += 1
        
        # Now rank-4 for B2 (col 4)
        for r4 in range(4):
            c = col + r4
            # P1 = A00*B0, P2 = A01*B1, P3 = A10*B0, P4 = A11*B1
            # C0 = P1 + P2
            # C1 = P3 + P4
            if r4 == 0:  # P1: A00 * B0
                U[a_col_base + 0][c] += 1
                V[b_row_base + 0][c] += 1
                W[4 * n + 0][c] += 1  # C[0, 4]
            elif r4 == 1:  # P2: A01 * B1
                U[a_col_base + 1][c] += 1
                V[b_row_base + 1][c] += 1
                W[4 * n + 0][c] += 1  # C[0, 4]
            elif r4 == 2:  # P3: A10 * B0
                U[a_col_base + 0][c] += 1
                V[b_row_base + 0][c] += 1
                W[4 * n + 1][c] += 1  # C[1, 4]
            elif r4 == 3:  # P4: A11 * B1
                U[a_col_base + 1][c] += 1
                V[b_row_base + 1][c] += 1
                W[4 * n + 1][c] += 1  # C[1, 4]
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
