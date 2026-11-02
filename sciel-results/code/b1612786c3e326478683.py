"""Matrix multiplication 4x4 x 4x7 using a 2x2 Strassen-like block decomposition."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    # We decompose the 4x4 x 4x7 multiplication into four 2x2 x 2x7 sub-problems.
    # Each 2x2 x 2x7 sub-problem uses Strassen on the first two columns of B (rank 7)
    # and schoolbook on the remaining 5 columns (rank 4*5=20). Total rank per sub: 27.
    # Total rank: 4 * 27 = 108.
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    
    # Shapes: U is (n*m, rank) = (16, 108), V is (m*p, rank) = (28, 108), W is (p*n, rank) = (28, 108)
    # Convention: W[k*n + i, r] where k is row of B (0..m-1 mapped to 0..3), i is row of A (0..n-1 mapped to 0..3)
    # Actually W[k*n+i, r] with k in 0..m-1 (rows of B) and i in 0..n-1 (rows of A).
    # Wait, the task says output factor W of shape (p*n, rank) with W[k*n+i,r].
    # So k ranges over p (columns of A / rows of C), i ranges over n (columns of C? No.)
    # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # Shape is (p*n, rank). p=7, n=4, so 28 entries.
    # In standard matrix multiplication C = A * B, where A is n x m, B is m x p, C is n x p.
    # The tensor is over indices (i, k, j) where A[i,j] * B[j,k] = C[i,k].
    # So U has shape (n*m, rank) = (4*4, R) with U[i*m+j, c]
    # V has shape (m*p, rank) = (4*7, R) with V[j*p+k, c]
    # W has shape (p*n, rank) = (7*4, R) with W[k*n+i, c]
    
    R = 108
    U = [[0]*R for _ in range(16)]
    V = [[0]*R for _ in range(28)]
    W = [[0]*R for _ in range(28)]
    
    # We split A into 2x2 blocks: A_block[bi][bj] where bi, bj in {0,1}
    # A is 4x4, so rows 0-1 are block row 0, rows 2-3 are block row 1
    # Cols 0-1 are block col 0, cols 2-3 are block col 1
    # Similarly B is 4x7, rows 0-1 are block row 0, rows 2-3 are block row 1
    # Cols: 7 cols, we'll handle as 2x2 x 2x7 subproblems
    # Actually, for each (bi, bj) pair of A blocks, we need B blocks.
    # A[bi, bj] is 2x2, B[bj, :] is 2x7.
    # So for each (bi, bj), we have a 2x2 x 2x7 problem.
    
    # Block indices: bi in {0,1} (A row block), bj in {0,1} (A col block = B row block)
    # A block (bi, bj): rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    # B block (bj, k_block): rows 2*bj to 2*bj+1, cols... B has 7 cols, so we just use all 7 cols for the subproblem
    
    # For each (bi, bj), the subproblem is:
    #   A_sub[2x2] * B_sub[2x7] = C_sub[2x7]
    #   A_sub[i,j] = A[2*bi+i, 2*bj+j]
    #   B_sub[i,k] = B[2*bj+i, k]
    #   C_sub[i,k] = C[2*bi+i, k]
    
    # Strassen on first two columns of B_sub:
    #   M1 = (a00+a11)*(b00+b11)
    #   M2 = (a10+a11)*b00
    #   M3 = a00*(b01-b11)
    #   M4 = a11*(b10-b00)
    #   M5 = (a00+a01)*b11
    #   M6 = (a10-a00)*(b00+b01)
    #   M7 = (a01-a11)*(b10+b11)
    #   C_sub[0,0] = M1+M4-M5+M7
    #   C_sub[0,1] = M3+M5
    #   C_sub[1,0] = M2+M4
    #   C_sub[1,1] = M1-M2+M3+M6
    
    # For columns k=2..6 of B_sub, schoolbook:
    #   C_sub[0,k] = a00*b0k + a01*b1k
    #   C_sub[1,k] = a10*b0k + a11*b1k
    #   Each column needs 4 terms.
    
    def add_term(c, u_idx, v_idx, w_idx, val_u=1, val_v=1, val_w=1):
        U[u_idx][c] += val_u
        V[v_idx][c] += val_v
        W[w_idx][c] += val_w
    
    col_offset = 0
    for bi in range(2):
        for bj in range(2):
            # A block starts at row 2*bi, col 2*bj
            # B block starts at row 2*bj, all cols 0..6
            # C block starts at row 2*bi, all cols 0..6
            
            # A indices: i in 0,1; j in 0,1
            # A[i,j] corresponds to U[(2*bi+i)*m + (2*bj+j)] = U[(2*bi+i)*4 + 2*bj+j]
            # B[i,k] corresponds to V[(2*bj+i)*p + k] = V[(2*bj+i)*7 + k]
            # C[i,k] corresponds to W[k*n + (2*bi+i)] = W[k*4 + 2*bi+i]
            
            # Strassen terms (rank 7)
            # M1 = (a00+a11)*(b00+b11)
            c = col_offset
            add_term(c, (2*bi+0)*4 + 2*bj+0, (2*bj+0)*7 + 0, 0*4 + 2*bi+0, 1, 1, 1)  # a00*b00 -> C[0,0]
            add_term(c, (2*bi+0)*4 + 2*bj+0, (2*bj+1)*7 + 1, 0*4 + 2*bi+0, 1, 1, 1)  # a00*b11 -> C[0,0]
            add_term(c, (2*bi+1)*4 + 2*bj+1, (2*bj+0)*7 + 0, 0*4 + 2*bi+0, 1, 1, 1)  # a11*b00 -> C[0,0]
            add_term(c, (2*bi+1)*4 + 2*bj+1, (2*bj+1)*7 + 1, 0*4 + 2*bi+0, 1, 1, 1)  # a11*b11 -> C[0,0]
            # But this is rank-1 for each product, and Strassen combines them.
            # Actually, Strassen terms are scalar products, so each term is one column with U, V, W entries.
            # Let me redo this properly.
            
            # For M1 = (a00+a11)*(b00+b11):
            # U column has a00 and a11, V column has b00 and b11, W column has C[0,0]
            # But this isn't a single rank-1 tensor; it's the product of two sums.
            # (a00+a11)*(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11
            # This requires 4 rank-1 terms, not 1.
            # So Strassen doesn't directly give rank-7 for this unless we use the Strassen trick correctly.
            
            # The Strassen algorithm for 2x2 x 2x2 uses 7 scalar multiplications.
            # For 2x2 x 2x7, we can use Strassen on the first 2 columns (rank 7) and schoolbook on the rest.
            # The Strassen terms are:
            # M1 = (a00+a11)*(b00+b11)  -- this is ONE multiplication, represented as one column
            # where U has (a00+a11) and V has (b00+b11).
            # So U entries: a00=1, a11=1; V entries: b00=1, b11=1; W entry: C[0,0]=1 (and other contributions)
            
            # Let me define the 7 Strassen terms properly.
            
            # M1: U: a00=1, a11=1; V: b00=1, b11=1; W: C[0,0]=1, C[1,1]=1
            # Wait, Strassen output:
            # C[0,0] = M1 + M4 - M5 + M7
            # C[0,1] = M3 + M5
            # C[1,0] = M2 + M4
            # C[1,1] = M1 - M2 + M3 + M6
            
            # So each Mk contributes to specific C entries with signs.
            
            # M1 = (a00+a11)*(b00+b11)
            c = col_offset
            U[(2*bi+0)*4 + 2*bj+0][c] = 1  # a00
            U[(2*bi+1)*4 + 2*bj+1][c] = 1  # a11
            V[(2*bj+0)*7 + 0][c] = 1  # b00
            V[(2*bj+1)*7 + 1][c] = 1  # b11
            W[0*4 + 2*bi+0][c] = 1   # C[0,0] += M1
            W[1*4 + 2*bi+1][c] = 1   # C[1,1] += M1
            col_offset += 1
            
            # M2 = (a10+a11)*b00
            c = col_offset
            U[(2*bi+1)*4 + 2*bj+0][c] = 1  # a10
            U[(2*bi+1)*4 + 2*bj+1][c] = 1  # a11
            V[(2*bj+0)*7 + 0][c] = 1  # b00
            W[0*4 + 2*bi+0][c] = -1  # C[0,0] -= M2? No, C[1,0] += M2, C[1,1] -= M2
            # C[1,0] = M2 + M4, C[1,1] = M1 - M2 + M3 + M6
            W[0*4 + 2*bi+0][c] = 0  # no C[0,0] from M2
            W[1*4 + 2*bi+1][c] = -1  # C[1,1] -= M2
            W[0*4 + 2*bi+1][c] = 1   # C[1,0] += M2
            col_offset += 1
            
            # M3 = a00*(b01-b11)
            c = col_offset
            U[(2*bi+0)*4 + 2*bj+0][c] = 1  # a00
            V[(2*bj+0)*7 + 1][c] = 1  # b01
            V[(2*bj+1)*7 + 1][c] = -1  # -b11
            W[0*4 + 2*bi+0][c] = 1   # C[0,1] += M3? No, C[0,1] = M3+M5
            # C[0,1] is W[1*4 + 2*bi+0] = W[4 + 2*bi]
            W[0*4 + 2*bi+0][c] = 0
            W[1*4 + 2*bi+0][c] = 1   # C[0,1] += M3
            W[1*4 + 2*bi+1][c] = 1   # C[1,1] += M3
            col_offset += 1
            
            # M4 = a11*(b10-b00)
            c = col_offset
            U[(2*bi+1)*4 + 2*bj+1][c] = 1  # a11
            V[(2*bj+1)*7 + 0][c] = 1  # b10
            V[(2*bj+0)*7 + 0][c] = -1  # -b00
            W[0*4 + 2*bi+0][c] = 1   # C[0,0] += M4
            W[1*4 + 2*bi+1][c] = 1   # C[1,0] += M4
            col_offset += 1
            
            # M5 = (a00+a01)*b11
            c = col_offset
            U[(2*bi+0)*4 + 2*bj+0][c] = 1  # a00
            U[(2*bi+0)*4 + 2*bj+1][c] = 1  # a01
            V[(2*bj+1)*7 + 1][c] = 1  # b11
            W[0*4 + 2*bi+0][c] = -1  # C[0,0] -= M5
            W[1*4 + 2*bi+0][c] = 1   # C[0,1] += M5
            col_offset += 1
            
            # M6 = (a10-a00)*(b00+b01)
            c = col_offset
            U[(2*bi+1)*4 + 2*bj+0][c] = 1  # a10
            U[(2*bi+0)*4 + 2*bj+0][c] = -1  # -a00
            V[(2*bj+0)*7 + 0][c] = 1  # b00
            V[(2*bj+0)*7 + 1][c] = 1  # b01
            W[1*4 + 2*bi+1][c] = 1   # C[1,1] += M6
            col_offset += 1
            
            # M7 = (a01-a11)*(b10+b11)
            c = col_offset
            U[(2*bi+0)*4 + 2*bj+1][c] = 1  # a01
            U[(2*bi+1)*4 + 2*bj+1][c] = -1  # -a11
            V[(2*bj+1)*7 + 0][c] = 1  # b10
            V[(2*bj+1)*7 + 1][c] = 1  # b11
            W[0*4 + 2*bi+0][c] = 1   # C[0,0] += M7
            col_offset += 1
            
            # Now schoolbook for columns k=2..6
            for k in range(2, 7):
                # C[0,k] = a00*b0k + a01*b1k
                # C[1,k] = a10*b0k + a11*b1k
                # Term 1: a00*b0k
                c = col_offset
                U[(2*bi+0)*4 + 2*bj+0][c] = 1
                V[(2*bj+0)*7 + k][c] = 1
                W[k*4 + 2*bi+0][c] = 1
                col_offset += 1
                # Term 2: a01*b1k
                c = col_offset
                U[(2*bi+0)*4 + 2*bj+1][c] = 1
                V[(2*bj+1)*7 + k][c] = 1
                W[k*4 + 2*bi+0][c] = 1
                col_offset += 1
                # Term 3: a10*b0k
                c = col_offset
                U[(2*bi+1)*4 + 2*bj+0][c] = 1
                V[(2*bj+0)*7 + k][c] = 1
                W[k*4 + 2*bi+1][c] = 1
                col_offset += 1
                # Term 4: a11*b1k
                c = col_offset
                U[(2*bi+1)*4 + 2*bj+1][c] = 1
                V[(2*bj+1)*7 + k][c] = 1
                W[k*4 + 2*bi+1][c] = 1
                col_offset += 1
    
    # col_offset should be 108
    assert col_offset == R, f"Expected {R}, got {col_offset}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

