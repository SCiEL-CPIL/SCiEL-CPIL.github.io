"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We'll try to use Strassen-style decomposition on 2x2 blocks
    # For 2x4 * 4x4:
    # A is 2x4, B is 4x4
    # Split A into columns: A = [A1 A2] where A1 is 2x2, A2 is 2x2
    # Split B into columns: B = [B1 B2] where B1 is 4x2, B2 is 4x2
    # Result C = A*B is 2x4
    # C = [A1*B1 + A2*B2_top, A1*B2_top + A2*B2_bottom]
    
    # Actually, let me think more carefully.
    # A is 2x4, B is 4x4, C is 2x4
    # Split A by columns: A = [A_left A_right] where A_left is 2x2, A_right is 2x2
    # Split B by rows: B = [B_top; B_bottom] where B_top is 2x4, B_bottom is 2x4
    # Then C = A_left * B_top + A_right * B_bottom
    # Each product is 2x2 * 2x4 = 2x4
    # So we need two 2x2-by-2x4 multiplications
    
    # For 2x2 * 2x4:
    # A' is 2x2, B' is 2x4
    # Split B' into columns: B' = [B1 B2] where B1 is 2x2, B2 is 2x2
    # C' = A' * B' = [A'*B1, A'*B2]
    # Each is 2x2 * 2x2
    
    # So 2x4 * 4x4 = 2 * (2x2 * 2x4) = 2 * 2 * (2x2 * 2x2) = 4 * (2x2*2x2)
    # If we use Strassen for each 2x2*2x2 (rank 7), total rank = 4*7 = 28
    
    # Let's construct this explicitly.
    # The 2x2*2x2 Strassen decomposition uses 7 multiplications.
    # Standard Strassen for C = A*B where A,B,C are 2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Now for our problem:
    # A is 2x4, B is 4x4
    # A = [A11 A12; A21 A22] where each is... wait, A is 2x4
    # Let me re-index. A has rows 0,1 and cols 0,1,2,3
    # B has rows 0,1,2,3 and cols 0,1,2,3
    
    # Split: A_left = A[:, 0:2] (2x2), A_right = A[:, 2:4] (2x2)
    # B_top = B[0:2, :] (2x4), B_bottom = B[2:4, :] (2x4)
    # C = A_left @ B_top + A_right @ B_bottom (both 2x4)
    
    # For A_left @ B_top:
    # A_left is 2x2, B_top is 2x4
    # B_top = [B_top1, B_top2] where B_top1 = B[0:2, 0:2] (2x2), B_top2 = B[0:2, 2:4] (2x2)
    # Result = [A_left @ B_top1, A_left @ B_top2]
    
    # Similarly for A_right @ B_bottom:
    # B_bottom = [B_bottom1, B_bottom2] where B_bottom1 = B[2:4, 0:2] (2x2), B_bottom2 = B[2:4, 2:4] (2x2)
    # Result = [A_right @ B_bottom1, A_right @ B_bottom2]
    
    # So C[:, 0:2] = A_left @ B_top1 + A_right @ B_bottom1
    # C[:, 2:4] = A_left @ B_top2 + A_right @ B_bottom2
    
    # Each of these is a sum of two 2x2*2x2 products.
    # We can use Strassen for each 2x2*2x2 product.
    # Total: 4 Strassen decompositions, each rank 7, total rank 28.
    
    # Let's build the factor matrices.
    # U has shape (n*m, rank) = (8, 28)
    # V has shape (m*p, rank) = (16, 28)
    # W has shape (p*n, rank) = (8, 28)
    # W index: W[k*n+i, r] corresponds to output C[i,k]
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # We have 4 Strassen blocks:
    # Block 0: A_left @ B_top1 -> contributes to C[:, 0:2]
    # Block 1: A_right @ B_bottom1 -> contributes to C[:, 0:2]
    # Block 2: A_left @ B_top2 -> contributes to C[:, 2:4]
    # Block 3: A_right @ B_bottom2 -> contributes to C[:, 2:4]
    
    # For each Strassen block, we need to map:
    # - A indices (which part of A)
    # - B indices (which part of B)
    # - C indices (which part of C)
    
    # A_left: rows 0,1 of A, cols 0,1 -> A[i, j] for i in 0,1, j in 0,1
    #   In U: index i*m + j = i*4 + j
    # A_right: rows 0,1 of A, cols 2,3 -> A[i, j] for i in 0,1, j in 2,3
    #   In U: index i*4 + j
    # B_top1: B[0:2, 0:2] -> B[i, j] for i in 0,1, j in 0,1
    #   In V: index i*p + j = i*4 + j
    # B_bottom1: B[2:4, 0:2] -> B[i, j] for i in 2,3, j in 0,1
    #   In V: index i*4 + j
    # B_top2: B[0:2, 2:4] -> B[i, j] for i in 0,1, j in 2,3
    #   In V: index i*4 + j
    # B_bottom2: B[2:4, 2:4] -> B[i, j] for i in 2,3, j in 2,3
    #   In V: index i*4 + j
    
    # C[:, 0:2]: C[i, k] for i in 0,1, k in 0,1
    #   In W: index k*n + i = k*2 + i
    # C[:, 2:4]: C[i, k] for i in 0,1, k in 2,3
    #   In W: index k*n + i = k*2 + i
    
    # Strassen 2x2*2x2:
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For a given Strassen block, we specify:
    # - a_row_offset, a_col_offset: where in A the 2x2 block starts
    # - b_row_offset, b_col_offset: where in B the 2x2 block starts
    # - c_row_offset, c_col_offset: where in C the 2x2 result goes
    # - sign: +1 or -1 for the contribution to C
    
    def add_strassen_block(base_col, a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off, sign=1):
        # A local indices: (0,0), (0,1), (1,0), (1,1)
        # A global: (a_row_off + r, a_col_off + c)
        # B local: (0,0), (0,1), (1,0), (1,1)
        # B global: (b_row_off + r, b_col_off + c)
        # C local: (0,0), (0,1), (1,0), (1,1)
        # C global: (c_row_off + r, c_col_off + c)
        
        # M1 = (a11+a22)*(b11+b22)
        col = base_col
        # U: a11 and a22
        U[a_row_off * m + a_col_off][col] = sign  # a11
        U[(a_row_off+1) * m + (a_col_off+1)][col] = sign  # a22
        # V: b11 and b22
        V[b_row_off * p + b_col_off][col] = sign  # b11
        V[(b_row_off+1) * p + (b_col_off+1)][col] = sign  # b22
        # W: C11 and C22
        W[(c_col_off) * n + (c_row_off)][col] = sign  # C11
        W[(c_col_off+1) * n + (c_row_off+1)][col] = sign  # C22
        col += 1
        
        # M2 = (a21+a22)*b11
        # U: a21 and a22
        U[(a_row_off+1) * m + a_col_off][col] = sign  # a21
        U[(a_row_off+1) * m + (a_col_off+1)][col] = sign  # a22
        # V: b11
        V[b_row_off * p + b_col_off][col] = sign  # b11
        # W: C21
        W[(c_col_off) * n + (c_row_off+1)][col] = sign  # C21
        col += 1
        
        # M3 = a11*(b12-b22)
        # U: a11
        U[a_row_off * m + a_col_off][col] = sign  # a11
        # V: b12 and -b22
        V[b_row_off * p + (b_col_off+1)][col] = sign  # b12
        V[(b_row_off+1) * p + (b_col_off+1)][col] = -sign  # -b22
        # W: C12 and C22
        W[(c_col_off+1) * n + (c_row_off)][col] = sign  # C12
        W[(c_col_off+1) * n + (c_row_off+1)][col] = sign  # C22
        col += 1
        
        # M4 = a22*(b21-b11)
        # U: a22
        U[(a_row_off+1) * m + (a_col_off+1)][col] = sign  # a22
        # V: b21 and -b11
        V[(b_row_off+1) * p + b_col_off][col] = sign  # b21
        V[b_row_off * p + b_col_off][col] = -sign  # -b11
        # W: C11 and C21
        W[(c_col_off) * n + (c_row_off)][col] = sign  # C11
        W[(c_col_off) * n + (c_row_off+1)][col] = sign  # C21
        col += 1
        
        # M5 = (a11+a12)*b22
        # U: a11 and a12
        U[a_row_off * m + a_col_off][col] = sign  # a11
        U[a_row_off * m + (a_col_off+1)][col] = sign  # a12
        # V: b22
        V[(b_row_off+1) * p + (b_col_off+1)][col] = sign  # b22
        # W: C11 (with -) and C12
        W[(c_col_off) * n + (c_row_off)][col] = -sign  # C11 (negative)
        W[(c_col_off+1) * n + (c_row_off)][col] = sign  # C12
        col += 1
        
        # M6 = (a21-a11)*(b11+b12)
        # U: a21 and -a11
        U[(a_row_off+1) * m + a_col_off][col] = sign  # a21
        U[a_row_off * m + a_col_off][col] = -sign  # -a11
        # V: b11 and b12
        V[b_row_off * p + b_col_off][col] = sign  # b11
        V[b_row_off * p + (b_col_off+1)][col] = sign  # b12
        # W: C22
        W[(c_col_off+1) * n + (c_row_off+1)][col] = sign  # C22
        col += 1
        
        # M7 = (a12-a22)*(b21+b22)
        # U: a12 and -a22
        U[a_row_off * m + (a_col_off+1)][col] = sign  # a12
        U[(a_row_off+1) * m + (a_col_off+1)][col] = -sign  # -a22
        # V: b21 and b22
        V[(b_row_off+1) * p + b_col_off][col] = sign  # b21
        V[(b_row_off+1) * p + (b_col_off+1)][col] = sign  # b22
        # W: C11
        W[(c_col_off) * n + (c_row_off)][col] = sign  # C11
        col += 1
        
        return col
    
    # Block 0: A_left @ B_top1 -> C[:, 0:2]
    # A_left: a_row_off=0, a_col_off=0
    # B_top1: b_row_off=0, b_col_off=0
    # C[:, 0:2]: c_row_off=0, c_col_off=0
    col = add_strassen_block(0, 0, 0, 0, 0, 0, 0, +1)
    
    # Block 1: A_right @ B_bottom1 -> C[:, 0:2]
    # A_right: a_row_off=0, a_col_off=2
    # B_bottom1: b_row_off=2, b_col_off=0
    # C[:, 0:2]: c_row_off=0, c_col_off=0
    col = add_strassen_block(col, 0, 2, 2, 0, 0, 0, +1)
    
    # Block 2: A_left @ B_top2 -> C[:, 2:4]
    # A_left: a_row_off=0, a_col_off=0
    # B_top2: b_row_off=0, b_col_off=2
    # C[:, 2:4]: c_row_off=0, c_col_off=2
    col = add_strassen_block(col, 0, 0, 0, 2, 0, 2, +1)
    
    # Block 3: A_right @ B_bottom2 -> C[:, 2:4]
    # A_right: a_row_off=0, a_col_off=2
    # B_bottom2: b_row_off=2, b_col_off=2
    # C[:, 2:4]: c_row_off=0, c_col_off=2
    col = add_strassen_block(col, 0, 2, 2, 2, 0, 2, +1)
    
    assert col == 28, f"Expected 28, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
