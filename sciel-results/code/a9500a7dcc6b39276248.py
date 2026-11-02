"""Program for matrix multiplication 2x4 * 4x5 with rank-minimal tensor decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Strategy: Exploit n=2 to decompose the tensor.
    # C = A * B where A is 2x4, B is 4x5, C is 2x5.
    # We split B into two 4x2 blocks and one 4x1 block: B = [B1, B2, B3]
    # where B1 is cols 0-1, B2 is cols 2-3, B3 is col 4.
    # C = A*B1 + A*B2 + A*B3.
    # Each A*Bj is 2x2 or 2x1.
    
    # For 2x4 * 4x2: We can split A into two 2x2 blocks: A = [A1, A2]
    # A*B = A1*B_top + A2*B_bot where B_top is 2x2, B_bot is 2x2.
    # Each 2x2 * 2x2 can be done with Strassen (rank 7).
    # So 2x4 * 4x2 costs 14.
    
    # For 2x4 * 4x1: We can split A into two 2x2 blocks: A = [A1, A2]
    # A*B = A1*B_top + A2*B_bot where B_top is 2x1, B_bot is 2x1.
    # Each 2x2 * 2x1 costs 4 (schoolbook).
    # So 2x4 * 4x1 costs 8.
    
    # Total: 14 + 14 + 8 = 36.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    col = 0
    
    def add_strassen_2x2(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """Add Strassen decomposition for 2x2 * 2x2 -> 2x2.
        A is at rows a_row_offset..a_row_offset+1, cols a_col_offset..a_col_offset+1 in A (2x4).
        B is at rows b_row_offset..b_row_offset+1, cols b_col_offset..b_col_offset+1 in B (4x5).
        C is at rows c_row_offset..c_row_offset+1, cols c_col_offset..c_col_offset+1 in C (2x5).
        """
        nonlocal col
        
        # U indices: A[i][j] -> U[i*m + j]
        u00 = a_row_offset * m + a_col_offset
        u01 = a_row_offset * m + a_col_offset + 1
        u10 = (a_row_offset + 1) * m + a_col_offset
        u11 = (a_row_offset + 1) * m + a_col_offset + 1
        
        # V indices: B[i][j] -> V[i*p + j]
        v00 = b_row_offset * p + b_col_offset
        v01 = b_row_offset * p + b_col_offset + 1
        v10 = (b_row_offset + 1) * p + b_col_offset
        v11 = (b_row_offset + 1) * p + b_col_offset + 1
        
        # W indices: C[i][j] -> W[j*n + i] (trace convention)
        w00 = c_col_offset * n + c_row_offset
        w01 = (c_col_offset + 1) * n + c_row_offset
        w10 = c_col_offset * n + c_row_offset + 1
        w11 = (c_col_offset + 1) * n + c_row_offset + 1
        
        # Strassen:
        # M1 = (a00+a11)(b00+b11)
        U[u00][col] += 1; U[u11][col] += 1
        V[v00][col] += 1; V[v11][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1
        
        # M2 = (a10+a11)b00
        U[u10][col] += 1; U[u11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1; W[w11][col] -= 1
        col += 1
        
        # M3 = a00(b01-b11)
        U[u00][col] += 1
        V[v01][col] += 1; V[v11][col] -= 1
        W[w01][col] += 1; W[w11][col] += 1
        col += 1
        
        # M4 = a11(b10-b00)
        U[u11][col] += 1
        V[v10][col] += 1; V[v00][col] -= 1
        W[w00][col] += 1; W[w10][col] += 1
        col += 1
        
        # M5 = (a00+a01)b11
        U[u00][col] += 1; U[u01][col] += 1
        V[v11][col] += 1
        W[w00][col] -= 1; W[w01][col] += 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[u10][col] += 1; U[u00][col] -= 1
        V[v00][col] += 1; V[v01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[u01][col] += 1; U[u11][col] -= 1
        V[v10][col] += 1; V[v11][col] += 1
        W[w00][col] += 1
        col += 1
    
    def add_schoolbook_2x2_2x1(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """Add schoolbook decomposition for 2x2 * 2x1 -> 2x1.
        A is at rows a_row_offset..a_row_offset+1, cols a_col_offset..a_col_offset+1 in A (2x4).
        B is at rows b_row_offset..b_row_offset+1, col b_col_offset in B (4x5).
        C is at rows c_row_offset..c_row_offset+1, col c_col_offset in C (2x5).
        """
        nonlocal col
        
        u00 = a_row_offset * m + a_col_offset
        u01 = a_row_offset * m + a_col_offset + 1
        u10 = (a_row_offset + 1) * m + a_col_offset
        u11 = (a_row_offset + 1) * m + a_col_offset + 1
        
        v0 = b_row_offset * p + b_col_offset
        v1 = (b_row_offset + 1) * p + b_col_offset
        
        w0 = c_col_offset * n + c_row_offset
        w1 = c_col_offset * n + c_row_offset + 1
        
        # C0 = a00*b0 + a01*b1
        U[u00][col] = 1; V[v0][col] = 1; W[w0][col] = 1
        col += 1
        U[u01][col] = 1; V[v1][col] = 1; W[w0][col] = 1
        col += 1
        
        # C1 = a10*b0 + a11*b1
        U[u10][col] = 1; V[v0][col] = 1; W[w1][col] = 1
        col += 1
        U[u11][col] = 1; V[v1][col] = 1; W[w1][col] = 1
        col += 1
    
    # C[:, 0:2] = A * B[:, 0:2] (2x4 * 4x2)
    # Split A = [A[:, 0:2], A[:, 2:4]]
    # A[:, 0:2] * B[0:2, 0:2] (2x2 * 2x2)
    add_strassen_2x2(0, 0, 0, 0, 0, 0)
    # A[:, 2:4] * B[2:4, 0:2] (2x2 * 2x2)
    add_strassen_2x2(0, 2, 2, 0, 0, 0)
    
    # C[:, 2:4] = A * B[:, 2:4] (2x4 * 4x2)
    # Split A = [A[:, 0:2], A[:, 2:4]]
    # A[:, 0:2] * B[0:2, 2:4] (2x2 * 2x2)
    add_strassen_2x2(0, 0, 0, 2, 0, 2)
    # A[:, 2:4] * B[2:4, 2:4] (2x2 * 2x2)
    add_strassen_2x2(0, 2, 2, 2, 0, 2)
    
    # C[:, 4] = A * B[:, 4] (2x4 * 4x1)
    # Split A = [A[:, 0:2], A[:, 2:4]]
    # A[:, 0:2] * B[0:2, 4] (2x2 * 2x1)
    add_schoolbook_2x2_2x1(0, 0, 0, 4, 0, 4)
    # A[:, 2:4] * B[2:4, 4] (2x2 * 2x1)
    add_schoolbook_2x2_2x1(0, 2, 2, 4, 0, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
