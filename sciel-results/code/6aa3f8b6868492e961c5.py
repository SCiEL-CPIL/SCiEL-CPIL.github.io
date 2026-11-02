"""Matrix multiplication 2x4x6 with reduced rank via Strassen-like decomposition."""
import time

DIMENSIONS = [2, 4, 6]

def _build_strassen_2x2x2():
    """Return U,V,W (each 4x7) for 2x2x2 matrix multiplication using Strassen."""
    # Strassen 7 multiplications for 2x2x2
    # C = A*B where A,B,C are 2x2
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    #
    # Multiplications:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # C entries:
    # c00 = M1 - M4 + M6 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M3 + M2 + M5
    #
    # In tensor decomposition form: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # U has 4 rows (indices 0..3 for A[0,0], A[0,1], A[1,0], A[1,1])
    # V has 4 rows (indices 0..3 for B[0,0], B[0,1], B[1,0], B[1,1])
    # W has 4 rows (indices 0..3 for C[0,0], C[0,1], C[1,0], C[1,1])
    # rank = 7

    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]

    # M1 = (a00+a11)*(b00+b11)
    U[0][0] = 1  # a00
    U[2][0] = 1  # a11
    V[0][0] = 1  # b00
    V[3][0] = 1  # b11

    # M2 = (a10+a11)*b00
    U[2][1] = 1  # a10
    U[3][1] = 1  # a11
    V[0][1] = 1  # b00

    # M3 = a00*(b01-b11)
    U[0][2] = 1  # a00
    V[1][2] = 1  # b01
    V[3][2] = -1 # b11

    # M4 = a11*(b10-b00)
    U[3][3] = 1  # a11
    V[2][3] = 1  # b10
    V[0][3] = -1 # b00

    # M5 = (a00+a01)*b11
    U[0][4] = 1  # a00
    U[1][4] = 1  # a01
    V[3][4] = 1  # b11

    # M6 = (a10-a00)*(b00+b01)
    U[2][5] = 1  # a10
    U[0][5] = -1 # a00
    V[0][5] = 1  # b00
    V[1][5] = 1  # b01

    # M7 = (a01-a11)*(b10+b11)
    U[1][6] = 1  # a01
    U[3][6] = -1 # a11
    V[2][6] = 1  # b10
    V[3][6] = 1  # b11

    # W for C entries:
    # c00 = M1 - M4 + M6 + M7
    W[0][0] = 1   # M1 -> c00
    W[0][3] = -1  # M4 -> c00 (negative)
    W[0][5] = 1   # M6 -> c00
    W[0][6] = 1   # M7 -> c00

    # c01 = M3 + M5
    W[1][2] = 1   # M3 -> c01
    W[1][4] = 1   # M5 -> c01

    # c10 = M2 + M4
    W[2][1] = 1   # M2 -> c10
    W[2][3] = 1   # M4 -> c10

    # c11 = M1 - M3 + M2 + M5
    W[3][0] = 1   # M1 -> c11
    W[3][2] = -1  # M3 -> c11 (negative)
    W[3][1] = 1   # M2 -> c11
    W[3][4] = 1   # M5 -> c11

    return U, V, W

def construct():
    """
    Decompose 2x4x6 matrix multiplication.
    
    Strategy: Split B into B1 (4x4) and B2 (4x2).
    C = A*B1 + A*B2
    
    For A*B1 (2x4 * 4x4 = 2x4):
      Split A into A1 (2x2) and A2 (2x2), B1 into B1_1 (2x2) and B1_2 (2x2).
      C_left = A1*B1_1 + A2*B1_2  (each term is 2x2x2)
      Each 2x2x2 uses Strassen rank 7, total 14.
      
    For A*B2 (2x4 * 4x2 = 2x2):
      Split A into A1 (2x2) and A2 (2x2), B2 into B2_1 (2x2) and B2_2 (2x2).
      C_right = A1*B2_1 + A2*B2_2  (each term is 2x2x2)
      Each 2x2x2 uses Strassen rank 7, total 14.
      
    Total rank = 14 + 14 = 28.
    
    Wait, let me reconsider the indices.
    
    A is 2x4: rows i=0,1; cols j=0,1,2,3
    B is 4x6: rows j=0,1,2,3; cols k=0,1,2,3,4,5
    C is 2x6: rows i=0,1; cols k=0,1,2,3,4,5
    
    C[i,k] = sum_j A[i,j]*B[j,k]
    
    Split B into B_left (4x4, cols k=0..3) and B_right (4x2, cols k=4..5).
    C_left[i,k] = sum_j A[i,j]*B[j,k] for k=0..3
    C_right[i,k] = sum_j A[i,j]*B[j,k] for k=4..5
    
    For C_left (2x4):
      Split A into A_col0_1 (2x2, cols j=0,1) and A_col2_3 (2x2, cols j=2,3).
      Split B_left into B_row0_1 (2x4, rows j=0,1) and B_row2_3 (2x4, rows j=2,3).
      
      C_left = A_col0_1 * B_row0_1 + A_col2_3 * B_row2_3
      
      But A_col0_1 is 2x2, B_row0_1 is 2x4. This is 2x2x4.
      
      For 2x2x4: Split B_row0_1 into two 2x2 blocks.
      A_col0_1 * B_row0_1 = A_col0_1 * [B_a | B_b] = [A_col0_1*B_a | A_col0_1*B_b]
      Each A_col0_1*B is 2x2x2, rank 7 each. Total 14.
      
      Similarly A_col2_3 * B_row2_3 = 14.
      
      So C_left total = 28.
      
    For C_right (2x2):
      A_col0_1 * B_right_row0_1 + A_col2_3 * B_right_row2_3
      Each is 2x2x2, rank 7 each. Total 14.
      
    Grand total = 28 + 14 = 42.
    
    This is the same as before. Let me just implement it carefully.
    """
    deadline = time.time() + 145  # Safety margin
    
    # Get Strassen decomposition for 2x2x2
    S_U, S_V, S_W = _build_strassen_2x2x2()
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 42
    
    U = [[0]*rank for _ in range(n*m)]  # 8x42
    V = [[0]*rank for _ in range(m*p)]  # 24x42
    W = [[0]*rank for _ in range(p*n)]  # 12x42
    
    # We'll build 6 Strassen blocks, each contributing 7 to the rank.
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # Block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    # Block 3: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    # Block 4: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    # Block 5: A[0:2,0:2] * B[0:2,4:6] -> C[0:2,4:6]
    # Block 6: A[0:2,2:4] * B[2:4,4:6] -> C[0:2,4:6]
    
    # For each Strassen block, we need to map the local 2x2x2 indices to global indices.
    # Local A indices: (i_local, j_local) -> global (i, j_base + j_local)
    # Local B indices: (j_local, k_local) -> global (j_base + j_local, k_base + k_local)
    # Local C indices: (i_local, k_local) -> global (i, k_base + k_local)
    
    def apply_strassen_block(base_U_row, base_V_row, base_W_row, j_base, k_base, col_start):
        """
        Apply one Strassen 2x2x2 block.
        base_U_row: base row index in U for A (0 or 2, since A has 2 rows)
        base_V_row: base row index in V for B (j_base)
        base_W_row: base row index in W for C (k_base)
        j_base: base column in A / base row in B
        k_base: base column in C
        col_start: starting column in the rank dimension
        """
        for c in range(7):
            col = col_start + c
            # U: A rows
            for i_local in range(2):
                global_i = 0  # A always has i=0,1 but we fix i_base=0 here
                # Actually, A has 2 rows, so i_local maps to row i_local
                u_row = i_local
                val = S_U[i_local][c]
                if val != 0:
                    U[u_row][col] = val
            
            # V: B rows
            for j_local in range(2):
                global_j = j_base + j_local
                v_row = global_j  # V has m*p = 24 rows, indexed by j*p + k... 
                # Wait, V is indexed by (j, k) as j*p + k? No.
                # V has shape (m*p, rank) = (24, rank).
                # The problem says V[j*p+k, c]. So V row index = j*p + k.
                # But in our Strassen block, we're multiplying A[i,j]*B[j,k].
                # V should have entries for B[j,k].
                # So V row = j*p + k.
                # In this block, j = j_base + j_local, k = k_base + k_local.
                # We need to iterate over both j_local and k_local.
                pass
            
            # Let me redo this. S_V has 4 rows for B[0,0], B[0,1], B[1,0], B[1,1].
            # S_V[j_local*2 + k_local][c]
            for j_local in range(2):
                for k_local in range(2):
                    global_j = j_base + j_local
                    global_k = k_base + k_local
                    v_row = global_j * p + global_k
                    val = S_V[j_local * 2 + k_local][c]
                    if val != 0:
                        V[v_row][col] = val
            
            # W: C rows
            # W has shape (p*n, rank) = (12, rank). W[k*n + i, c].
            # S_W[i_local*2 + k_local][c] for C[i_local, k_local]
            for i_local in range(2):
                for k_local in range(2):
                    global_k = k_base + k_local
                    global_i = i_local  # C row = i
                    w_row = global_k * n + global_i
                    val = S_W[i_local * 2 + k_local][c]
                    if val != 0:
                        W[w_row][col] = val
    
    # Block 1: A cols 0-1, B rows 0-1, C cols 0-1
    apply_strassen_block(0, 0, 0, 0, 0, 0)
    
    # Block 2: A cols 0-1, B rows 0-1, C cols 2-3
    apply_strassen_block(0, 0, 2, 0, 2, 7)
    
    # Block 3: A cols 2-3, B rows 2-3, C cols 0-1
    apply_strassen_block(0, 2, 0, 2, 0, 14)
    
    # Block 4: A cols 2-3, B rows 2-3, C cols 2-3
    apply_strassen_block(0, 2, 2, 2, 2, 21)
    
    # Block 5: A cols 0-1, B rows 0-1, C cols 4-5
    apply_strassen_block(0, 0, 4, 0, 4, 28)
    
    # Block 6: A cols 2-3, B rows 2-3, C cols 4-5
    apply_strassen_block(0, 2, 4, 2, 4, 35)
    
    return {"U": U, "V": V, "W": W}
