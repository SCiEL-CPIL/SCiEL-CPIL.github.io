"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-28 exact tensor decomposition for 2x4 by 4x4 matrix multiplication.
    
    Uses Strassen's 2x2x2 algorithm applied to four 2x2 blocks.
    The 2x4 matrix A is split into A_left (cols 0-1) and A_right (cols 2-3).
    The 4x4 matrix B is split into B_tl, B_tr, B_bl, B_br.
    C = A*B.
    C_left = A_left * B_tl + A_right * B_bl
    C_right = A_left * B_tr + A_right * B_br
    
    Each 2x2x2 multiplication uses Strassen's 7 multiplications, giving 4*7 = 28 total.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Index mapping:
    # A is 2x4, flattened A[k*m+i]. A[0]=A00, A[1]=A01, A[2]=A02, A[3]=A03, A[4]=A10, A[5]=A11, A[6]=A12, A[7]=A13
    # B is 4x4, flattened B[k*m+i]. B[0..3]=row0, B[4..7]=row1, B[8..11]=row2, B[12..15]=row3
    # C is 2x4, output W[k*n+i] where k is col_C (0..3), i is row_C (0..1).
    # W index = col_C * 2 + row_C.
    # C[0,0]->0, C[1,0]->1, C[0,1]->2, C[1,1]->3, C[0,2]->4, C[1,2]->5, C[0,3]->6, C[1,3]->7.

    def add_strassen_block(u_idx, v_idx, w_idx, col_offset):
        """
        Add a Strassen 2x2x2 block.
        u_idx: [u00, u01, u10, u11]
        v_idx: [v00, v01, v10, v11]
        w_idx: [w00, w01, w10, w11]
        """
        # M1 = (A00 + A11)(B00 + B11)
        U[u_idx[0]][col_offset + 0] += 1
        U[u_idx[3]][col_offset + 0] += 1
        V[v_idx[0]][col_offset + 0] += 1
        V[v_idx[3]][col_offset + 0] += 1
        W[w_idx[0]][col_offset + 0] += 1
        W[w_idx[3]][col_offset + 0] += 1
        
        # M2 = (A10 + A11)B00
        U[u_idx[2]][col_offset + 1] += 1
        U[u_idx[3]][col_offset + 1] += 1
        V[v_idx[0]][col_offset + 1] += 1
        W[w_idx[2]][col_offset + 1] += 1
        W[w_idx[3]][col_offset + 1] -= 1
        
        # M3 = A00(B01 - B11)
        U[u_idx[0]][col_offset + 2] += 1
        V[v_idx[1]][col_offset + 2] += 1
        V[v_idx[3]][col_offset + 2] -= 1
        W[w_idx[1]][col_offset + 2] += 1
        W[w_idx[3]][col_offset + 2] += 1
        
        # M4 = A11(B10 - B00)
        U[u_idx[3]][col_offset + 3] += 1
        V[v_idx[2]][col_offset + 3] += 1
        V[v_idx[0]][col_offset + 3] -= 1
        W[w_idx[0]][col_offset + 3] += 1
        W[w_idx[2]][col_offset + 3] += 1
        
        # M5 = (A00 + A01)B11
        U[u_idx[0]][col_offset + 4] += 1
        U[u_idx[1]][col_offset + 4] += 1
        V[v_idx[3]][col_offset + 4] += 1
        W[w_idx[0]][col_offset + 4] -= 1
        W[w_idx[1]][col_offset + 4] += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[u_idx[2]][col_offset + 5] += 1
        U[u_idx[0]][col_offset + 5] -= 1
        V[v_idx[0]][col_offset + 5] += 1
        V[v_idx[1]][col_offset + 5] += 1
        W[w_idx[3]][col_offset + 5] += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[u_idx[1]][col_offset + 6] += 1
        U[u_idx[3]][col_offset + 6] -= 1
        V[v_idx[2]][col_offset + 6] += 1
        V[v_idx[3]][col_offset + 6] += 1
        W[w_idx[0]][col_offset + 6] += 1

    # Block 1: A_left x B_tl -> C_left
    # A_left: [0, 1, 4, 5] (A00, A01, A10, A11)
    # B_tl:   [0, 1, 4, 5] (B00, B01, B10, B11)
    # C_left: [0, 2, 1, 3] (C00, C01, C10, C11) -> W[0], W[2], W[1], W[3]
    u1 = [0, 1, 4, 5]
    v1 = [0, 1, 4, 5]
    w1 = [0, 2, 1, 3]
    add_strassen_block(u1, v1, w1, 0)

    # Block 2: A_right x B_bl -> C_left
    # A_right: [2, 3, 6, 7] (A02, A03, A12, A13)
    # B_bl:    [8, 9, 12, 13] (B20, B21, B30, B31)
    # C_left:  [0, 2, 1, 3]
    u2 = [2, 3, 6, 7]
    v2 = [8, 9, 12, 13]
    w2 = [0, 2, 1, 3]
    add_strassen_block(u2, v2, w2, 7)

    # Block 3: A_left x B_tr -> C_right
    # A_left:  [0, 1, 4, 5]
    # B_tr:    [2, 3, 6, 7] (B02, B03, B12, B13)
    # C_right: [4, 6, 5, 7] (C02, C03, C12, C13) -> W[4], W[6], W[5], W[7]
    u3 = [0, 1, 4, 5]
    v3 = [2, 3, 6, 7]
    w3 = [4, 6, 5, 7]
    add_strassen_block(u3, v3, w3, 14)

    # Block 4: A_right x B_br -> C_right
    # A_right: [2, 3, 6, 7]
    # B_br:    [10, 11, 14, 15] (B22, B23, B32, B33)
    # C_right: [4, 6, 5, 7]
    u4 = [2, 3, 6, 7]
    v4 = [10, 11, 14, 15]
    w4 = [4, 6, 5, 7]
    add_strassen_block(u4, v4, w4, 21)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
