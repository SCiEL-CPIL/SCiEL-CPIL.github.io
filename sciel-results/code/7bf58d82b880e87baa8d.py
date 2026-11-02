"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-28 exact tensor decomposition for 2x4 by 4x4 matrix multiplication.
    
    Uses Strassen's 2x2x2 algorithm applied to four 2x2 blocks of the matrices.
    The 2x4 matrix A is split into two 2x2 blocks: A_left (cols 0-1) and A_right (cols 2-3).
    The 4x4 matrix B is split into four 2x2 blocks: B_tl, B_tr, B_bl, B_br.
    The 2x4 output C = A*B is split into two 2x2 blocks: C_left (cols 0-1) and C_right (cols 2-3).
    
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
    # A is 2x4, flattened as A[k*m+i] where k is row (0,1), i is col (0,1,2,3)
    # So A[0]=A00, A[1]=A01, A[2]=A02, A[3]=A03, A[4]=A10, A[5]=A11, A[6]=A12, A[7]=A13
    # 
    # B is 4x4, flattened as B[k*m+i] where k is row (0-3), i is col (0-3)
    # B[0]=B00, B[1]=B01, B[2]=B02, B[3]=B03,
    # B[4]=B10, B[5]=B11, B[6]=B12, B[7]=B13,
    # B[8]=B20, B[9]=B21, B[10]=B22, B[11]=B23,
    # B[12]=B30, B[13]=B31, B[14]=B32, B[15]=B33
    #
    # C is 2x4, output indexed as W[k*n+i] per the trace convention
    # C[0]=C00, C[1]=C01, C[2]=C02, C[3]=C03,
    # C[4]=C10, C[5]=C11, C[6]=C12, C[7]=C13
    #
    # Wait, let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
    # This means W has shape (p*n, rank) = (4*2, rank) = (8, rank)
    # W[k*n+i] where k is... let me think about what k and i represent.
    # For C = A*B where A is 2x4 and B is 4x4, C is 2x4.
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # The tensor T_{i,j,k} = A[i,k] * B[k,j] -> C[i,j]
    # So the output index is (i,j) where i in 0..1 (row of C), j in 0..3 (col of C)
    # W[k*n+i] with n=2: if we interpret this as the row index being k and col being i...
    # Actually, for matrix multiplication tensor, the three indices are:
    #   - input A index: (row_A, col_A) 
    #   - input B index: (row_B, col_B)
    #   - output C index: (row_C, col_C)
    # With A being n x m, B being m x p, C being n x p.
    # Here n=2, m=4, p=4.
    # U has shape (n*m, rank) = (8, rank): indices (row_A, col_A)
    # V has shape (m*p, rank) = (16, rank): indices (row_B, col_B)
    # W has shape (p*n, rank) = (8, rank): indices... (col_C, row_C)? or (row_C, col_C)?
    #
    # The convention W[k*n+i] suggests: first index k ranges over p values (0..3), second index i ranges over n values (0..1).
    # So W index = k*n + i where k in 0..3, i in 0..1.
    # This means the output is indexed as (k, i) where k is... col_C and i is row_C? 
    # Or k is row_C and i is col_C? But row_C only has 2 values and col_C has 4.
    # Since k ranges over p=4 and i ranges over n=2, k must be col_C and i must be row_C.
    # So W[col_C * n + row_C] = W[j * 2 + i] where C[i,j] is the output element.
    
    # Let me verify with the existing code's comments:
    # "C00 (k=0,i=0) -> 0. C01 (k=1,i=0) -> 2. C10 (k=0,i=1) -> 1. C11 (k=1,i=1) -> 3."
    # This says C01 has k=1, i=0 -> index 1*2+0 = 2. So k is col, i is row.
    # C[i,j] -> W[j*n + i]
    
    # So for C_left (cols 0,1):
    # C[0,0] -> W[0*2+0] = W[0]
    # C[0,1] -> W[1*2+0] = W[2]
    # C[1,0] -> W[0*2+1] = W[1]
    # C[1,1] -> W[1*2+1] = W[3]
    #
    # For C_right (cols 2,3):
    # C[0,2] -> W[2*2+0] = W[4]
    # C[0,3] -> W[3*2+0] = W[6]
    # C[1,2] -> W[2*2+1] = W[5]
    # C[1,3] -> W[3*2+1] = W[7]
    
    def add_strassen_block(u_idx, v_idx, w_idx, col_offset):
        """
        Add a Strassen 2x2x2 block.
        u_idx: [u00, u01, u10, u11] - U row indices for A00, A01, A10, A11
        v_idx: [v00, v01, v10, v11] - V row indices for B00, B01, B10, B11
        w_idx: [w00, w01, w10, w11] - W row indices for C00, C01, C10, C11
        col_offset: starting column in U, V, W
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
    
    # Block 1: A_left(2x2) x B_tl(2x2) -> C_left(2x2)
    # A_left: A[0]=A00, A[1]=A01, A[4]=A10, A[5]=A11
    # B_tl: B[0]=B00, B[1]=B01, B[4]=B10, B[5]=B11
    # C_left: C[0,0]->W[0], C[0,1]->W[2], C[1,0]->W[1], C[1,1]->W[3]
    u1 = [0, 1, 4, 5]
    v1 = [0, 1, 4, 5]
    w1 = [0, 2, 1, 3]
    add_strassen_block(u1, v1, w1, 0)
    
    # Block 2: A_right(2x2) x B_bl(2x2) -> C_left(2x2)
    # A_right: A[2]=A02, A[3]=A03, A[6]=A12, A[7]=A13
    # B_bl: B[8]=B20, B[9]=B21, B[12]=B30, B[13]=B31
    # C_left: same as Block 1
    u2 = [2, 3, 6, 7]
    v2 = [8, 9, 12, 13]
    w2 = [0, 2, 1, 3]
    add_strassen_block(u2, v2, w2, 7)
    
    # Block 3: A_left(2x2) x B_tr(2x2) -> C_right(2x2)
    # A_left: same as Block 1
    # B_tr: B[2]=B02, B[3]=B03, B[6]=B12, B[7]=B13
    # C_right: C[0,2]->W[4], C[0,3]->W[6], C[1,2]->W[5], C[1,3]->W[7]
    u3 = [0, 1, 4, 5]
    v3 = [2, 3, 6, 7]
    w3 = [4, 6, 5, 7]
    add_strassen_block(u3, v3, w3, 14)
    
    # Block 4: A_right(2x2) x B_br(2x2) -> C_right(2x2)
    # A_right: same as Block 2
    # B_br: B[10]=B22, B[11]=B23, B[14]=B32, B[15]=B33
    # C_right: same as Block 3
    u4 = [2, 3, 6, 7]
    v4 = [10, 11, 14, 15]
    w4 = [4, 6, 5, 7]
    add_strassen_block(u4, v4, w4, 21)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
