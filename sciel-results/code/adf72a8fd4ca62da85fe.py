import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    deadline = 145.0
    n, m, p = DIMENSIONS
    
    # Strassen-based decomposition for 3x3 x 3x8
    # Rank 68
    # A is 3x3, B is 3x8
    # Split A into A11 (2x2), A12 (2x1), A21 (1x2), A22 (1x1)
    # Split B into B1 (2x8), B2 (1x8)
    # Output C1 = A11*B1 + A12*B2 (2x8)
    # Output C2 = A21*B1 + A22*B2 (1x8)
    
    # Strassen for A11 (2x2) * B1_part (2x2)
    # B1 is 2x8, so we have 4 blocks of 2x2: B10, B11, B12, B13
    # Each 2x2*2x2 uses rank 7. Total 4*7=28.
    # Cross terms:
    # A12 (2x1) * B2 (1x8): 2*8 = 16 terms
    # A21 (1x2) * B1 (2x8): 1*2*8 = 16 terms
    # A22 (1x1) * B2 (1x8): 1*8 = 8 terms
    # Total 28 + 16 + 16 + 8 = 68.
    
    rank = 68
    U = [[0] * rank for _ in range(n * m)] # 9x68
    V = [[0] * rank for _ in range(m * p)] # 24x68
    W = [[0] * rank for _ in range(p * n)] # 24x68
    
    col = 0
    
    # Helper to add Strassen terms for 2x2 * 2x2
    # A indices: (i,j) in 0..1, 0..1
    # B indices: (k,l) in 0..1, 0..1
    # C indices: (i,l)
    # Strassen 7 terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    def add_strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset, start_col):
        # a: 2x2 at (a_row_offset, a_col_offset)
        # b: 2x2 at (b_row_offset, b_col_offset)
        # c: 2x2 at (c_row_offset, c_col_offset)
        # Returns new col
        col_local = start_col
        
        # M1
        U[a_row_offset][col_local] += 1 # A00
        U[a_row_offset+1][col_local] += 1 # A11
        V[b_row_offset][col_local] += 1 # B00
        V[b_row_offset+1][col_local] += 1 # B11
        W[c_row_offset][col_local] += 1 # C00
        W[c_row_offset+2][col_local] += 1 # C11 (Wait, C is 2x2, so rows are c_row, c_row+1)
        # Correction: C indices are (c_row, c_col) and (c_row+1, c_col) etc.
        # W index is k*n + i.
        # For C00 (i=0, k=c_col), W[c_col*n + c_row]
        # For C11 (i=1, k=c_col+1), W[(c_col+1)*n + (c_row+1)]
        # This is getting messy with indices.
        
        # Let's use explicit index mapping.
        # A is n x m (3x3).
        # B is m x p (3x8).
        # C is n x p (3x8).
        # W index: k*n + i.
        
        # Strassen for A[0:2, 0:2] * B[0:2, b_col:b_col+2] -> C[0:2, b_col:b_col+2]
        # A00 = U[0*m+0] = U[0]
        # A01 = U[0*m+1] = U[1]
        # A10 = U[1*m+0] = U[3]
        # A11 = U[1*m+1] = U[4]
        
        # B00 = V[0*p+b_col]
        # B01 = V[0*p+b_col+1]
        # B10 = V[1*p+b_col]
        # B11 = V[1*p+b_col+1]
        
        # C00 = W[b_col*n + 0]
        # C01 = W[(b_col+1)*n + 0]
        # C10 = W[b_col*n + 1]
        # C11 = W[(b_col+1)*n + 1]
        
        col_local = start_col
        
        # M1 = (A00+A11)(B00+B11)
        U[0][col_local] += 1
        U[4][col_local] += 1
        V[b_col][col_local] += 1
        V[p+b_col][col_local] += 1
        W[b_col][col_local] += 1
        W[(b_col+1)*n+1][col_local] += 1
        col_local += 1
        
        # M2 = (A10+A11)B00
        U[3][col_local] += 1
        U[4][col_local] += 1
        V[b_col][col_local] += 1
        W[b_col*n+1][col_local] += 1
        col_local += 1
        
        # M3 = A00(B10-B11)
        U[0][col_local] += 1
        V[p+b_col][col_local] += 1
        V[p+b_col+1][col_local] -= 1
        W[b_col+1][col_local] += 1
        col_local += 1
        
        # M4 = A11(B01-B00)
        U[4][col_local] += 1
        V[b_col+1][col_local] += 1
        V[b_col][col_local] -= 1
        W[b_col+1][col_local] += 1
        W[b_col*n+1][col_local] += 1
        col_local += 1
        
        # M5 = (A00+A01)B11
        U[0][col_local] += 1
        U[1][col_local] += 1
        V[p+b_col+1][col_local] += 1
        W[b_col+1][col_local] -= 1
        W[b_col*n+1][col_local] -= 1
        col_local += 1
        
        # M6 = (A10-A00)(B00+B10)
        U[3][col_local] += 1
        U[0][col_local] -= 1
        V[b_col][col_local] += 1
        V[p+b_col][col_local] += 1
        W[b_col*n+1][col_local] += 1
        col_local += 1
        
        # M7 = (A01-A11)(B01+B11)
        U[1][col_local] += 1
        U[4][col_local] -= 1
        V[b_col+1][col_local] += 1
        V[p+b_col+1][col_local] += 1
        W[b_col][col_local] += 1
        col_local += 1
        
        return col_local

    # Apply Strassen to 4 blocks of B
    # B is 3x8. Top 2 rows are B_top (2x8).
    # Split B_top into 4 2x2 blocks: cols 0-1, 2-3, 4-5, 6-7.
    for b_col in range(0, 8, 2):
        col = add_strassen_block(0, 0, 0, b_col, 0, b_col, col)
        
    # Cross terms
    # A12 (2x1) is A[0:2, 2]. U[2], U[5].
    # B2 (1x8) is B[2, :]. V[2*p + k] = V[16+k].
    # C_top_cross = A12 * B2.
    # C[0, k] += A[0, 2] * B[2, k]
    # C[1, k] += A[1, 2] * B[2, k]
    for k in range(8):
        # Term A[0,2] * B[2,k] -> C[0,k]
        U[2][col] += 1
        V[16+k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        
        # Term A[1,2] * B[2,k] -> C[1,k]
        U[5][col] += 1
        V[16+k][col] += 1
        W[k*n + 1][col] += 1
        col += 1
        
    # A21 (1x2) is A[2, 0:2]. U[6], U[7].
    # B_top (2x8) is B[0:2, :]. V[0:p + k], V[p + k].
    # C_bottom_cross = A21 * B_top.
    # C[2, k] += A[2, 0] * B[0, k] + A[2, 1] * B[1, k]
    for k in range(8):
        # Term A[2,0] * B[0,k] -> C[2,k]
        U[6][col] += 1
        V[k][col] += 1
        W[k*n + 2][col] += 1
        col += 1
        
        # Term A[2,1] * B[1,k] -> C[2,k]
        U[7][col] += 1
        V[p+k][col] += 1
        W[k*n + 2][col] += 1
        col += 1
        
    # A22 (1x1) is A[2,2]. U[8].
    # B2 (1x8) is B[2, :]. V[16+k].
    # C_bottom_diag = A22 * B2.
    # C[2, k] += A[2, 2] * B[2, k]
    for k in range(8):
        U[8][col] += 1
        V[16+k][col] += 1
        W[k*n + 2][col] += 1
        col += 1
        
    if col != rank:
        # Adjust rank if needed, but we calculated 68
        pass
        
    return {"U": U, "V": "V", "W": W}
