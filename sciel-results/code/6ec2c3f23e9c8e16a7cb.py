import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=5
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (15, rank)
    
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1
        
    def add_strassen(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col
        # A block: 2x2 at (a_row_offset, a_col_offset)
        # B block: 2x2 at (b_row_offset, b_col_offset)
        # C block: 2x2 at (c_row_offset, c_col_offset)
        
        # Indices in U (A): row * m + col
        # Indices in V (B): row * p + col
        # Indices in W (C): row * n + col
        
        # Strassen M1 = (A00+A11)(B00+B11)
        # A00: (ar, ac), A11: (ar+1, ac+1)
        # B00: (br, bc), B11: (br+1, bc+1)
        # C00: (cr, cc)
        
        # Term 1
        U[a_row_offset * m + a_col_offset][col] = 1
        U[(a_row_offset+1) * m + (a_col_offset+1)][col] = 1
        V[b_row_offset * p + b_col_offset][col] = 1
        V[(b_row_offset+1) * p + (b_col_offset+1)][col] = 1
        W[c_row_offset * n + c_col_offset][col] = 1
        W[(c_row_offset+1) * n + (c_col_offset+1)][col] = 1
        col += 1
        
        # Term 2: (A10+A11)B00
        U[a_row_offset * m + a_col_offset][col] = 0 # A10 is (ar+1, ac)
        U[(a_row_offset+1) * m + a_col_offset][col] = 1
        U[(a_row_offset+1) * m + (a_col_offset+1)][col] = 1
        V[b_row_offset * p + b_col_offset][col] = 1
        W[(c_row_offset+1) * n + c_col_offset][col] = 1
        W[(c_row_offset+1) * n + (c_col_offset+1)][col] = 1
        col += 1
        
        # Term 3: A00(B01-B11)
        U[a_row_offset * m + a_col_offset][col] = 1
        V[b_row_offset * p + (b_col_offset+1)][col] = 1
        V[(b_row_offset+1) * p + (b_col_offset+1)][col] = -1
        W[c_row_offset * n + (c_col_offset+1)][col] = 1
        W[(c_row_offset+1) * n + (c_col_offset+1)][col] = 1
        col += 1
        
        # Term 4: A11(B10-B00)
        U[(a_row_offset+1) * m + (a_col_offset+1)][col] = 1
        V[(b_row_offset+1) * p + b_col_offset][col] = 1
        V[b_row_offset * p + b_col_offset][col] = -1
        W[(c_row_offset+1) * n + c_col_offset][col] = 1
        W[(c_row_offset+1) * n + (c_col_offset+1)][col] = 1
        col += 1
        
        # Term 5: (A00+A01)B11
        U[a_row_offset * m + a_col_offset][col] = 1
        U[a_row_offset * m + (a_col_offset+1)][col] = 1
        V[(b_row_offset+1) * p + (b_col_offset+1)][col] = 1
        W[c_row_offset * n + c_col_offset][col] = -1
        W[c_row_offset * n + (c_col_offset+1)][col] = 1
        W[(c_row_offset+1) * n + (c_col_offset+1)][col] = 1
        col += 1
        
        # Term 6: (A10-A00)(B00+B01)
        U[(a_row_offset+1) * m + a_col_offset][col] = 1
        U[a_row_offset * m + a_col_offset][col] = -1
        V[b_row_offset * p + b_col_offset][col] = 1
        V[b_row_offset * p + (b_col_offset+1)][col] = 1
        W[(c_row_offset+1) * n + c_col_offset][col] = 1
        W[(c_row_offset+1) * n + (c_col_offset+1)][col] = 1
        col += 1
        
        # Term 7: (A01-A11)(B10+B11)
        U[a_row_offset * m + (a_col_offset+1)][col] = 1
        U[(a_row_offset+1) * m + (a_col_offset+1)][col] = -1
        V[(b_row_offset+1) * p + b_col_offset][col] = 1
        V[(b_row_offset+1) * p + (b_col_offset+1)][col] = 1
        W[c_row_offset * n + c_col_offset][col] = 1
        W[c_row_offset * n + (c_col_offset+1)][col] = -1
        W[(c_row_offset+1) * n + (c_col_offset+1)][col] = 1
        col += 1

    # 1. A_tL B_lT1: A(0:2, 0:2) * B(0:2, 0:2) -> C(0:2, 0:2)
    add_strassen(0, 0, 0, 0, 0, 0)
    
    # 2. A_tL B_lT2: A(0:2, 0:2) * B(0:2, 2:4) -> C(0:2, 2:4)
    add_strassen(0, 0, 0, 2, 0, 2)
    
    # 3. A_tR B_lB1: A(0:2, 2:4) * B(2:4, 0:2) -> C(0:2, 0:2)
    add_strassen(0, 2, 2, 0, 0, 0)
    
    # 4. A_tR B_lB2: A(0:2, 2:4) * B(2:4, 2:4) -> C(0:2, 2:4)
    add_strassen(0, 2, 2, 2, 0, 2)
    
    # 5. A_tL B_rT: A(0:2, 0:2) * B(0:2, 4) -> C(0:2, 4)
    # Schoolbook 2x2x1
    # C0,4 += A00 B04 + A01 B14
    add_term(0*m+0, 1, 0*p+4, 1, 4*n+0, 1)
    add_term(0*m+1, 1, 1*p+4, 1, 4*n+0, 1)
    # C1,4 += A10 B04 + A11 B14
    add_term(1*m+0, 1, 0*p+4, 1, 4*n+1, 1)
    add_term(1*m+1, 1, 1*p+4, 1, 4*n+1, 1)
    
    # 6. A_tR B_rB: A(0:2, 2:4) * B(2:4, 4) -> C(0:2, 4)
    # Schoolbook 2x2x1
    # C0,4 += A02 B24 + A03 B34
    add_term(0*m+2, 1, 2*p+4, 1, 4*n+0, 1)
    add_term(0*m+3, 1, 3*p+4, 1, 4*n+0, 1)
    # C1,4 += A12 B24 + A13 B34
    add_term(1*m+2, 1, 2*p+4, 1, 4*n+1, 1)
    add_term(1*m+3, 1, 3*p+4, 1, 4*n+1, 1)
    
    # 7. A_bL B_lT1: A(2, 0:2) * B(0:2, 0:2) -> C(2, 0:2)
    # Schoolbook 1x2x2
    # C2,0 += A20 B00 + A21 B10
    add_term(2*m+0, 1, 0*p+0, 1, 4*n+2, 1)
    add_term(2*m+1, 1, 1*p+0, 1, 4*n+2, 1)
    # C2,1 += A20 B01 + A21 B11
    add_term(2*m+0, 1, 0*p+1, 1, 4*n+2, 1)
    add_term(2*m+1, 1, 1*p+1, 1, 4*n+2, 1)
    
    # 8. A_bL B_lT2: A(2, 0:2) * B(0:2, 2:4) -> C(2, 2:4)
    # Schoolbook 1x2x2
    # C2,2 += A20 B02 + A21 B12
    add_term(2*m+0, 1, 0*p+2, 1, 4*n+2, 1)
    add_term(2*m+1, 1, 1*p+2, 1, 4*n+2, 1)
    # C2,3 += A20 B03 + A21 B13
    add_term(2*m+0, 1, 0*p+3, 1, 4*n+2, 1)
    add_term(2*m+1, 1, 1*p+3, 1, 4*n+2, 1)
    
    # 9. A_bR B_lB1: A(2, 2:4) * B(2:4, 0:2) -> C(2, 0:2)
    # Schoolbook 1x2x2
    # C2,0 += A22 B20 + A23 B30
    add_term(2*m+2, 1, 2*p+0, 1, 4*n+2, 1)
    add_term(2*m+3, 1, 3*p+0, 1, 4*n+2, 1)
    # C2,1 += A22 B21 + A23 B31
    add_term(2*m+2, 1, 2*p+1, 1, 4*n+2, 1)
    add_term(2*m+3, 1, 3*p+1, 1, 4*n+2, 1)
    
    # 10. A_bR B_lB2: A(2, 2:4) * B(2:4, 2:4) -> C(2, 2:4)
    # Schoolbook 1x2x2
    # C2,2 += A22 B22 + A23 B32
    add_term(2*m+2, 1, 2*p+2, 1, 4*n+2, 1)
    add_term(2*m+3, 1, 3*p+2, 1, 4*n+2, 1)
    # C2,3 += A22 B23 + A23 B33
    add_term(2*m+2, 1, 2*p+3, 1, 4*n+2, 1)
    add_term(2*m+3, 1, 3*p+3, 1, 4*n+2, 1)
    
    # 11. A_bL B_rT: A(2, 0:2) * B(0:2, 4) -> C(2, 4)
    # Schoolbook 1x2x1
    # C2,4 += A20 B04 + A21 B14
    add_term(2*m+0, 1, 0*p+4, 1, 4*n+2, 1)
    add_term(2*m+1, 1, 1*p+4, 1, 4*n+2, 1)
    
    # 12. A_bR B_rB: A(2, 2:4) * B(2:4, 4) -> C(2, 4)
    # Schoolbook 1x2x1
    # C2,4 += A22 B24 + A23 B34
    add_term(2*m+2, 1, 2*p+4, 1, 4*n+2, 1)
    add_term(2*m+3, 1, 3*p+4, 1, 4*n+2, 1)
    
    return {"U": U, "V": V, "W": W}
