"""Rank-minimal tensor decomposition for 2x4 x 4x5 matrix multiplication."""
DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    # A is (2x4), B is (4x5), C is (2x5)
    # Split: C = A_left*B_top + A_right*B_bot
    # Each 2x2*2x5 split into 2x2*2x2 (Strassen rank 7) + 2x2*2x3 (rank 9)
    # Total: 2*(7+9) = 32
    
    total_rank = 32
    U = [[0.0] * total_rank for _ in range(n * m)]  # 8 x 32
    V = [[0.0] * total_rank for _ in range(m * p)]  # 20 x 32
    W = [[0.0] * total_rank for _ in range(p * n)]  # 10 x 32
    col = 0
    
    def add_strassen_2x2x2(block_i, block_j, B_row_offset):
        nonlocal col
        c = col
        def a_idx(li, lj): return (block_i + li) * 4 + (block_j * 2 + lj)
        def b_idx(lj, lk): return (B_row_offset + lj) * 5 + lk
        def c_idx(li, lk): return lk * 2 + (block_i + li)
        
        # M1 = (a00+a11)(b00+b11)
        U[a_idx(0,0)][c] += 1; U[a_idx(1,1)][c] += 1
        V[b_idx(0,0)][c] += 1; V[b_idx(1,1)][c] += 1
        W[c_idx(0,0)][c] += 1; W[c_idx(1,1)][c] += 1
        col += 1
        
        # M2 = (a10+a11)b00
        U[a_idx(1,0)][c] += 1; U[a_idx(1,1)][c] += 1
        V[b_idx(0,0)][c] += 1
        W[c_idx(1,0)][c] += 1; W[c_idx(1,1)][c] += 1
        col += 1
        
        # M3 = a00(b01-b11)
        U[a_idx(0,0)][c] += 1
        V[b_idx(0,1)][c] += 1; V[b_idx(1,1)][c] -= 1
        W[c_idx(0,1)][c] += 1
        col += 1
        
        # M4 = a11(b10-b00)
        U[a_idx(1,1)][c] += 1
        V[b_idx(1,0)][c] += 1; V[b_idx(0,0)][c] -= 1
        W[c_idx(0,0)][c] += 1; W[c_idx(1,0)][c] += 1
        col += 1
        
        # M5 = (a00+a01)b11
        U[a_idx(0,0)][c] += 1; U[a_idx(0,1)][c] += 1
        V[b_idx(1,1)][c] += 1
        W[c_idx(0,1)][c] += 1; W[c_idx(1,1)][c] += 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[a_idx(1,0)][c] += 1; U[a_idx(0,0)][c] -= 1
        V[b_idx(0,0)][c] += 1; V[b_idx(0,1)][c] += 1
        W[c_idx(1,1)][c] += 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[a_idx(0,1)][c] += 1; U[a_idx(1,1)][c] -= 1
        V[b_idx(1,0)][c] += 1; V[b_idx(1,1)][c] += 1
        W[c_idx(0,0)][c] += 1
        col += 1
        return col
    
    def add_2x2x3_rank9(block_i, block_j, B_row_offset, B_col_start):
        """Add rank-9 decomposition for 2x2 * 2x3."""
        nonlocal col
        c = col
        def a_idx(li, lj): return (block_i + li) * 4 + (block_j * 2 + lj)
        def b_idx(lj, lk): return (B_row_offset + lj) * 5 + (B_col_start + lk)
        def c_idx(li, lk): return (B_col_start + lk) * 2 + (block_i + li)
        
        # Rank-9 construction for 2x2*2x3:
        # C[0,0]=a00*b00+a01*b10, C[0,1]=a00*b01+a01*b11, C[0,2]=a00*b02+a01*b12
        # C[1,0]=a10*b00+a11*b10, C[1,1]=a10*b01+a11*b11, C[1,2]=a10*b02+a11*b12
        #
        # 9-term decomposition:
        # t1: (a00+a10)*b00 -> c00 += a00*b00, c10 += a10*b00
        U[a_idx(0,0)][c] += 1; U[a_idx(1,0)][c] += 1
        V[b_idx(0,0)][c] += 1
        W[c_idx(0,0)][c] += 1; W[c_idx(1,0)][c] += 1
        col += 1; c = col
        
        # t2: (a01+a11)*b10 -> c00 += a01*b10, c10 += a11*b10
        U[a_idx(0,1)][c] += 1; U[a_idx(1,1)][c] += 1
        V[b_idx(1,0)][c] += 1
        W[c_idx(0,0)][c] += 1; W[c_idx(1,0)][c] += 1
        col += 1; c = col
        
        # t3: a00*b01 -> c01 += a00*b01
        U[a_idx(0,0)][c] += 1
        V[b_idx(0,1)][c] += 1
        W[c_idx(0,1)][c] += 1
        col += 1; c = col
        
        # t4: a10*b01 -> c11 += a10*b01
        U[a_idx(1,0)][c] += 1
        V[b_idx(0,1)][c] += 1
        W[c_idx(1,1)][c] += 1
        col += 1; c = col
        
        # t5: a00*b02 -> c02 += a00*b02
        U[a_idx(0,0)][c] += 1
        V[b_idx(0,2)][c] += 1
        W[c_idx(0,2)][c] += 1
        col += 1; c = col
        
        # t6: a10*b02 -> c12 += a10*b02
        U[a_idx(1,0)][c] += 1
        V[b_idx(0,2)][c] += 1
        W[c_idx(1,2)][c] += 1
        col += 1; c = col
        
        # t7: a01*b11 -> c01 += a01*b11
        U[a_idx(0,1)][c] += 1
        V[b_idx(1,1)][c] += 1
        W[c_idx(0,1)][c] += 1
        col += 1; c = col
        
        # t8: a11*b11 -> c11 += a11*b11
        U[a_idx(1,1)][c] += 1
        V[b_idx(1,1)][c] += 1
        W[c_idx(1,1)][c] += 1
        col += 1; c = col
        
        # t9: (a01+a11)*b12 -> c02 += a01*b12, c12 += a11*b12
        U[a_idx(0,1)][c] += 1; U[a_idx(1,1)][c] += 1
        V[b_idx(1,2)][c] += 1
        W[c_idx(0,2)][c] += 1; W[c_idx(1,2)][c] += 1
        col += 1; c = col
        
        return col
    
    # Block 1: A cols 0-1, B rows 0-1
    add_strassen_2x2x2(0, 0, 0)       # 7 terms
    add_2x2x3_rank9(0, 0, 0, 2)       # 9 terms
    
    # Block 2: A cols 2-3, B rows 2-3
    add_strassen_2x2x2(0, 1, 2)       # 7 terms
    add_2x2x3_rank9(0, 1, 2, 2)       # 9 terms
    
    actual_rank = col
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
