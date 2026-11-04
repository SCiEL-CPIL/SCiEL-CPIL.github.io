"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen(a_rows, b_rows, c_rows, col_start):
        # a_rows: [a00, a01, a10, a11] indices in U
        # b_rows: [b00, b01, b10, b11] indices in V
        # c_rows: [c00, c01, c10, c11] indices in W
        # Strassen 7 multiplications
        # m1 = (a00+a10)(b00+b10)
        # m2 = (a10+a11)b00
        # m3 = a00(b01-b11)
        # m4 = a11(b10-b11)
        # m5 = (a00+a01)b11
        # m6 = (a10-a00)(b00+b01)
        # m7 = (a01-a11)(b10+b11)
        
        # m1
        U[a_rows[0]][col_start] += 1
        U[a_rows[2]][col_start] += 1
        V[b_rows[0]][col_start] += 1
        V[b_rows[2]][col_start] += 1
        W[c_rows[0]][col_start] += 1 # c00
        W[c_rows[3]][col_start] += 1 # c11
        
        # m2
        U[a_rows[2]][col_start+1] += 1
        U[a_rows[3]][col_start+1] += 1
        V[b_rows[0]][col_start+1] += 1
        W[c_rows[2]][col_start+1] += 1 # c10
        W[c_rows[3]][col_start+1] -= 1 # c11
        
        # m3
        U[a_rows[0]][col_start+2] += 1
        V[b_rows[1]][col_start+2] += 1
        V[b_rows[3]][col_start+2] -= 1
        W[c_rows[1]][col_start+2] += 1 # c01
        W[c_rows[3]][col_start+2] += 1 # c11
        
        # m4
        U[a_rows[3]][col_start+3] += 1
        V[b_rows[2]][col_start+3] += 1
        V[b_rows[3]][col_start+3] -= 1
        W[c_rows[0]][col_start+3] += 1 # c00
        W[c_rows[2]][col_start+3] += 1 # c10
        
        # m5
        U[a_rows[0]][col_start+4] += 1
        U[a_rows[1]][col_start+4] += 1
        V[b_rows[3]][col_start+4] += 1
        W[c_rows[0]][col_start+4] -= 1 # c00
        W[c_rows[1]][col_start+4] += 1 # c01
        
        # m6
        U[a_rows[2]][col_start+5] += 1
        U[a_rows[0]][col_start+5] -= 1
        V[b_rows[0]][col_start+5] += 1
        V[b_rows[1]][col_start+5] += 1
        W[c_rows[3]][col_start+5] += 1 # c11
        
        # m7
        U[a_rows[1]][col_start+6] += 1
        U[a_rows[3]][col_start+6] -= 1
        V[b_rows[2]][col_start+6] += 1
        V[b_rows[3]][col_start+6] += 1
        W[c_rows[0]][col_start+6] += 1 # c00
        W[c_rows[1]][col_start+6] += 1 # c01

    def add_schoolbook_2x2x1(a_rows, b_rows, c_rows, col_start):
        # a_rows: [a00, a01, a10, a11]
        # b_rows: [b00, b01, b10, b11] -> only b00, b10 used (k is fixed)
        # c_rows: [c00, c01, c10, c11] -> only c00, c10 used
        # m1 = a00*b00
        # m2 = a01*b01 (not used here, b is 2x1 so only col 0 of B block)
        # Actually, for 2x2 by 2x1:
        # C00 = a00*b00 + a01*b10
        # C10 = a10*b00 + a11*b10
        # 4 multiplications: a00*b00, a01*b10, a10*b00, a11*b10
        
        # m1 = a00*b00
        U[a_rows[0]][col_start] += 1
        V[b_rows[0]][col_start] += 1
        W[c_rows[0]][col_start] += 1 # c00
        
        # m2 = a01*b10
        U[a_rows[1]][col_start+1] += 1
        V[b_rows[2]][col_start+1] += 1
        W[c_rows[0]][col_start+1] += 1 # c00
        
        # m3 = a10*b00
        U[a_rows[2]][col_start+2] += 1
        V[b_rows[0]][col_start+2] += 1
        W[c_rows[2]][col_start+2] += 1 # c10
        
        # m4 = a11*b10
        U[a_rows[3]][col_start+3] += 1
        V[b_rows[2]][col_start+3] += 1
        W[c_rows[2]][col_start+3] += 1 # c10

    # A rows mapping:
    # A[i,j] -> U[i*m+j]
    # A_left: i in 0,1; j in 0,1
    a_left = [0*4+0, 0*4+1, 1*4+0, 1*4+1] # 0, 1, 4, 5
    # A_right: i in 0,1; j in 2,3
    a_right = [0*4+2, 0*4+3, 1*4+2, 1*4+3] # 2, 3, 6, 7

    # B rows mapping:
    # B[j,k] -> V[j*p+k]
    # B_top: j in 0,1
    # B_bot: j in 2,3
    
    # C rows mapping:
    # C[i,k] -> W[k*n+i]
    
    col = 0
    # Block 1: A_left * B_top[0:2, 0:2]
    b_t1 = [0*7+0, 0*7+1, 1*7+0, 1*7+1] # 0, 1, 7, 8
    c_01 = [0*2+0, 1*2+0, 0*2+1, 1*2+1] # 0, 2, 1, 3
    add_strassen(a_left, b_t1, c_01, col)
    col += 7

    # Block 2: A_left * B_top[0:2, 2:4]
    b_t2 = [0*7+2, 0*7+3, 1*7+2, 1*7+3] # 2, 3, 9, 10
    c_23 = [2*2+0, 3*2+0, 2*2+1, 3*2+1] # 4, 6, 5, 7
    add_strassen(a_left, b_t2, c_23, col)
    col += 7

    # Block 3: A_left * B_top[0:2, 4:6]
    b_t3 = [0*7+4, 0*7+5, 1*7+4, 1*7+5] # 4, 5, 11, 12
    c_45 = [4*2+0, 5*2+0, 4*2+1, 5*2+1] # 8, 10, 9, 11
    add_strassen(a_left, b_t3, c_45, col)
    col += 7

    # Block 4: A_left * B_top[0:2, 6:7]
    b_t4 = [0*7+6, 0*7+6, 1*7+6, 1*7+6] # 6, 6, 13, 13 (dummy for unused)
    c_6 = [6*2+0, 6*2+0, 6*2+1, 6*2+1] # 12, 12, 13, 13
    add_schoolbook_2x2x1(a_left, b_t4, c_6, col)
    col += 4

    # Block 5: A_right * B_bot[2:4, 0:2]
    b_b1 = [2*7+0, 2*7+1, 3*7+0, 3*7+1] # 14, 15, 21, 22
    add_strassen(a_right, b_b1, c_01, col)
    col += 7

    # Block 6: A_right * B_bot[2:4, 2:4]
    b_b2 = [2*7+2, 2*7+3, 3*7+2, 3*7+3] # 16, 17, 23, 24
    add_strassen(a_right, b_b2, c_23, col)
    col += 7

    # Block 7: A_right * B_bot[2:4, 4:6]
    b_b3 = [2*7+4, 2*7+5, 3*7+4, 3*7+5] # 18, 19, 25, 26
    add_strassen(a_right, b_b3, c_45, col)
    col += 7

    # Block 8: A_right * B_bot[2:4, 6:7]
    b_b4 = [2*7+6, 2*7+6, 3*7+6, 3*7+6] # 20, 20, 27, 27
    add_schoolbook_2x2x1(a_right, b_b4, c_6, col)
    col += 4

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
