"""Matrix multiplication 2x4 x 4x6 rank-minimal decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Split inner dim 4 into two halves of 2
    # A (2x4) = [A_left (2x2), A_right (2x2)]
    # B (4x6) = [B_top (2x6); B_bottom (2x6)]
    # C = A_left * B_top + A_right * B_bottom
    # Each 2x2 * 2x6 has rank 21 (Strassen on 2x2 part, replicated for 3 col-pairs)
    # Total rank = 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen 2x2 x 2x2 coefficients
    # For a 2x2 * 2x2 multiplication:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = A*B
    # 7 products:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 2x2 * 2x6: B has 3 column-pairs, each 2x2
    # For each block b (0,1,2), we have 7 products, total 21
    
    # U indices for A_left (rows 0-3 of A, i.e., n*m = 8, but A_left is first 2 rows of 4 cols)
    # A is 2x4, indexed as A[i, j] where i in 0..1, j in 0..3
    # U row index = i*m + j = i*4 + j
    # A_left: i in 0..1, j in 0..1 -> U rows 0,1,4,5
    # A_right: i in 0..1, j in 2..3 -> U rows 2,3,6,7
    
    # V indices for B_top (rows 0-1 of B, 2x6) and B_bottom (rows 2-3 of B, 2x6)
    # B is 4x6, indexed B[j, k] where j in 0..3, k in 0..5
    # V row index = j*p + k = j*6 + k
    # B_top: j in 0..1, k in 0..5 -> V rows 0-5, 6-11
    # B_bottom: j in 2..3, k in 0..5 -> V rows 12-17, 18-23
    
    # W indices for C (2x6), W[k*n + i, c]
    # C[i, k] where i in 0..1, k in 0..5
    # W row index = k*n + i = k*2 + i
    
    # Helper: add a Strassen 2x2*2x2 term
    # a_rows: 4 U row indices for A = [[a00, a01], [a10, a11]]
    # b_rows: 4 V row indices for B = [[b00, b01], [b10, b11]]
    # c_rows: 4 W row indices for C = [[c00, c01], [c10, c11]]
    # col: starting column offset
    # Returns next column
    
    def add_strassen_2x2_2x2(a_rows, b_rows, c_rows, col):
        # a_rows = [a00, a01, a10, a11]
        # b_rows = [b00, b01, b10, b11]
        # c_rows = [c00, c01, c10, c11]
        a00, a01, a10, a11 = a_rows
        b00, b01, b10, b11 = b_rows
        c00, c01, c10, c11 = c_rows
        
        # M1 = (a00+a11)*(b00+b11)
        U[a00][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M2 = (a10+a11)*b00
        U[a10][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # M3 = a00*(b01-b11)
        U[a00][col] += 1
        V[b01][col] += 1
        V[b11][col] -= 1
        W[c01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M4 = a11*(b10-b00)
        U[a11][col] += 1
        V[b10][col] += 1
        V[b00][col] -= 1
        W[c00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # M5 = (a00+a01)*b11
        U[a00][col] += 1
        U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1
        W[c01][col] += 1
        col += 1
        
        # M6 = (a10-a00)*(b00+b01)
        U[a10][col] += 1
        U[a00][col] -= 1
        V[b00][col] += 1
        V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7 = (a01-a11)*(b10+b11)
        U[a01][col] += 1
        U[a11][col] -= 1
        V[b10][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c11][col] -= 1
        col += 1
        
        return col
    
    # A_left: A[i,j] for i in 0..1, j in 0..1
    # U row for A[i,j] = i*4 + j
    # a00 = A[0,0] -> U[0]
    # a01 = A[0,1] -> U[1]
    # a10 = A[1,0] -> U[4]
    # a11 = A[1,1] -> U[5]
    
    # A_right: A[i,j] for i in 0..1, j in 2..3
    # a00 = A[0,2] -> U[2]
    # a01 = A[0,3] -> U[3]
    # a10 = A[1,2] -> U[6]
    # a11 = A[1,3] -> U[7]
    
    # B_top: B[j,k] for j in 0..1, k in 0..5
    # For block b (0,1,2), columns 2b, 2b+1
    # b00 = B[0, 2b] -> V[0*6 + 2b] = V[2b]
    # b01 = B[0, 2b+1] -> V[2b+1]
    # b10 = B[1, 2b] -> V[1*6 + 2b] = V[6+2b]
    # b11 = B[1, 2b+1] -> V[6+2b+1]
    
    # B_bottom: B[j,k] for j in 2..3, k in 0..5
    # b00 = B[2, 2b] -> V[2*6 + 2b] = V[12+2b]
    # b01 = B[2, 2b+1] -> V[12+2b+1]
    # b10 = B[3, 2b] -> V[3*6 + 2b] = V[18+2b]
    # b11 = B[3, 2b+1] -> V[18+2b+1]
    
    # C: C[i,k] for i in 0..1, k in 0..5
    # W row for C[i,k] = k*2 + i
    # For block b, columns 2b, 2b+1
    # c00 = C[0, 2b] -> W[2b*2 + 0] = W[4b]
    # c01 = C[0, 2b+1] -> W[4b+1]
    # c10 = C[1, 2b] -> W[4b+2]
    # c11 = C[1, 2b+1] -> W[4b+3]
    
    col = 0
    
    # A_left * B_top
    for b in range(3):
        a_rows = [0, 1, 4, 5]  # a00, a01, a10, a11
        b_rows = [2*b, 2*b+1, 6+2*b, 6+2*b+1]  # b00, b01, b10, b11
        c_rows = [4*b, 4*b+1, 4*b+2, 4*b+3]  # c00, c01, c10, c11
        col = add_strassen_2x2_2x2(a_rows, b_rows, c_rows, col)
    
    # A_right * B_bottom
    for b in range(3):
        a_rows = [2, 3, 6, 7]  # a00, a01, a10, a11
        b_rows = [12+2*b, 12+2*b+1, 18+2*b, 18+2*b+1]  # b00, b01, b10, b11
        c_rows = [4*b, 4*b+1, 4*b+2, 4*b+3]  # c00, c01, c10, c11
        col = add_strassen_2x2_2x2(a_rows, b_rows, c_rows, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
