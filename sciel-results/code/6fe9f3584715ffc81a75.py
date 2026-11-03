import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    rank = 41
    U = [[0] * rank for _ in range(n * m)]  # 6 x 41
    V = [[0] * rank for _ in range(m * p)]  # 24 x 41
    W = [[0] * rank for _ in range(p * n)]  # 16 x 41
    
    col = 0

    def add_term(u_idx, v_idx, w_idx, val):
        nonlocal col
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1

    def add_strassen(a_rows, b_rows, c_cols, a_offset, b_offset, c_offset):
        # a_rows: list of 2 row indices for A
        # b_rows: list of 2 row indices for B
        # c_cols: list of 2 col indices for C
        # Offsets are global indices
        
        # M1 = (A00 + A11) * (B00 + B11)
        u0 = a_rows[0] * m + a_offset[0]
        u1 = a_rows[1] * m + a_offset[1]
        v0 = b_rows[0] * p + b_offset[0]
        v1 = b_rows[1] * p + b_offset[1]
        w0 = c_cols[0] * n + c_offset[0]
        w1 = c_cols[1] * n + c_offset[1]
        
        # We need to add 7 terms.
        # This is getting complex with offsets. 
        # Let's just write out the 7 terms explicitly for each block.
        pass

    # It's easier to just write the loops for each part.
    
    # Part 1: 2x3x3 (cols 0,1,2 of B)
    # 1a: 2x2x2 (A rows 0,1; B rows 0,1; C cols 0,1)
    # A indices: (0,0), (0,1), (1,0), (1,1) -> u_idx: 0, 1, 3, 4
    # B indices: (0,0), (0,1), (1,0), (1,1) -> v_idx: 0, 1, 8, 9
    # C indices: (0,0), (0,1), (1,0), (1,1) -> w_idx: 0, 1, 2, 3
    
    # Strassen 2x2x2
    # M1 = (A00 + A11) * (B00 + B11)
    U[0][col] += 1; U[4][col] += 1
    V[0][col] += 1; V[9][col] += 1
    W[0][col] += 1; W[3][col] += 1
    col += 1
    
    # M2 = (A10 + A11) * B00
    U[3][col] += 1; U[4][col] += 1
    V[0][col] += 1
    W[2][col] += 1; W[3][col] -= 1
    col += 1
    
    # M3 = A00 * (B01 - B11)
    U[0][col] += 1
    V[1][col] += 1; V[9][col] -= 1
    W[1][col] += 1; W[3][col] += 1
    col += 1
    
    # M4 = A11 * (B10 - B00)
    U[4][col] += 1
    V[8][col] += 1; V[0][col] -= 1
    W[0][col] += 1; W[2][col] += 1
    col += 1
    
    # M5 = (A00 + A01) * B11
    U[0][col] += 1; U[1][col] += 1
    V[9][col] += 1
    W[0][col] -= 1; W[1][col] += 1
    col += 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[3][col] += 1; U[0][col] -= 1
    V[0][col] += 1; V[1][col] += 1
    W[3][col] += 1
    col += 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1][col] += 1; U[4][col] -= 1
    V[8][col] += 1; V[9][col] += 1
    W[0][col] += 1
    col += 1

    # 1b: 2x1x2 (A row 2; B rows 0,1; C cols 0,1)
    # A index: (2,0), (2,1) -> u_idx: 6, 7
    # B indices: (0,0), (0,1), (1,0), (1,1) -> v_idx: 0, 1, 8, 9
    # C indices: (0,0), (0,1), (1,0), (1,1) -> w_idx: 0, 1, 2, 3
    
    # C00 += A20*B00, C01 += A20*B01
    U[6][col] += 1; V[0][col] += 1; W[0][col] += 1; col += 1
    U[6][col] += 1; V[1][col] += 1; W[1][col] += 1; col += 1
    # C10 += A21*B00, C11 += A21*B01
    U[7][col] += 1; V[0][col] += 1; W[2][col] += 1; col += 1
    U[7][col] += 1; V[1][col] += 1; W[3][col] += 1; col += 1

    # 1c: 2x2x1 (A rows 0,1; B row 2; C cols 0,1)
    # A indices: (0,0), (0,1), (1,0), (1,1) -> u_idx: 0, 1, 3, 4
    # B indices: (2,0), (2,1) -> v_idx: 16, 17
    # C indices: (0,0), (0,1), (1,0), (1,1) -> w_idx: 0, 1, 2, 3
    
    # C00 += A00*B20, C01 += A00*B21
    U[0][col] += 1; V[16][col] += 1; W[0][col] += 1; col += 1
    U[0][col] += 1; V[17][col] += 1; W[1][col] += 1; col += 1
    # C10 += A10*B20, C11 += A10*B21
    U[3][col] += 1; V[16][col] += 1; W[2][col] += 1; col += 1
    U[3][col] += 1; V[17][col] += 1; W[3][col] += 1; col += 1

    # Part 2: 2x3x5 (cols 3,4,5,6,7 of B)
    # Split into 2x3x3 (cols 3,4,5) and 2x3x2 (cols 6,7)
    
    # 2a: 2x3x3 (cols 3,4,5)
    # Same structure as Part 1, but shifted.
    # B cols 3,4,5 correspond to B rows 0,1,2 but columns 3,4,5.
    # Wait, B is 3x8.
    # B[i, j] where j is 3,4,5.
    # v_idx = i * 8 + j.
    
    # 2a1: 2x2x2 (A rows 0,1; B rows 0,1; C cols 3,4)
    # A u_idx: 0, 1, 3, 4
    # B v_idx: 0*8+3=3, 0*8+4=4, 1*8+3=11, 1*8+4=12
    # C w_idx: 3*2+0=6, 4*2+0=8, 3*2+1=7, 4*2+1=9
    
    # M1
    U[0][col] += 1; U[4][col] += 1
    V[3][col] += 1; V[12][col] += 1
    W[6][col] += 1; W[9][col] += 1
    col += 1
    # M2
    U[3][col] += 1; U[4][col] += 1
    V[3][col] += 1
    W[8][col] += 1; W[9][col] -= 1
    col += 1
    # M3
    U[0][col] += 1
    V[4][col] += 1; V[12][col] -= 1
    W[7][col] += 1; W[9][col] += 1
    col += 1
    # M4
    U[4][col] += 1
    V[11][col] += 1; V[3][col] -= 1
    W[6][col] += 1; W[8][col] += 1
    col += 1
    # M5
    U[0][col] += 1; U[1][col] += 1
    V[12][col] += 1
    W[6][col] -= 1; W[7][col] += 1
    col += 1
    # M6
    U[3][col] += 1; U[0][col] -= 1
    V[3][col] += 1; V[4][col] += 1
    W[9][col] += 1
    col += 1
    # M7
    U[1][col] += 1; U[4][col] -= 1
    V[11][col] += 1; V[12][col] += 1
    W[6][col] += 1
    col += 1

    # 2a2: 2x1x2 (A row 2; B rows 0,1; C cols 3,4)
    # A u_idx: 6, 7
    # B v_idx: 3, 4, 11, 12
    # C w_idx: 6, 7, 8, 9
    U[6][col] += 1; V[3][col] += 1; W[6][col] += 1; col += 1
    U[6][col] += 1; V[4][col] += 1; W[7][col] += 1; col += 1
    U[7][col] += 1; V[3][col] += 1; W[8][col] += 1; col += 1
    U[7][col] += 1; V[4][col] += 1; W[9][col] += 1; col += 1

    # 2a3: 2x2x1 (A rows 0,1; B row 2; C cols 3,4)
    # A u_idx: 0, 1, 3, 4
    # B v_idx: 19, 20 (2*8+3, 2*8+4)
    # C w_idx: 6, 7, 8, 9
    U[0][col] += 1; V[19][col] += 1; W[6][col] += 1; col += 1
    U[0][col] += 1; V[20][col] += 1; W[7][col] += 1; col += 1
    U[3][col] += 1; V[19][col] += 1; W[8][col] += 1; col += 1
    U[3][col] += 1; V[20][col] += 1; W[9][col] += 1; col += 1

    # 2b: 2x3x2 (cols 6,7)
    # Rank 11.
    # Split m=3 into 1+2.
    # 2b1: 2x1x2 (A row 0; B rows 0,1,2; C cols 6,7) -> Wait, A is 2x3.
    # A rows 0,1. B rows 0,1,2.
    # We split A into A_row0 (1x3) and A_row1 (1x3)? No.
    # We split the multiplication $2 \times 3 \times 2$ into:
    # $1 \times 3 \times 2$ (A row 0) + $1 \times 3 \times 2$ (A row 1)?
    # Rank of $1 \times 3 \times 2$ is 6.
    # Total 12.
    # But we want 11.
    # The 11 rank decomposition is:
    # $2 \times 1 \times 2$ (A col 0) + $2 \times 2 \times 2$ (A cols 1,2).
    # A is $2 \times 3$.
    # A_col0 is $2 \times 1$.
    # A_cols12 is $2 \times 2$.
    # B is $3 \times 2$.
    # B_row0 is $1 \times 2$.
    # B_rows12 is $2 \times 2$.
    
    # 2b1: 2x1x2 (A col 0; B row 0; C cols 6,7)
    # A u_idx: 0 (A[0,0]), 3 (A[1,0])
    # B v_idx: 6 (B[0,6]), 7 (B[0,7])
    # C w_idx: 12, 13, 14, 15
    U[0][col] += 1; V[6][col] += 1; W[12][col] += 1; col += 1
    U[0][col] += 1; V[7][col] += 1; W[13][col] += 1; col += 1
    U[3][col] += 1; V[6][col] += 1; W[14][col] += 1; col += 1
    U[3][col] += 1; V[7][col] += 1; W[15][col] += 1; col += 1

    # 2b2: 2x2x2 (A cols 1,2; B rows 1,2; C cols 6,7)
    # A u_idx: 1 (A[0,1]), 2 (A[0,2]), 4 (A[1,1]), 5 (A[1,2])
    # B v_idx: 14 (B[1,6]), 15 (B[1,7]), 22 (B[2,6]), 23 (B[2,7])
    # C w_idx: 12, 13, 14, 15
    
    # M1
    U[1][col] += 1; U[5][col] += 1
    V[14][col] += 1; V[23][col] += 1
    W[12][col] += 1; W[15][col] += 1
    col += 1
    # M2
    U[4][col] += 1; U[5][col] += 1
    V[14][col] += 1
    W[14][col] += 1; W[15][col] -= 1
    col += 1
    # M3
    U[1][col] += 1
    V[15][col] += 1; V[23][col] -= 1
    W[13][col] += 1; W[15][col] += 1
    col += 1
    # M4
    U[5][col] += 1
    V[22][col] += 1; V[14][col] -= 1
    W[12][col] += 1; W[14][col] += 1
    col += 1
    # M5
    U[1][col] += 1; U[2][col] += 1
    V[23][col] += 1
    W[12][col] -= 1; W[13][col] += 1
    col += 1
    # M6
    U[4][col] += 1; U[1][col] -= 1
    V[14][col] += 1; V[15][col] += 1
    W[15][col] += 1
    col += 1
    # M7
    U[2][col] += 1; U[5][col] -= 1
    V[22][col] += 1; V[23][col] += 1
    W[12][col] += 1
    col += 1

    return {"U": U, "V": V, "W": W}
