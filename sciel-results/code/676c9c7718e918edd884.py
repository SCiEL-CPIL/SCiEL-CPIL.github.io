import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2 block
    # u_rows: list of global U row indices for local rows 0,1,2,3
    # v_rows: list of global V row indices for local rows 0,1,2,3
    # w_rows: list of global W row indices for local rows 0,1,2,3
    def add_strassen_block(u_rows, v_rows, w_rows, start_col):
        nonlocal col
        # Strassen coefficients
        # M1..M7
        # U coeffs (local row index -> coeff)
        # Local rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # M1
        U[u_rows[0]][start_col] += 1
        U[u_rows[3]][start_col] += 1
        V[v_rows[0]][start_col] += 1
        V[v_rows[3]][start_col] += 1
        W[w_rows[0]][start_col] += 1 # Z11
        W[w_rows[1]][start_col] += 0 # Z21
        W[w_rows[2]][start_col] += 0 # Z12
        W[w_rows[3]][start_col] += 1 # Z22
        
        # M2
        U[u_rows[2]][start_col+1] += 1
        U[u_rows[3]][start_col+1] += 1
        V[v_rows[0]][start_col+1] += 1
        W[w_rows[0]][start_col+1] += 0
        W[w_rows[1]][start_col+1] += 1 # Z21
        W[w_rows[2]][start_col+1] += 0
        W[w_rows[3]][start_col+1] += 0
        
        # M3
        U[u_rows[0]][start_col+2] += 1
        V[v_rows[1]][start_col+2] += 1
        V[v_rows[3]][start_col+2] += -1
        W[w_rows[0]][start_col+2] += 0
        W[w_rows[1]][start_col+2] += 0
        W[w_rows[2]][start_col+2] += 1 # Z12
        W[w_rows[3]][start_col+2] += -1 # Z22
        
        # M4
        U[u_rows[3]][start_col+3] += 1
        V[v_rows[2]][start_col+3] += 1
        V[v_rows[0]][start_col+3] += -1
        W[w_rows[0]][start_col+3] += 1 # Z11
        W[w_rows[1]][start_col+3] += 1 # Z21
        W[w_rows[2]][start_col+3] += 0
        W[w_rows[3]][start_col+3] += 0
        
        # M5
        U[u_rows[0]][start_col+4] += 1
        U[u_rows[1]][start_col+4] += 1
        V[v_rows[3]][start_col+4] += 1
        W[w_rows[0]][start_col+4] += -1 # Z11
        W[w_rows[1]][start_col+4] += 0
        W[w_rows[2]][start_col+4] += 1 # Z12
        W[w_rows[3]][start_col+4] += 0
        
        # M6
        U[u_rows[2]][start_col+5] += 1
        U[u_rows[0]][start_col+5] += -1
        V[v_rows[0]][start_col+5] += 1
        V[v_rows[1]][start_col+5] += 1
        W[w_rows[0]][start_col+5] += 0
        W[w_rows[1]][start_col+5] += 0
        W[w_rows[2]][start_col+5] += 0
        W[w_rows[3]][start_col+5] += 1 # Z22
        
        # M7
        U[u_rows[1]][start_col+6] += 1
        U[u_rows[3]][start_col+6] += -1
        V[v_rows[2]][start_col+6] += 1
        V[v_rows[3]][start_col+6] += 1
        W[w_rows[0]][start_col+6] += 1 # Z11
        W[w_rows[1]][start_col+6] += 0
        W[w_rows[2]][start_col+6] += 0
        W[w_rows[3]][start_col+6] += 1 # Z22
        
        col += 7

    # Process A1 B1 (2x4 x 4x4)
    # A1 is rows 0,1 of A.
    # B1 is cols 0,1,2,3 of B.
    
    # Block 1.1: A1L (cols 0,1) * B11 (rows 0,1, cols 0,1) -> C[0:2, 0:2]
    # U rows: A[0,0], A[0,1], A[1,0], A[1,1] -> 0, 1, 4, 5
    u_rows = [0, 1, 4, 5]
    # V rows: B[0,0], B[0,1], B[1,0], B[1,1] -> 0, 1, 8, 9
    v_rows = [0, 1, 8, 9]
    # W rows: C[0,0], C[1,0], C[0,1], C[1,1] -> 0, 1, 3, 4
    w_rows = [0, 1, 3, 4]
    add_strassen_block(u_rows, v_rows, w_rows, col)
    
    # Block 1.2: A1R (cols 2,3) * B21 (rows 2,3, cols 0,1) -> C[0:2, 0:2]
    # U rows: A[0,2], A[0,3], A[1,2], A[1,3] -> 2, 3, 6, 7
    u_rows = [2, 3, 6, 7]
    # V rows: B[2,0], B[2,1], B[3,0], B[3,1] -> 16, 17, 24, 25
    v_rows = [16, 17, 24, 25]
    # W rows: same as 1.1
    w_rows = [0, 1, 3, 4]
    add_strassen_block(u_rows, v_rows, w_rows, col)
    
    # Block 1.3: A1L (cols 0,1) * B12 (rows 0,1, cols 2,3) -> C[0:2, 2:4]
    # U rows: 0, 1, 4, 5
    u_rows = [0, 1, 4, 5]
    # V rows: B[0,2], B[0,3], B[1,2], B[1,3] -> 2, 3, 10, 11
    v_rows = [2, 3, 10, 11]
    # W rows: C[0,2], C[1,2], C[0,3], C[1,3] -> 6, 7, 9, 10
    w_rows = [6, 7, 9, 10]
    add_strassen_block(u_rows, v_rows, w_rows, col)
    
    # Block 1.4: A1R (cols 2,3) * B22 (rows 2,3, cols 2,3) -> C[0:2, 2:4]
    # U rows: 2, 3, 6, 7
    u_rows = [2, 3, 6, 7]
    # V rows: B[2,2], B[2,3], B[3,2], B[3,3] -> 18, 19, 26, 27
    v_rows = [18, 19, 26, 27]
    # W rows: same as 1.3
    w_rows = [6, 7, 9, 10]
    add_strassen_block(u_rows, v_rows, w_rows, col)

    # Process A1 B2 (2x4 x 4x4)
    # B2 is cols 4,5,6,7 of B.
    
    # Block 2.1: A1L * B21 (rows 0,1, cols 4,5) -> C[0:2, 4:6]
    u_rows = [0, 1, 4, 5]
    # V rows: B[0,4], B[0,5], B[1,4], B[1,5] -> 4, 5, 12, 13
    v_rows = [4, 5, 12, 13]
    # W rows: C[0,4], C[1,4], C[0,5], C[1,5] -> 12, 13, 15, 16
    w_rows = [12, 13, 15, 16]
    add_strassen_block(u_rows, v_rows, w_rows, col)
    
    # Block 2.2: A1R * B22 (rows 2,3, cols 4,5) -> C[0:2, 4:6]
    u_rows = [2, 3, 6, 7]
    # V rows: B[2,4], B[2,5], B[3,4], B[3,5] -> 20, 21, 28, 29
    v_rows = [20, 21, 28, 29]
    w_rows = [12, 13, 15, 16]
    add_strassen_block(u_rows, v_rows, w_rows, col)
    
    # Block 2.3: A1L * B23 (rows 0,1, cols 6,7) -> C[0:2, 6:8]
    u_rows = [0, 1, 4, 5]
    # V rows: B[0,6], B[0,7], B[1,6], B[1,7] -> 6, 7, 14, 15
    v_rows = [6, 7, 14, 15]
    # W rows: C[0,6], C[1,6], C[0,7], C[1,7] -> 18, 19, 21, 22
    w_rows = [18, 19, 21, 22]
    add_strassen_block(u_rows, v_rows, w_rows, col)
    
    # Block 2.4: A1R * B24 (rows 2,3, cols 6,7) -> C[0:2, 6:8]
    u_rows = [2, 3, 6, 7]
    # V rows: B[2,6], B[2,7], B[3,6], B[3,7] -> 22, 23, 30, 31
    v_rows = [22, 23, 30, 31]
    w_rows = [18, 19, 21, 22]
    add_strassen_block(u_rows, v_rows, w_rows, col)

    # Process A2 B1 (1x4 x 4x4)
    # A2 is row 2 of A.
    # B1 is cols 0..3.
    for j in range(4):
        for k in range(4):
            U[8 + j][col] = 1
            V[j * 8 + k][col] = 1
            W[k * 3 + 2][col] = 1
            col += 1
            
    # Process A2 B2 (1x4 x 4x4)
    # B2 is cols 4..7.
    for j in range(4):
        for k in range(4):
            U[8 + j][col] = 1
            V[j * 8 + (k + 4)][col] = 1
            W[(k + 4) * 3 + 2][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
