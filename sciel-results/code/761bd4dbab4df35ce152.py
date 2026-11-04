"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Decompose 3x4x6 into 6 blocks of 3x2x2
    # A is 3x4, split into A1, A2 (3x2)
    # B is 4x6, split into B1, B2 (2x6)
    # B1 split into B11, B12, B13 (2x2)
    # B2 split into B21, B22, B23 (2x2)
    
    # Each 3x2x2 block uses rank 11
    # Total rank = 6 * 11 = 66
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 3x2x2 decomposition
    # a_indices: list of 6 indices for A (3x2)
    # b_indices: list of 4 indices for B (2x2)
    # w_indices: list of 6 indices for W (6x1, corresponding to 3x2 output)
    def add_3x2x2_block(a_idx, b_idx, w_idx):
        nonlocal col
        # Strassen 7 products for top 2x2
        # A = [[a00, a01], [a10, a11], [a20, a21]]
        # B = [[b00, b01], [b10, b11]]
        
        a00, a01, a10, a11, a20, a21 = a_idx
        b00, b01, b10, b11 = b_idx
        c00, c01, c10, c11, c20, c21 = w_idx
        
        # P1 = (a00+a01)(b00+b10)
        U[a00][col] += 1
        U[a01][col] += 1
        V[b00][col] += 1
        V[b10][col] += 1
        W[c00][col] += 1
        W[c01][col] -= 1
        W[c10][col] -= 1
        W[c11][col] += 1
        col += 1
        
        # P2 = (a00+a01)b10
        U[a00][col] += 1
        U[a01][col] += 1
        V[b10][col] += 1
        W[c00][col] -= 1
        W[c01][col] += 1
        W[c10][col] += 1
        W[c11][col] -= 1
        col += 1
        
        # P3 = a00(a11-a10)
        U[a00][col] += 1
        U[a11][col] += 1
        U[a10][col] -= 1
        V[b00][col] += 1
        W[c01][col] += 1
        W[c10][col] += 1
        col += 1
        
        # P4 = a11(b01-b00)
        U[a11][col] += 1
        V[b01][col] += 1
        V[b00][col] -= 1
        W[c00][col] += 1
        W[c01][col] += 1
        col += 1
        
        # P5 = (a00+a11)b00
        U[a00][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        W[c01][col] += 1
        W[c11][col] -= 1
        col += 1
        
        # P6 = (a10+a11)b01
        U[a10][col] += 1
        U[a11][col] += 1
        V[b01][col] += 1
        W[c10][col] += 1
        W[c11][col] += 1
        col += 1
        
        # P7 = (a10-a00)(b00+b10)
        U[a10][col] += 1
        U[a00][col] -= 1
        V[b00][col] += 1
        V[b10][col] += 1
        W[c10][col] -= 1
        W[c11][col] += 1
        col += 1
        
        # P8 = a20*b00
        U[a20][col] += 1
        V[b00][col] += 1
        W[c20][col] += 1
        col += 1
        
        # P9 = a20*b10
        U[a20][col] += 1
        V[b10][col] += 1
        W[c20][col] += 1
        col += 1
        
        # P10 = a21*b01
        U[a21][col] += 1
        V[b01][col] += 1
        W[c21][col] += 1
        col += 1
        
        # P11 = a21*b11
        U[a21][col] += 1
        V[b11][col] += 1
        W[c21][col] += 1
        col += 1

    # Indices for A (3x4)
    # A1: cols 0,1. A2: cols 2,3
    # A1 indices: (0, 1, 4, 5, 8, 9) -> row*2 + col
    # A2 indices: (2, 3, 6, 7, 10, 11)
    
    a1_idx = [0, 1, 4, 5, 8, 9]
    a2_idx = [2, 3, 6, 7, 10, 11]
    
    # Indices for B (4x6)
    # B1: rows 0,1. B2: rows 2,3
    # B11: cols 0,1 of B1 -> B indices (0, 2, 4, 6) ? 
    # B is 4x6. Index = row*6 + col.
    # B1 is rows 0,1.
    # B11 is cols 0,1.
    # b00 = B[0,0]=0, b01 = B[0,1]=1, b10 = B[1,0]=6, b11 = B[1,1]=7
    # B12 is cols 2,3.
    # b00 = B[0,2]=2, b01 = B[0,3]=3, b10 = B[1,2]=8, b11 = B[1,3]=9
    # B13 is cols 4,5.
    # b00 = B[0,4]=4, b01 = B[0,5]=5, b10 = B[1,4]=10, b11 = B[1,5]=11
    
    # B2 is rows 2,3.
    # B21 is cols 0,1.
    # b00 = B[2,0]=12, b01 = B[2,1]=13, b10 = B[3,0]=18, b11 = B[3,1]=19
    # B22 is cols 2,3.
    # b00 = B[2,2]=14, b01 = B[2,3]=15, b10 = B[3,2]=20, b11 = B[3,3]=21
    # B23 is cols 4,5.
    # b00 = B[2,4]=16, b01 = B[2,5]=17, b10 = B[3,4]=22, b11 = B[3,5]=23
    
    b11_idx = [0, 1, 6, 7]
    b12_idx = [2, 3, 8, 9]
    b13_idx = [4, 5, 10, 11]
    b21_idx = [12, 13, 18, 19]
    b22_idx = [14, 15, 20, 21]
    b23_idx = [16, 17, 22, 23]
    
    # Indices for W (6x3)
    # W[k*n+i, r] where n=3, p=6. Size 18.
    # Output C is 3x6.
    # C1 (from A1*B1) contributes to cols 0,1,2,3,4,5?
    # No, B1 is 2x6. B11 is 2x2 (cols 0,1).
    # So C11 (from A1*B11) is 3x2, contributing to C cols 0,1.
    # W indices for C11:
    # C[0,0]=0, C[0,1]=1
    # C[1,0]=3, C[1,1]=4
    # C[2,0]=6, C[2,1]=7
    # W index = k*3 + i.
    # k=0, i=0 -> 0
    # k=1, i=0 -> 3
    # k=2, i=0 -> 6
    # k=0, i=1 -> 1
    # k=1, i=1 -> 4
    # k=2, i=1 -> 7
    # Order in w_idx: c00, c01, c10, c11, c20, c21
    # c00=C[0,0], c01=C[0,1], c10=C[1,0], c11=C[1,1], c20=C[2,0], c21=C[2,1]
    
    w11_idx = [0, 1, 3, 4, 6, 7]
    w12_idx = [2, 3, 5, 6, 8, 9] # Cols 2,3 -> k=2,3? No, k is col index.
    # C cols 2,3.
    # C[0,2]=2, C[0,3]=3
    # C[1,2]=5, C[1,3]=6
    # C[2,2]=8, C[2,3]=9
    # w_idx: c00=2, c01=3, c10=5, c11=6, c20=8, c21=9
    
    w13_idx = [4, 5, 7, 8, 10, 11] # Cols 4,5
    # C[0,4]=4, C[0,5]=5
    # C[1,4]=7, C[1,5]=8
    # C[2,4]=10, C[2,5]=11
    
    # A2*B2 contributions to same C positions.
    # w21_idx same as w11_idx
    w21_idx = [0, 1, 3, 4, 6, 7]
    w22_idx = [2, 3, 5, 6, 8, 9]
    w23_idx = [4, 5, 7, 8, 10, 11]
    
    add_3x2x2_block(a1_idx, b11_idx, w11_idx)
    add_3x2x2_block(a1_idx, b12_idx, w12_idx)
    add_3x2x2_block(a1_idx, b13_idx, w13_idx)
    add_3x2x2_block(a2_idx, b21_idx, w21_idx)
    add_3x2x2_block(a2_idx, b22_idx, w22_idx)
    add_3x2x2_block(a2_idx, b23_idx, w23_idx)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
