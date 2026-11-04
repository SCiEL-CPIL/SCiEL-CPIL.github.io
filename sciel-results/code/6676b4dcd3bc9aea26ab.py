import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 39
    rank = 39
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen block for 2x2 x 2x2
    # A indices: a00, a01, a10, a11
    # B indices: b00, b01, b10, b11
    # C indices: c00, c01, c10, c11
    def add_strassen(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        nonlocal col
        # m1 = (a00+a11)(b00+b11)
        U[a00][col] = 1; U[a11][col] = 1
        V[b00][col] = 1; V[b11][col] = 1
        W[c00][col] = 1; W[c01][col] = 1; W[c10][col] = 1; W[c11][col] = 1
        col += 1
        # m2 = (a10+a11)b00
        U[a10][col] = 1; U[a11][col] = 1
        V[b00][col] = 1
        W[c10][col] = 1; W[c11][col] = -1
        col += 1
        # m3 = a00(b01-b11)
        U[a00][col] = 1
        V[b01][col] = 1; V[b11][col] = -1
        W[c01][col] = 1; W[c11][col] = 1
        col += 1
        # m4 = a11(b10-b00)
        U[a11][col] = 1
        V[b10][col] = 1; V[b00][col] = -1
        W[c00][col] = 1; W[c10][col] = 1
        col += 1
        # m5 = (a00+a01)b11
        U[a00][col] = 1; U[a01][col] = 1
        V[b11][col] = 1
        W[c00][col] = -1; W[c01][col] = 1
        col += 1
        # m6 = (a10-a00)(b00+b10)
        U[a10][col] = 1; U[a00][col] = -1
        V[b00][col] = 1; V[b10][col] = 1
        W[c11][col] = 1
        col += 1
        # m7 = (a01-a11)(b01+b11)
        U[a01][col] = 1; U[a11][col] = -1
        V[b01][col] = 1; V[b11][col] = 1
        W[c00][col] = 1; W[c11][col] = 1
        col += 1

    # Helper to add Schoolbook 2x2 x 2x1
    def add_sb_221(a00, a01, a10, a11, b0, b1, c0, c1):
        nonlocal col
        # c0 = a00*b0 + a01*b1
        # c1 = a10*b0 + a11*b1
        U[a00][col] = 1
        V[b0][col] = 1
        W[c0][col] = 1
        col += 1
        U[a01][col] = 1
        V[b1][col] = 1
        W[c0][col] = 1
        col += 1
        U[a10][col] = 1
        V[b0][col] = 1
        W[c1][col] = 1
        col += 1
        U[a11][col] = 1
        V[b1][col] = 1
        W[c1][col] = 1
        col += 1

    # Helper to add Outer Product 2x1 x 1x7
    def add_op_217(a0, a1, b0, b1, b2, b3, b4, b5, b6, c0_0, c0_1, c0_2, c0_3, c0_4, c0_5, c0_6, c1_0, c1_1, c1_2, c1_3, c1_4, c1_5, c1_6):
        nonlocal col
        # c0_k = a0*b_k
        for k in range(7):
            U[a0][col] = 1
            V[[b0, b1, b2, b3, b4, b5, b6][k]][col] = 1
            W[[c0_0, c0_1, c0_2, c0_3, c0_4, c0_5, c0_6][k]][col] = 1
            col += 1
        # c1_k = a1*b_k
        for k in range(7):
            U[a1][col] = 1
            V[[b0, b1, b2, b3, b4, b5, b6][k]][col] = 1
            W[[c1_0, c1_1, c1_2, c1_3, c1_4, c1_5, c1_6][k]][col] = 1
            col += 1

    # Indices for A (2x3)
    # A[0,0]=0, A[0,1]=1, A[0,2]=2
    # A[1,0]=3, A[1,1]=4, A[1,2]=5
    # Indices for B (3x7)
    # B[j, k] = j*7 + k
    # B[0,0]=0 ... B[0,6]=6
    # B[1,0]=7 ... B[1,6]=13
    # B[2,0]=14 ... B[2,6]=20
    # Indices for C (2x7) -> W[k*n+i]
    # C[0, k] = k*2 + 0 = 2k
    # C[1, k] = k*2 + 1 = 2k+1
    
    # Block 1: A[0:2, 0:2] x B[0:2, 0:2]
    # A: 0,1,3,4. B: 0,1,7,8. C: C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3
    add_strassen(0, 1, 3, 4, 0, 1, 7, 8, 0, 2, 1, 3)
    
    # Block 2: A[0:2, 0:2] x B[0:2, 2:4]
    # B: 2,3,9,10. C: C[0,2]=4, C[0,3]=6, C[1,2]=5, C[1,3]=7
    add_strassen(0, 1, 3, 4, 2, 3, 9, 10, 4, 6, 5, 7)
    
    # Block 3: A[0:2, 0:2] x B[0:2, 4:6]
    # B: 4,5,11,12. C: C[0,4]=8, C[0,5]=10, C[1,4]=9, C[1,5]=11
    add_strassen(0, 1, 3, 4, 4, 5, 11, 12, 8, 10, 9, 11)
    
    # Block 4: A[0:2, 0:2] x B[0:2, 6:7]
    # B: 6,13. C: C[0,6]=12, C[1,6]=13
    add_sb_221(0, 1, 3, 4, 6, 13, 12, 13)
    
    # Block 5: A[0:2, 2:3] x B[2:3, 0:7]
    # A: 2,5. B: 14..20. C: all cols
    # C[0,k] = 2k, C[1,k] = 2k+1
    b_ids = [14, 15, 16, 17, 18, 19, 20]
    c0_ids = [0, 2, 4, 6, 8, 10, 12]
    c1_ids = [1, 3, 5, 7, 9, 11, 13]
    add_op_217(2, 5, *b_ids, *c0_ids, *c1_ids)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
