"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    # 2x2x12 tensor = (2x2x2 matrix-mult tensor, rank 7 via Strassen) x (2x2x1 block, rank 2).
    # Product structure gives rank <= 7*2 = 14.
    # Strassen 7-term factors (standard, over integers):
    #   M0 = (a00+a11)(b00+b11)
    #   M1 = (a10+a11)*b00
    #   M2 = a00*(b01-b11)
    #   M3 = a11*(b10-b00)
    #   M4 = (a00+a01)*b11
    #   M5 = (a10-a00)*(b00+b01)
    #   M6 = (a01-a11)*(b10+b11)
    # c00=M0+M3-M4+M6 ; c01=M2+M4
    # c10=M1+M3 ;       c11=M0+M2-M1+M5
    A = [
        [1, 1, 0, 0, 0, 0, 0],  # a00
        [0, 0, 1, 0, 0, -1, 0], # a10
        [0, 1, 0, 0, 1, 1, 0],  # a11
        [0, 0, 0, 1, 1, 0, -1], # a01
    ]
    B = [
        [1, 0, 0, 1, 0, 0, 0],  # b00
        [1, 1, -1, 0, 1, 1, 1], # b01
        [0, 0, 0, 1, 0, 0, 0],  # b10
        [1, 0, 1, -1, 1, 0, 1], # b11
    ]
    C = [
        [1, 0, 0, 1, -1, 0, 1], # c00
        [0, 0, 1, 0, 1, 0, 0],  # c01
        [0, 1, 0, 1, 0, 0, 0],  # c10
        [1, -1, 1, 0, 0, 1, 0], # c11
    ]

    # 2x2x1 block (two scalar multiplications), rank 2:
    #   a00*b00 -> c00 ; a11*b11 -> c11
    A1 = [[1, 0], [0, 0], [0, 0], [0, 1]]   # a00, a10, a11, a01
    B1 = [[1, 0], [0, 0], [0, 0], [0, 1]]   # b00, b01, b10, b11
    C1 = [[1, 0], [0, 0], [0, 0], [0, 1]]   # c00, c01, c10, c11

    # Kronecker product of the two decompositions (rank 14).
    # U[i*4 + a, c] = A[i,a] * A1[a',a'']  (input index i in {0,1}, a in {0..3})
    # V[j*4 + b, c] = B[j,b] * B1[b',b'']  (j in {0,1}, b in {0..3})
    # W[k*2 + i, c] = C1[i',i''] * C[i,i'] (output index k in {0..11}, i in {0,1})
    rank = 14
    U = [[0] * rank for _ in range(8)]   # 2*4
    V = [[0] * rank for _ in range(8)]   # 2*4
    W = [[0] * rank for _ in range(24)]  # 12*2

    for c in range(rank):
        s = c // 2
        t = c % 2
        for i in range(2):
            for a in range(4):
                U[i * 4 + a][c] = A[i][a] * A1[a][t]
        for j in range(2):
            for b in range(4):
                V[j * 4 + b][c] = B[j][b] * B1[b][t]
        for i in range(2):
            for k in range(12):
                j = k // 2
                b = k % 2
                W[k * 2 + i][c] = C1[b][t] * C[i][s]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
