"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    # Matrix multiplication 2x2 * 2x4 -> 2x4
    # A is 2x2, B is 2x4, C = A*B is 2x4
    #
    # Tensor convention:
    #   U: shape (n*m, rank) = (4, rank), indexed by (i*m+j, c)
    #   V: shape (m*p, rank) = (8, rank), indexed by (j*p+k, c)
    #   W: shape (p*n, rank) = (8, rank), indexed by (k*n+i, c)
    #
    # For C[i,k] = sum_{j} A[i,j] * B[j,k]:
    #   The (i,k) entry gets contributions from columns c where
    #   U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for each j.
    #
    # We use Strassen's 7-multiplication scheme for the two 2x2 blocks
    # (B left half columns 0,1 and B right half columns 2,3).
    #
    # A = [[a00, a01], [a10, a11]]
    # B_left = [[b00, b01], [b10, b11]]  (columns 0,1)
    # B_right = [[b02, b03], [b12, b13]] (columns 2,3)
    #
    # C_left = A * B_left:
    #   C[0,0] = a00*b00 + a01*b10
    #   C[0,1] = a00*b01 + a01*b11
    #   C[1,0] = a10*b00 + a11*b10
    #   C[1,1] = a10*b01 + a11*b11
    #
    # Strassen for 2x2*2x2:
    #   M1 = (a00+a11)*(b00+b11)
    #   M2 = (a10+a11)*b00
    #   M3 = a00*(b10-b00)
    #   M4 = a11*(b01-b11)
    #   M5 = (a00+a01)*b11
    #   M6 = (a10-a00)*(b00+b01)
    #   M7 = (a01-a11)*(b10+b11)
    #
    # C[0,0] = M1 - M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M6 - M3
    #
    # Similarly for B_right with b02,b03,b12,b13.
    #
    # Rank = 14 (7 for left + 7 for right)
    #
    # U indices: i*m+j -> i*2+j
    #   U[0] = a00, U[1] = a01, U[2] = a10, U[3] = a11
    #
    # V indices: j*p+k -> j*4+k
    #   V[0]=b00, V[1]=b01, V[2]=b02, V[3]=b03
    #   V[4]=b10, V[5]=b11, V[6]=b12, V[7]=b13
    #
    # W indices: k*n+i -> k*2+i
    #   W[0]=C[0,0], W[1]=C[1,0]
    #   W[2]=C[0,1], W[3]=C[1,1]
    #   W[4]=C[0,2], W[5]=C[1,2]
    #   W[6]=C[0,3], W[7]=C[1,3]

    rank = 14
    U = [[0.0] * rank for _ in range(4)]
    V = [[0.0] * rank for _ in range(8)]
    W = [[0.0] * rank for _ in range(8)]

    # --- Block 1: left half (columns 0,1) ---
    # M1 = (a00+a11)*(b00+b11)
    U[0][0] = 1.0; U[3][0] = 1.0
    V[0][0] = 1.0; V[5][0] = 1.0
    W[0][0] = 1.0; W[3][0] = 1.0  # C00 += 1, C11 += 1

    # M2 = (a10+a11)*b00
    U[2][1] = 1.0; U[3][1] = 1.0
    V[0][1] = 1.0
    W[1][1] = 1.0; W[3][1] = -1.0  # C10 += 1, C11 -= 1

    # M3 = a00*(b10-b00)
    U[0][2] = 1.0
    V[4][2] = 1.0; V[0][2] = -1.0
    W[0][2] = -1.0; W[2][2] = 1.0  # C00 -= 1, C01 += 1

    # M4 = a11*(b01-b11)
    U[3][3] = 1.0
    V[1][3] = 1.0; V[5][3] = -1.0
    W[0][3] = -1.0; W[1][3] = 1.0  # C00 -= 1, C10 += 1

    # M5 = (a00+a01)*b11
    U[0][4] = 1.0; U[1][4] = 1.0
    V[5][4] = 1.0
    W[0][4] = -1.0; W[2][4] = 1.0  # C00 -= 1, C01 += 1

    # M6 = (a10-a00)*(b00+b01)
    U[2][5] = 1.0; U[0][5] = -1.0
    V[0][5] = 1.0; V[1][5] = 1.0
    W[3][5] = 1.0  # C11 += 1

    # M7 = (a01-a11)*(b10+b11)
    U[1][6] = 1.0; U[3][6] = -1.0
    V[4][6] = 1.0; V[5][6] = 1.0
    W[0][6] = 1.0  # C00 += 1

    # --- Block 2: right half (columns 2,3) ---
    # M8 = (a00+a11)*(b02+b13)
    U[0][7] = 1.0; U[3][7] = 1.0
    V[2][7] = 1.0; V[7][7] = 1.0
    W[4][7] = 1.0; W[7][7] = 1.0  # C02 += 1, C13 += 1

    # M9 = (a10+a11)*b02
    U[2][8] = 1.0; U[3][8] = 1.0
    V[2][8] = 1.0
    W[5][8] = 1.0; W[7][8] = -1.0  # C12 += 1, C13 -= 1

    # M10 = a00*(b12-b02)
    U[0][9] = 1.0
    V[6][9] = 1.0; V[2][9] = -1.0
    W[4][9] = -1.0; W[6][9] = 1.0  # C02 -= 1, C03 += 1

    # M11 = a11*(b03-b13)
    U[3][10] = 1.0
    V[3][10] = 1.0; V[7][10] = -1.0
    W[4][10] = -1.0; W[5][10] = 1.0  # C02 -= 1, C12 += 1

    # M12 = (a00+a01)*b13
    U[0][11] = 1.0; U[1][11] = 1.0
    V[7][11] = 1.0
    W[4][11] = -1.0; W[6][11] = 1.0  # C02 -= 1, C03 += 1

    # M13 = (a10-a00)*(b02+b03)
    U[2][12] = 1.0; U[0][12] = -1.0
    V[2][12] = 1.0; V[3][12] = 1.0
    W[7][12] = 1.0  # C13 += 1

    # M14 = (a01-a11)*(b12+b13)
    U[1][13] = 1.0; U[3][13] = -1.0
    V[6][13] = 1.0; V[7][13] = 1.0
    W[4][13] = 1.0  # C02 += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
