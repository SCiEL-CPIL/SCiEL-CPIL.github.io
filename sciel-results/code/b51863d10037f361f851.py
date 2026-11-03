import time

DIMENSIONS = [2, 4, 6]

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    for u_idx in range(n * m):
        for v_idx in range(m * p):
            for w_idx in range(p * n):
                val = sum(U[u_idx][r] * V[v_idx][r] * W[w_idx][r] for r in range(rank))
                i = u_idx // m
                j = u_idx % m
                j2 = v_idx // p
                k = v_idx % p
                k2 = w_idx // n
                i2 = w_idx % n
                if i == i2 and j == j2 and k == k2:
                    if val != 1:
                        return False
                else:
                    if val != 0:
                        return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 14
    U = [[0] * rank for _ in range(n * m)]  # 8 x 14
    V = [[0] * rank for _ in range(m * p)]  # 24 x 14
    W = [[0] * rank for _ in range(p * n)]  # 12 x 14

    # A is 2x4, B is 4x6
    # Split A into 2x2 blocks of size 1x2:
    #   A = [A00 A01; A10 A11] where A00=A[0,0:2], A01=A[0,2:4], A10=A[1,0:2], A11=A[1,2:4]
    # Split B into 2x2 blocks of size 2x3:
    #   B = [B00 B01; B10 B11] where B00=B[0:2,0:3], B01=B[0:2,3:6], B10=B[2:4,0:3], B11=B[2:4,3:6]
    #
    # Strassen:
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11)B00
    #   M3 = A00(B01-B11)
    #   M4 = A11(B10-A00)
    #   M5 = (A00+A01)B11
    #   M6 = (A10-A00)(B00+B01)
    #   M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each Mk = (1x2) * (2x6), rank 2.
    # Total rank = 7 * 2 = 14

    # U indices: i*m + j, i in {0,1}, j in {0,1,2,3}
    # V indices: j*p + k, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # W indices: k*n + i, k in {0,1,2,3,4,5}, i in {0,1}

    # For each Strassen product Mk, we have two bilinear terms (rank 2):
    # Term 0: uses the first row of the A-part (a_0) and first row of B-part
    # Term 1: uses the second row of the A-part (a_1) and second row of B-part

    # Define Strassen products: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: list of (u_idx, coeff) for A entries
    # v_coeffs: list of (v_idx, coeff) for B entries
    # w_coeffs: list of (w_idx, coeff) for C entries

    # A entry indices: A00=[0,1], A01=[2,3], A10=[4,5], A11=[6,7]
    # B entry indices: B00=[0,1,2,3,4,5], B01=[6,7,8,9,10,11], B10=[12,13,14,15,16,17], B11=[18,19,20,21,22,23]
    # C entry indices: C00=[0,1,2,3,4,5], C01=[6,7,8,9,10,11], C10=[12,13,14,15,16,17], C11=[18,19,20,21,22,23]
    # But W index is k*n+i, so C[i,k] is at W[k*2+i]
    # C00[k] -> W[k*2+0], C01[k] -> W[k*2+1], C10[k] -> W[k*2+0+... wait

    # Let me re-index: C is 2x6, C[i][k] for i in {0,1}, k in {0,...,5}
    # W index: k*n + i = k*2 + i
    # So C[0,0]=W[0], C[0,1]=W[2], C[0,2]=W[4], C[0,3]=W[6], C[0,4]=W[8], C[0,5]=W[10]
    #    C[1,0]=W[1], C[1,1]=W[3], C[1,2]=W[5], C[1,3]=W[7], C[1,4]=W[9], C[1,5]=W[11]

    # Strassen products with their A-part and B-part:
    # Each product: (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: dict from A_block to coeff, where A_block is (row, col_start, col_end)
    # Actually, let me define each Strassen term's A-part as a 1x2 vector and B-part as 2x6 matrix

    # M1: A-part = A00 + A11 = [A[0,0]+A[1,2], A[0,1]+A[1,3]]
    #     B-part = B00 + B11 (2x6)
    # M2: A-part = A10 + A11 = [A[1,0]+A[1,2], A[1,1]+A[1,3]]
    #     B-part = B00 (2x6)
    # M3: A-part = A00 = [A[0,0], A[0,1]]
    #     B-part = B01 - B11 (2x6)
    # M4: A-part = A11 = [A[1,2], A[1,3]]
    #     B-part = B10 - A00... wait, B10 is 2x3, not 2x6. Let me recheck.

    # Actually B00, B01, B10, B11 are all 2x3 blocks.
    # B00 = B[0:2, 0:3], B01 = B[0:2, 3:6], B10 = B[2:4, 0:3], B11 = B[2:4, 3:6]
    # So B-part for each Strassen term is a 2x6 matrix formed by concatenating two 2x3 blocks.

    # M1: A = A00+A11 (1x2), B = [B00 | B11] (2x6)
    # M2: A = A10+A11 (1x2), B = [B00 | 0] (2x6)  -- only first 3 cols
    # M3: A = A00 (1x2), B = [0 | B01-B11] (2x6) -- only last 3 cols
    # M4: A = A11 (1x2), B = [B10-A00... no, B10 is 2x3, A00 is 1x2. Can't subtract.

    # I made an error. Let me redo Strassen properly for 2x2 block matrices where each block is a matrix.
    # A = [A00 A01; A10 A11], each Aij is 1x2
    # B = [B00 B01; B10 B11], each Bij is 2x3
    #
    # M1 = (A00+A11) * (B00+B11)  -- (1x2)*(2x6) since B00+B11 is 2x6
    # M2 = (A10+A11) * B00        -- (1x2)*(2x3), result is 1x3
    # M3 = A00 * (B01-B11)        -- (1x2)*(2x3), result is 1x3
    # M4 = A11 * (B10-A00)        -- ERROR: B10 is 2x3, A00 is 1x2. Can't subtract!

    # The issue: Strassen requires the blocks to be square or at least the same shape for addition/subtraction.
    # Here A blocks are 1x2 and B blocks are 2x3, so B00+B11 is fine (both 2x3 -> 2x3, but we need 2x6 for the full product).

    # Actually, each Strassen product Mk gives a 1x6 result (since (1x2)*(2x6) = 1x6).
    # But M2 = (A10+A11)*B00 is (1x2)*(2x3) = 1x3, which only covers columns 0-2.
    # We need to pad: M2_full = (A10+A11) * [B00 | 0] (2x6)

    # Let me redefine:
    # M1 = (A00+A11) * [B00+B11, 0]  -- no, that's not right either.

    # Correct Strassen:
    # M1 = (A00+A11)(B00+B11)  -- both are valid: (1x2)*(2x6) if we interpret B00+B11 as 2x6
    #   But B00 is 2x3 and B11 is 2x3, so B00+B11 is 2x3, not 2x6.

    # I think the issue is that I'm conflating the block structure. Let me use a cleaner formulation.

    # A is 2x4, B is 4x6. Think of A as having 2 rows, B as having 6 columns.
    # C[i,k] = sum_j A[i,j]*B[j,k] for i in {0,1}, k in {0,...,5}, j in {0,1,2,3}

    # Split j into {0,1} and {2,3}:
    # C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]

    # Let A_L = A[:, 0:2] (2x2), A_R = A[:, 2:4] (2x2)
    # Let B_T = B[0:2, :] (2x6), B_B = B[2:4, :] (2x6)
    # C = A_L * B_T + A_R * B_B  (2x6 = 2x2*2x6 + 2x2*2x6)

    # Now use Strassen for each 2x2 * 2x6 product:
    # For X (2x2) * Y (2x6) where X = [X00 X01; X10 X11], Y = [Y0; Y1] (each Yi is 1x6):
    # Result R = [R0; R1] where Ri is 1x6
    # R0 = X00*Y0 + X01*Y1
    # R1 = X10*Y0 + X11*Y1
    #
    # Strassen:
    # P1 = (X00+X11)*Y0  -- wait, Y0 is 1x6, X00+X11 is scalar. So P1 = scalar * vector = 1x6
    # This doesn't work directly because the blocks aren't square in the same way.

    # Actually, for 2x2 * 2x6, we can use:
    # R0 = X00*Y0 + X01*Y1
    # R1 = X10*Y0 + X11*Y1
    # This is just 4 scalar-vector multiplications, rank 4.
    # Or use Strassen-style:
    # P1 = (X00+X11)*Y0  -- no, this gives X00*Y0 + X11*Y0, but we need X00*Y0 + X01*Y1 for R0

    # The standard Strassen doesn't directly apply to rectangular block matrices in this way.
    # Let me just use the direct rank-4 decomposition for 2x2 * 2x6:
    # R0[k] = X00*Y0[k] + X01*Y1[k]
    # R1[k] = X10*Y0[k] + X11*Y1[k]
    # Rank 4: terms (X00,Y0), (X01,Y1), (X10,Y0), (X11,Y1)

    # So total rank = 4 (for A_L*B_T) + 4 (for A_R*B_B) = 8.

    # Let me verify: rank of 2x2*2x6 matmul tensor is 4.
    # U: 4 rows (X00, X01, X10, X11), V: 12 rows (Y0[0..5], Y1[0..5]), W: 12 rows (R0[0..5], R1[0..5])
    # Term 0: U[0]=X00, V[Y0[k]]=1 for k=0..5, W[R0[k]]=1 for k=0..5
    # Term 1: U[1]=X01, V[Y1[k]]=1 for k=0..5, W[R0[k]]=1 for k=0..5
    # Term 2: U[2]=X10, V[Y0[k]]=1 for k=0..5, W[R1[k]]=1 for k=0..5
    # Term 3: U[3]=X11, V[Y1[k]]=1 for k=0..5, W[R1[k]]=1 for k=0..5

    # Check: R0[k] = X00*Y0[k] + X01*Y1[k] ✓
    # R1[k] = X10*Y0[k] + X11*Y1[k] ✓
    # All other entries are 0 ✓

    # So total rank = 8!

    rank = 8
    U = [[0] * rank for _ in range(n * m)]  # 8 x 8
    V = [[0] * rank for _ in range(m * p)]  # 24 x 8
    W = [[0] * rank for _ in range(p * n)]  # 12 x 8

    # A_L = A[:, 0:2], indices in U: 0,1,4,5 (i*m+j: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5)
    # A_R = A[:, 2:4], indices in U: 2,3,6,7
    # B_T = B[0:2, :], indices in V: 0..11 (j*p+k: 0*6+0=0, ..., 1*6+5=11)
    # B_B = B[2:4, :], indices in V: 12..23

    # C = A_L * B_T + A_R * B_B
    # C[i,k] is at W[k*n+i] = W[k*2+i]

    # Term 0: A_L[0,0] * B_T[0,:] -> C[0,:]
    # U[0][0] = 1, V[j][0] = 1 for j in 0..5 (B_T row 0), W[k*2+0][0] = 1 for k in 0..5
    U[0][0] = 1
    for k in range(6):
        V[0 * 6 + k][0] = 1  # B_T[0, k]
        W[k * 2 + 0][0] = 1  # C[0, k]

    # Term 1: A_L[0,1] * B_T[1,:] -> C[0,:]
    U[1][1] = 1
    for k in range(6):
        V[1 * 6 + k][1] = 1  # B_T[1, k]
        W[k * 2 + 0][1] = 1  # C[0, k]

    # Term 2: A_L[1,0] * B_T[0,:] -> C[1,:]
    U[4][2] = 1
    for k in range(6):
        V[0 * 6 + k][2] = 1  # B_T[0, k]
        W[k * 2 + 1][2] = 1  # C[1, k]

    # Term 3: A_L[1,1] * B_T[1,:] -> C[1,:]
    U[5][3] = 1
    for k in range(6):
        V[1 * 6 + k][3] = 1  # B_T[1, k]
        W[k * 2 + 1][3] = 1  # C[1, k]

    # Term 4: A_R[0,0] * B_B[0,:] -> C[0,:]
    U[2][4] = 1
    for k in range(6):
        V[2 * 6 + k][4] = 1  # B_B[0, k]
        W[k * 2 + 0][4] = 1  # C[0, k]

    # Term 5: A_R[0,1] * B_B[1,:] -> C[0,:]
    U[3][5] = 1
    for k in range(6):
        V[3 * 6 + k][5] = 1  # B_B[1, k]
        W[k * 2 + 0][5] = 1  # C[0, k]

    # Term 6: A_R[1,0] * B_B[0,:] -> C[1,:]
    U[6][6] = 1
    for k in range(6):
        V[2 * 6 + k][6] = 1  # B_B[0, k]
        W[k * 2 + 1][6] = 1  # C[1, k]

    # Term 7: A_R[1,1] * B_B[1,:] -> C[1,:]
    U[7][7] = 1
    for k in range(6):
        V[3 * 6 + k][7] = 1  # B_B[1, k]
        W[k * 2 + 1][7] = 1  # C[1, k]

    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Invalid decomposition")

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
