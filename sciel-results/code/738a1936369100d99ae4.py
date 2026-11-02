import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS

    def verify(U, V, W):
        # Check: for all (i,j,k), sum_c U[i*m+j, c]*V[j*p+k, c]*W[k*n+i, c] == 1
        # T[i,j,k] = 1 always (since C[i,k] = sum_j A[i,j] B[j,k] for all i,j,k in range)
        # Actually T[i,j,k] = 1 for all valid i,j,k because every pair contributes.
        # Wait: C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]. So T[i,j,k]=1 for all i,j,k.
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        # T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # For matrix mult, T[i,j,k] = 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Strassen 2x2x2 decomposition (rank 7)
    # A = [[A00,A01],[A10,A11]], B = [[B00,B01],[B10,B11]], C = AB
    # Products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 - M3 + M6
    #
    # U2[row, col]: coefficient of A[row//2, row%2] in product col
    # V2[row, col]: coefficient of B[row//2, row%2] in product col
    # W2[row, col]: coefficient of product col in C[row//2, row%2]

    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]

    # Product 0: (A00+A11)(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1  # C00 += M1, C11 += M1

    # Product 1: (A10+A11)B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1  # C10 += M2

    # Product 2: A00(B01-B11)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1  # C01 += M3

    # Product 3: A11(B10-B00)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = -1; W2[2][3] = 1  # C00 -= M4, C10 += M4

    # Product 4: (A00+A01)B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1  # C00 -= M5, C01 += M5

    # Product 5: (A10-A00)(B00+B01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1  # C11 += M6

    # Product 6: (A01-A11)(B10+B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1  # C00 += M7

    # Verify 2x2x2
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = sum(U2[i*2+j][c] * V2[j*2+k][c] * W2[k*2+i][c] for c in range(7))
                if abs(s - 1) > 1e-9:
                    print(f"2x2x2 error at ({i},{j},{k}): {s}")
                    return None

    # Build 4x4x4 Strassen (rank 49)
    # A is 4x4, split into 2x2 blocks of 2x2:
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # Block (bi,bj) occupies rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # A[i,j] = A_block[i//2, j//2][i%2, j%2]
    #
    # For each Strassen product p (0..6) and each 2x2x2 sub-product c (0..6):
    #   The A-part: A2[i%2, j%2] in block (ai, aj) where (ai,aj) determined by product p
    #   The B-part: B2[k%2, l%2] in block (bk, bl) where (bk,bl) determined by product p
    #   The C-part: C2[m%2, n%2] in block (ci, cj) where (ci,cj) determined by sub-product c

    # For Strassen product p, the A-blocks involved:
    # p=0: A00+A11 -> blocks (0,0) and (1,1)
    # p=1: A10+A11 -> blocks (1,0) and (1,1)
    # p=2: A00 -> block (0,0)
    # p=3: A11 -> block (1,1)
    # p=4: A00+A01 -> blocks (0,0) and (0,1)
    # p=5: A10-A00 -> blocks (1,0) and (0,0)
    # p=6: A01-A11 -> blocks (0,1) and (1,1)

    # A-blocks for product p (list of (block_row, block_col, sign)):
    A_blocks = [
        [(0,0,1), (1,1,1)],   # p=0
        [(1,0,1), (1,1,1)],   # p=1
        [(0,0,1)],            # p=2
        [(1,1,1)],            # p=3
        [(0,0,1), (0,1,1)],   # p=4
        [(1,0,1), (0,0,-1)],  # p=5
        [(0,1,1), (1,1,-1)],  # p=6
    ]

    # B-blocks for product p:
    B_blocks = [
        [(0,0,1), (1,1,1)],   # p=0
        [(0,0,1)],            # p=1
        [(0,1,1), (1,1,-1)],  # p=2
        [(1,0,1), (0,0,-1)],  # p=3
        [(1,1,1)],            # p=4
        [(0,0,1), (0,1,1)],   # p=5
        [(1,0,1), (1,1,1)],   # p=6
    ]

    # C-blocks for sub-product c (the 2x2x2 result block position):
    # c=0: C00 -> block (0,0)
    # c=1: C01 -> block (0,1)
    # c=2: C10 -> block (1,0)
    # c=3: C11 -> block (1,1)
    C_blocks = [(0,0), (0,1), (1,0), (1,1)]

    rank_444 = 49
    U4 = [[0]*rank_444 for _ in range(16)]
    V4 = [[0]*rank_444 for _ in range(16)]
    W4 = [[0]*rank_444 for _ in range(16)]

    for p in range(7):
        for c in range(7):
            col = p * 7 + c
            # U part
            a_coeff = U2[c * 2 + 0][c]  # Wait, I need to be more careful
            # Actually: for sub-product c, the 2x2 A-matrix entries are:
            # A2[ai, aj] where ai = row//2, aj = row%2 in the 2x2 local index
            # U2[row][c] gives coefficient of A[row//2, row%2] in product c
            # But we need to map to the 4x4 global index
            # A_global[i,j] where i = 2*block_row + local_row, j = 2*block_col + local_col

            for local_i in range(2):
                for local_j in range(2):
                    coeff = U2[local_i * 2 + local_j][c]
                    if coeff == 0:
                        continue
                    for (br, bc, sign) in A_blocks[p]:
                        global_i = 2 * br + local_i
                        global_j = 2 * bc + local_j
                        U4[global_i * 4 + global_j][col] += sign * coeff

            for local_j in range(2):
                for local_k in range(2):
                    coeff = V2[local_j * 2 + local_k][c]
                    if coeff == 0:
                        continue
                    for (br, bc, sign) in B_blocks[p]:
                        global_j = 2 * br + local_j
                        global_k = 2 * bc + local_k
                        V4[global_j * 4 + global_k][col] += sign * coeff

            for local_k in range(2):
                for local_i in range(2):
                    coeff = W2[local_k * 2 + local_i][c]
                    if coeff == 0:
                        continue
                    cr, cc = C_blocks[c]
                    global_k = 2 * cr + local_k
                    global_i = 2 * cc + local_i
                    W4[global_k * 4 + global_i][col] += coeff

    # Verify 4x4x4
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = sum(U4[i*4+j][c] * V4[j*4+k][c] * W4[k*4+i][c] for c in range(49))
                if abs(s - 1) > 1e-9:
                    print(f"4x4x4 error at ({i},{j},{k}): {s}")
                    return None

    # Build full 4x4x7: rank = 49 + 48 = 97
    rank = 97
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    # Part 1: Strassen 4x4x4 (columns 0..48)
    for i in range(4):
        for j in range(4):
            for c in range(49):
                U[i*4+j][c] = U4[i*4+j][c]
            for k in range(4):
                V[j*4+k][c] = V4[j*4+k][c]
                W[k*4+i][c] = W4[k*4+i][c]

    # Part 2: Schoolbook 4x4x3 (columns 49..96)
    col = 49
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[i*4+j][col] = 1
                V[j*4+k][col] = 1
                W[k*4+i][col] = 1
                col += 1

    # Final verification
    if not verify(U, V, W):
        print("Verification failed")
        return None

    return {"U": U, "V": V, "W": W}
