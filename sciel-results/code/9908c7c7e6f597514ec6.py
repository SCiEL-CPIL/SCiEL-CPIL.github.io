import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Winograd 3x3 rank-8 algorithm
    def get_winograd_3x3():
        U = np.zeros((9, 8), dtype=int)
        V = np.zeros((9, 8), dtype=int)
        W = np.zeros((9, 8), dtype=int)

        # Winograd's rank-8 algorithm for 3x3 matrix multiplication
        # M = A * B, where A and B are 3x3
        # U[i,j,r] multiplies A[i,j], V[j,k,r] multiplies B[j,k], W[k,i,r] multiplies C[i,j] contribution
        # Convention: U is (n*m, rank) indexed U[i*m+j, r]
        #              V is (m*p, rank) indexed V[j*p+k, r]
        #              W is (p*n, rank) indexed W[k*n+i, r]
        # Condition: sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1 for all i,j,k

        # p0 = (a00+a01+a02)(b00+b10+b20)
        U[0, 0] = 1; U[1, 0] = 1; U[2, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1; V[6, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1; W[6, 0] = 1
        W[1, 0] = -1; W[4, 0] = -1; W[7, 0] = -1
        W[2, 0] = -1; W[5, 0] = -1; W[8, 0] = -1

        # p1 = (a01-a02)(b00-b10)
        U[1, 1] = 1; U[2, 1] = -1
        V[0, 1] = 1; V[3, 1] = -1
        W[1, 1] = 1; W[4, 1] = 1; W[7, 1] = 1
        W[3, 1] = -1; W[6, 1] = -1; W[8, 1] = -1

        # p2 = (a02-a01)(b02-b12)
        U[1, 2] = -1; U[2, 2] = 1
        V[2, 2] = 1; V[5, 2] = -1
        W[2, 2] = 1; W[5, 2] = 1; W[8, 2] = 1
        W[1, 2] = -1; W[4, 2] = -1; W[7, 2] = -1

        # p3 = (a10+a11+a12)(b00+b10+b20)
        U[3, 3] = 1; U[4, 3] = 1; U[5, 3] = 1
        V[0, 3] = 1; V[3, 3] = 1; V[6, 3] = 1
        W[3, 3] = 1; W[6, 3] = 1; W[8, 3] = 1
        W[4, 3] = -1; W[7, 3] = -1; W[5, 3] = -1

        # p4 = (a11-a12)(b00-b10)
        U[4, 4] = 1; U[5, 4] = -1
        V[0, 4] = 1; V[3, 4] = -1
        W[4, 4] = 1; W[7, 4] = 1; W[8, 4] = 1
        W[6, 4] = -1; W[5, 4] = -1; W[3, 4] = -1

        # p5 = (a12-a11)(b02-b12)
        U[4, 5] = -1; U[5, 5] = 1
        V[2, 5] = 1; V[5, 5] = -1
        W[5, 5] = 1; W[8, 5] = 1; W[7, 5] = 1
        W[4, 5] = -1; W[6, 5] = -1; W[3, 5] = -1

        # p6 = (a20+a21+a22)(b00+b10+b20)
        U[6, 6] = 1; U[7, 6] = 1; U[8, 6] = 1
        V[0, 6] = 1; V[3, 6] = 1; V[6, 6] = 1
        W[6, 6] = 1; W[8, 6] = 1; W[5, 6] = 1
        W[7, 6] = -1; W[4, 6] = -1; W[3, 6] = -1

        # p7 = (a21-a22)(b00-b10)
        U[7, 7] = 1; U[8, 7] = -1
        V[0, 7] = 1; V[3, 7] = -1
        W[7, 7] = 1; W[8, 7] = 1; W[5, 7] = 1
        W[6, 7] = -1; W[4, 7] = -1; W[3, 7] = -1

        return U, V, W

    # Strassen 2x2 rank-7 algorithm
    def get_strassen_2x2():
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)

        # Strassen 2x2
        # m0 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1

        # m1 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[3, 1] = -1

        # m2 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1

        # m3 = A11(B10-B00)
        U[3, 3] = 1
        V[0, 3] = -1; V[2, 3] = 1
        W[0, 3] = 1; W[1, 3] = 1

        # m4 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1

        # m5 = (A10-A00)(B00+B01)
        U[0, 5] = -1; U[2, 5] = 1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1

        # m6 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1

        return U, V, W

    # Verify a decomposition: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all i,j,k
    def verify_decomposition(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(U.shape[1]):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if total != 1:
                        return False
        return True

    # Get Winograd 3x3 and verify
    U3, V3, W3 = get_winograd_3x3()
    winograd_valid = verify_decomposition(U3, V3, W3, 3, 3, 3)

    # Get Strassen 2x2 and verify
    U2, V2, W2 = get_strassen_2x2()
    strassen_valid = verify_decomposition(U2, V2, W2, 2, 2, 2)

    if winograd_valid and strassen_valid:
        # Combine: 6x6 as 2x2 blocks of 3x3
        # Rank = 7 * 8 = 56
        rank_3x3 = 8
        rank_2x2 = 7
        rank = rank_2x2 * rank_3x3

        U6 = np.zeros((36, rank), dtype=int)
        V6 = np.zeros((36, rank), dtype=int)
        W6 = np.zeros((36, rank), dtype=int)

        for r2 in range(rank_2x2):
            for r3 in range(rank_3x3):
                r = r2 * rank_3x3 + r3
                for i2 in range(2):
                    for j2 in range(2):
                        if U2[i2*2+j2, r2] != 0:
                            val_u2 = U2[i2*2+j2, r2]
                            for i3 in range(3):
                                for j3 in range(3):
                                    if U3[i3*3+j3, r3] != 0:
                                        val_u3 = U3[i3*3+j3, r3]
                                        i6 = i2*3 + i3
                                        j6 = j2*3 + j3
                                        U6[i6*6+j6, r] += val_u2 * val_u3

                for j2 in range(2):
                    for k2 in range(2):
                        if V2[j2*2+k2, r2] != 0:
                            val_v2 = V2[j2*2+k2, r2]
                            for j3 in range(3):
                                for k3 in range(3):
                                    if V3[j3*3+k3, r3] != 0:
                                        val_v3 = V3[j3*3+k3, r3]
                                        j6 = j2*3 + j3
                                        k6 = k2*3 + k3
                                        V6[j6*6+k6, r] += val_v2 * val_v3

                for k2 in range(2):
                    for i2 in range(2):
                        if W2[k2*2+i2, r2] != 0:
                            val_w2 = W2[k2*2+i2, r2]
                            for k3 in range(3):
                                for i3 in range(3):
                                    if W3[k3*3+i3, r3] != 0:
                                        val_w3 = W3[k3*3+i3, r3]
                                        k6 = k2*3 + k3
                                        i6 = i2*3 + i3
                                        W6[k6*6+i6, r] += val_w2 * val_w3

        # Final verification of the 6x6 decomposition
        if verify_decomposition(U6, V6, W6, 6, 6, 6):
            return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}

    # Fallback: Strassen 2x2 with schoolbook 3x3 (rank 7*27=189)
    U3s = np.zeros((9, 27), dtype=int)
    V3s = np.zeros((9, 27), dtype=int)
    W3s = np.zeros((9, 27), dtype=int)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r = i * 9 + j * 3 + k
                U3s[i * 3 + j, r] = 1
                V3s[j * 3 + k, r] = 1
                W3s[k * 3 + i, r] = 1

    rank = 7 * 27
    U6 = np.zeros((36, rank), dtype=int)
    V6 = np.zeros((36, rank), dtype=int)
    W6 = np.zeros((36, rank), dtype=int)

    for r2 in range(7):
        for r3 in range(27):
            r = r2 * 27 + r3
            for i2 in range(2):
                for j2 in range(2):
                    if U2[i2*2+j2, r2] != 0:
                        val_u2 = U2[i2*2+j2, r2]
                        for i3 in range(3):
                            for j3 in range(3):
                                if U3s[i3*3+j3, r3] != 0:
                                    val_u3 = U3s[i3*3+j3, r3]
                                    i6 = i2*3 + i3
                                    j6 = j2*3 + j3
                                    U6[i6*6+j6, r] += val_u2 * val_u3

            for j2 in range(2):
                for k2 in range(2):
                    if V2[j2*2+k2, r2] != 0:
                        val_v2 = V2[j2*2+k2, r2]
                        for j3 in range(3):
                            for k3 in range(3):
                                if V3s[j3*3+k3, r3] != 0:
                                    val_v3 = V3s[j3*3+k3, r3]
                                    j6 = j2*3 + j3
                                    k6 = k2*3 + k3
                                    V6[j6*6+k6, r] += val_v2 * val_v3

            for k2 in range(2):
                for i2 in range(2):
                    if W2[k2*2+i2, r2] != 0:
                        val_w2 = W2[k2*2+i2, r2]
                        for k3 in range(3):
                            for i3 in range(3):
                                if W3s[k3*3+i3, r3] != 0:
                                    val_w3 = W3s[k3*3+i3, r3]
                                    k6 = k2*3 + k3
                                    i6 = i2*3 + i3
                                    W6[k6*6+i6, r] += val_w2 * val_w3

    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
# EVOLVE-BLOCK-END
