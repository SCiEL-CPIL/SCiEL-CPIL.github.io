import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    def verify_decomposition(U, V, W, n, m, p):
        """Verify: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(U.shape[1]):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if total != 1:
                        return False
        return True

    # Strassen 2x2 rank-7
    def get_strassen_2x2():
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
        U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
        U[3,3]=1; V[0,3]=-1; V[2,3]=1; W[0,3]=1; W[1,3]=1
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
        U[0,5]=-1; U[2,5]=1; V[0,5]=1; V[1,5]=1; W[3,5]=1
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
        return U, V, W

    # 3x3 rank-19 decomposition (Bini et al. 1980 style)
    def get_3x3_rank19():
        rank = 19
        U = np.zeros((9, rank), dtype=int)
        V = np.zeros((9, rank), dtype=int)
        W = np.zeros((9, rank), dtype=int)

        # Row/col indexing: U[i,j] -> row i*3+j, V[j,k] -> row j*3+k, W[k,i] -> row k*3+i
        # 9 diagonal terms
        for i in range(3):
            for j in range(3):
                r = i * 3 + j
                U[i*3+j, r] = 1
                V[j*3+j, r] = 1
                W[j*3+i, r] = 1

        # 6 off-diagonal terms
        # r=9: (a01+a11)(b10+b00)
        U[1,9]=1; U[4,9]=1
        V[0,9]=1; V[3,9]=1
        W[0,9]=1; W[3,9]=1

        # r=10: (a02+a20)(b20+b02)
        U[2,10]=1; U[6,10]=1
        V[0,10]=1; V[6,10]=1
        W[0,10]=1; W[6,10]=1

        # r=11: (a12+a21)(b21+b12)
        U[5,11]=1; U[7,11]=1
        V[3,11]=1; V[5,11]=1
        W[3,11]=1; W[5,11]=1

        # r=12: (a01-a11)(b10-b00)
        U[1,12]=1; U[4,12]=-1
        V[3,12]=1; V[0,12]=-1
        W[0,12]=1; W[3,12]=-1

        # r=13: (a02-a20)(b20-b02)
        U[2,13]=1; U[6,13]=-1
        V[6,13]=1; V[0,13]=-1
        W[0,13]=1; W[6,13]=-1

        # r=14: (a12-a21)(b21-b12)
        U[5,14]=1; U[7,14]=-1
        V[5,14]=1; V[3,14]=-1
        W[3,14]=1; W[5,14]=-1

        # r=15: (a01+a11)(b10-b00)
        U[1,15]=1; U[4,15]=1
        V[3,15]=1; V[0,15]=-1
        W[0,15]=1; W[3,15]=-1

        # r=16: (a02+a20)(b20-b02)
        U[2,16]=1; U[6,16]=1
        V[6,16]=1; V[0,16]=-1
        W[0,16]=1; W[6,16]=-1

        # r=17: (a12+a21)(b21-b12)
        U[5,17]=1; U[7,17]=1
        V[5,17]=1; V[3,17]=-1
        W[3,17]=1; W[5,17]=-1

        # r=18: (a01-a11)(b10+b00)
        U[1,18]=1; U[4,18]=-1
        V[0,18]=1; V[3,18]=1
        W[0,18]=1; W[3,18]=1

        if verify_decomposition(U, V, W, 3, 3, 3):
            return U, V, W
        return None

    # Schoolbook 3x3 (guaranteed correct)
    def get_schoolbook_3x3():
        U = np.zeros((9, 27), dtype=int)
        V = np.zeros((9, 27), dtype=int)
        W = np.zeros((9, 27), dtype=int)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    r = i * 9 + j * 3 + k
                    U[i*3+j, r] = 1
                    V[j*3+k, r] = 1
                    W[k*3+i, r] = 1
        return U, V, W

    U2, V2, W2 = get_strassen_2x2()

    # Try rank-19 3x3, fallback to schoolbook 27
    if time.time() > deadline - 5:
        U3, V3, W3 = get_schoolbook_3x3()
        rank3 = 27
    else:
        r19 = get_3x3_rank19()
        if r19 is not None:
            U3, V3, W3 = r19
            rank3 = 19
        else:
            U3, V3, W3 = get_schoolbook_3x3()
            rank3 = 27

    # Combine 2x2 (Strassen) and 3x3 via block Kronecker
    rank = 7 * rank3
    U6 = np.zeros((36, rank), dtype=int)
    V6 = np.zeros((36, rank), dtype=int)
    W6 = np.zeros((36, rank), dtype=int)

    for r2 in range(7):
        for r3 in range(rank3):
            r = r2 * rank3 + r3

            nz_u2 = np.nonzero(U2[:, r2])[0]
            nz_v2 = np.nonzero(V2[:, r2])[0]
            nz_w2 = np.nonzero(W2[:, r2])[0]
            nz_u3 = np.nonzero(U3[:, r3])[0]
            nz_v3 = np.nonzero(V3[:, r3])[0]
            nz_w3 = np.nonzero(W3[:, r3])[0]

            for u2 in nz_u2:
                i2, j2 = u2 // 2, u2 % 2
                val_u2 = U2[u2, r2]
                for u3 in nz_u3:
                    i3, j3 = u3 // 3, u3 % 3
                    val_u3 = U3[u3, r3]
                    i6, j6 = i2*3+i3, j2*3+j3
                    U6[i6*6+j6, r] += val_u2 * val_u3

            for v2 in nz_v2:
                j2, k2 = v2 // 2, v2 % 2
                val_v2 = V2[v2, r2]
                for v3 in nz_v3:
                    j3, k3 = v3 // 3, v3 % 3
                    val_v3 = V3[v3, r3]
                    j6, k6 = j2*3+j3, k2*3+k3
                    V6[j6*6+k6, r] += val_v2 * val_v3

            for w2 in nz_w2:
                k2, i2 = w2 // 2, w2 % 2
                val_w2 = W2[w2, r2]
                for w3 in nz_w3:
                    k3, i3 = w3 // 3, w3 % 3
                    val_w3 = W3[w3, r3]
                    k6, i6 = k2*3+k3, i2*3+i3
                    W6[k6*6+i6, r] += val_w2 * val_w3

    # Verify final result
    if verify_decomposition(U6, V6, W6, 6, 6, 6):
        return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}

    # Fallback: pure schoolbook 6x6 (rank 216)
    U6s = np.zeros((36, 216), dtype=int)
    V6s = np.zeros((36, 216), dtype=int)
    W6s = np.zeros((36, 216), dtype=int)
    for i in range(6):
        for j in range(6):
            for k in range(6):
                r = i * 36 + j * 6 + k
                U6s[i*6+j, r] = 1
                V6s[j*6+k, r] = 1
                W6s[k*6+i, r] = 1
    return {"U": U6s.tolist(), "V": V6s.tolist(), "W": W6s.tolist()}
# EVOLVE-BLOCK-END
