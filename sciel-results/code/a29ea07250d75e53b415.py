import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s limit to be safe

    # Strassen decomposition for 2x2x2: rank 7
    # Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # The tensor T[i,j,k] = sum_c U[i*2+j,c] * V[j*2+k,c] * W[k*2+i,c] = 1 if (i,j,k) valid
    #
    # Strassen's 7 products:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B11)
    # M3 = A11(B01-B10)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    #
    # C00 = M0 + M2 - M4 + M6
    # C01 = M1 + M4
    # C10 = M3 + M5
    # C11 = M0 - M1 + M3 + M5

    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # M0: U gets A00, A11; V gets B00, B11; W gets C00, C11
    U2[0, 0] = 1  # i=0,j=0
    U2[3, 0] = 1  # i=1,j=1
    V2[0, 0] = 1  # j=0,k=0
    V2[3, 0] = 1  # j=1,k=1
    W2[0, 0] = 1  # k=0,i=0 -> C00
    W2[3, 0] = 1  # k=1,i=1 -> C11

    # M1: U gets A10, A11; V gets B00; W gets C01, C11
    U2[2, 1] = 1  # i=1,j=0
    U2[3, 1] = 1  # i=1,j=1
    V2[0, 1] = 1  # j=0,k=0
    W2[1, 1] = 1  # k=0,i=1 -> C01
    W2[3, 1] = -1  # k=1,i=1 -> C11

    # M2: U gets A00; V gets B10, B11; W gets C00, C11
    U2[0, 2] = 1  # i=0,j=0
    V2[1, 2] = 1  # j=1,k=0
    V2[3, 2] = -1  # j=1,k=1
    W2[2, 2] = 1  # k=1,i=0 -> C10... wait
    # C00 = M0 + M2 - M4 + M6, C11 = M0 - M1 + M3 + M5
    # M2 contributes to C00 (coefficient +1) and... let me re-derive
    # Actually: C00 = M0 + M2 - M4 + M6
    #            C01 = M1 + M4
    #            C10 = M3 + M5
    #            C11 = M0 - M1 + M3 + M5
    # So M2 only appears in C00.
    W2[0, 2] = 1  # k=0,i=0 -> C00

    # M3: U gets A11; V gets B01, B10; W gets C10, C11
    U2[3, 3] = 1  # i=1,j=1
    V2[2, 3] = 1  # j=0,k=1
    V2[1, 3] = -1  # j=1,k=0
    W2[2, 3] = 1  # k=1,i=0 -> C10
    W2[3, 3] = 1  # k=1,i=1 -> C11

    # M4: U gets A00, A01; V gets B11; W gets C00, C01
    U2[0, 4] = 1  # i=0,j=0
    U2[1, 4] = 1  # i=0,j=1
    V2[3, 4] = 1  # j=1,k=1
    W2[0, 4] = -1  # k=0,i=0 -> C00
    W2[1, 4] = 1  # k=0,i=1 -> C01

    # M5: U gets A10, A00; V gets B00, B01; W gets C10, C11
    U2[2, 5] = 1  # i=1,j=0
    U2[0, 5] = -1  # i=0,j=0
    V2[0, 5] = 1  # j=0,k=0
    V2[2, 5] = 1  # j=0,k=1
    W2[2, 5] = 1  # k=1,i=0 -> C10
    W2[3, 5] = 1  # k=1,i=1 -> C11

    # M6: U gets A01, A11; V gets B10, B11; W gets C00
    U2[1, 6] = 1  # i=0,j=1
    U2[3, 6] = -1  # i=1,j=1
    V2[1, 6] = 1  # j=1,k=0
    V2[3, 6] = 1  # j=1,k=1
    W2[0, 6] = 1  # k=0,i=0 -> C00

    # Verify 2x2 Strassen
    valid2 = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(7):
                    s += U2[i*2+j, c] * V2[j*2+k, c] * W2[k*2+i, c]
                if s != 1:
                    valid2 = False
                    break
            if not valid2:
                break
        if not valid2:
            break

    if not valid2:
        # Fallback to schoolbook rank-64
        n, m, p = 4, 4, 4
        rank = 64
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Now construct 4x4x4 via Kronecker product of the two 2x2 Strassen decompositions
    # The 4x4 matrix multiplication tensor is the tensor product of two 2x2x2 tensors
    # T4[i,j,k] = T2[i1,j1,k1] * T2[i2,j2,k2] where i=i1*2+i2, j=j1*2+j2, k=k1*2+k2
    #
    # The Kronecker product decomposition:
    # U4[i*4+j, c1*7+c2] = U2[i1*2+j1, c1] * U2[i2*2+j2, c2]
    # V4[j*4+k, c1*7+c2] = V2[j1*2+k1, c1] * V2[j2*2+k2, c2]
    # W4[k*4+i, c1*7+c2] = W2[k1*2+i1, c1] * W2[k2*2+i2, c2]

    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    for i in range(n):
        i1 = i // 2
        i2 = i % 2
        for j in range(m):
            j1 = j // 2
            j2 = j % 2
            u_row = i * m + j
            U[u_row, :] = np.kron(U2[i1 * 2 + j1, :], U2[i2 * 2 + j2, :])

    for j in range(m):
        j1 = j // 2
        j2 = j % 2
        for k in range(p):
            k1 = k // 2
            k2 = k % 2
            v_row = j * p + k
            V[v_row, :] = np.kron(V2[j1 * 2 + k1, :], V2[j2 * 2 + k2, :])

    for k in range(p):
        k1 = k // 2
        k2 = k % 2
        for i in range(n):
            i1 = i // 2
            i2 = i % 2
            w_row = k * n + i
            W[w_row, :] = np.kron(W2[k1 * 2 + i1, :], W2[k2 * 2 + i2, :])

    # Verify the 4x4x4 decomposition
    valid4 = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0
                for c in range(49):
                    s += U[i*4+j, c] * V[j*4+k, c] * W[k*4+i, c]
                if s != 1:
                    valid4 = False
                    break
            if not valid4:
                break
        if not valid4:
            break

    if not valid4:
        # Fallback to schoolbook rank-64
        rank = 64
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Check time
    if time.time() > deadline:
        # Should not happen, but safety
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
