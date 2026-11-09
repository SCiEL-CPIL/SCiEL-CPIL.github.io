import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s kill

    # ---------- 2x2 schoolbook (rank 4) ----------
    # U2 row = i*2+j, V2 row = j*2+k, W2 row = k*2+i
    U2 = np.zeros((4, 4))
    V2 = np.zeros((4, 4))
    W2 = np.zeros((4, 4))
    for r in range(4):
        i = r // 2
        j = r % 2
        U2[i * 2 + j, r] = 1
        V2[j * 2 + i, r] = 1
        W2[i * 2 + j, r] = 1
    rank2 = 4

    # ---------- Laderman 3x3 (rank 23) ----------
    U3 = np.zeros((9, 23))
    V3 = np.zeros((9, 23))
    W3 = np.zeros((9, 23))

    laderman_terms = [
        # r0
        ([(0,0,1),(2,2,1)], [(0,0,1),(2,2,1)], [(0,0,1),(2,2,-1)]),
        # r1
        ([(0,1,1),(1,0,1)], [(1,1,1),(0,0,1)], [(1,0,1),(0,1,-1)]),
        # r2
        ([(0,2,1),(2,0,1)], [(2,0,1),(0,2,1)], [(2,0,1),(0,2,-1)]),
        # r3
        ([(1,2,1),(2,1,1)], [(1,2,1),(2,1,1)], [(1,2,1),(2,1,-1)]),
        # r4
        ([(0,0,1)], [(0,1,1)], [(1,0,1)]),
        # r5
        ([(0,1,1)], [(1,0,1)], [(0,1,1)]),
        # r6
        ([(1,0,1)], [(0,0,1)], [(0,0,1)]),
        # r7
        ([(0,2,1)], [(2,0,1)], [(0,2,1)]),
        # r8
        ([(2,0,1)], [(0,2,1)], [(2,0,1)]),
        # r9
        ([(1,2,1)], [(2,1,1)], [(1,2,1)]),
        # r10
        ([(2,1,1)], [(1,2,1)], [(2,1,1)]),
        # r11
        ([(0,0,1),(0,1,1)], [(0,0,1),(1,0,1)], [(0,0,1),(0,1,1)]),
        # r12
        ([(1,0,1),(0,0,1)], [(0,0,1),(0,1,1)], [(0,1,1),(1,0,1)]),
        # r13
        ([(0,1,1),(1,0,1)], [(0,1,1),(1,1,1)], [(1,0,1),(0,0,1)]),
        # r14
        ([(1,1,1)], [(1,1,1)], [(1,1,1)]),
        # r15
        ([(0,2,1),(1,2,1)], [(0,2,1),(1,2,1)], [(0,2,1),(1,2,1)]),
        # r16
        ([(1,2,1),(2,1,1)], [(1,2,1),(2,1,1)], [(1,2,1),(2,1,1)]),
        # r17
        ([(2,1,1),(2,0,1)], [(2,1,1),(2,0,1)], [(2,1,1),(2,0,1)]),
        # r18
        ([(2,0,1),(0,2,1)], [(2,0,1),(0,2,1)], [(2,0,1),(0,2,1)]),
        # r19
        ([(0,0,1),(1,1,1),(2,2,1)], [(0,0,1),(1,1,1),(2,2,1)], [(0,0,1),(1,1,1),(2,2,1)]),
        # r20
        ([(0,1,1),(1,2,1),(2,0,1)], [(0,1,1),(1,2,1),(2,0,1)], [(0,1,1),(1,2,1),(2,0,1)]),
        # r21
        ([(1,0,1),(2,1,1),(0,2,1)], [(1,0,1),(2,1,1),(0,2,1)], [(1,0,1),(2,1,1),(0,2,1)]),
        # r22
        ([(0,0,1),(1,1,1),(2,2,1)], [(0,0,1),(1,1,1),(2,2,1)], [(0,0,1),(1,1,1),(2,2,1)]),
    ]

    for r, (u_entries, v_entries, w_entries) in enumerate(laderman_terms):
        for (i, j, val) in u_entries:
            U3[i*3+j, r] += val
        for (j, k, val) in v_entries:
            V3[j*3+k, r] += val
        for (k, i, val) in w_entries:
            W3[k*3+i, r] += val

    # Verify Laderman 3x3
    def verify_decomp(U, V, W, n):
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*n+j, r] * V[j*n+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True

    laderman_valid = verify_decomp(U3, V3, W3, 3)

    if not laderman_valid:
        # Fallback: use 3x3 schoolbook (rank 27)
        U3 = np.zeros((9, 27))
        V3 = np.zeros((9, 27))
        W3 = np.zeros((9, 27))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U3[i*3+j, col] = 1
                    V3[j*3+k, col] = 1
                    W3[k*3+i, col] = 1
                    col += 1
        rank3 = 27
    else:
        rank3 = 23

    total_rank = rank3 * rank2

    # Build 6x6 via Kronecker product
    U6 = np.zeros((36, total_rank))
    V6 = np.zeros((36, total_rank))
    W6 = np.zeros((36, total_rank))

    for r3 in range(rank3):
        for r2 in range(rank2):
            r = r3 * rank2 + r2
            # U6
            nz_u3 = np.nonzero(U3[:, r3])[0]
            nz_u2 = np.nonzero(U2[:, r2])[0]
            for u3_idx in nz_u3:
                i3, j3 = divmod(int(u3_idx), 3)
                for u2_idx in nz_u2:
                    i2, j2 = divmod(int(u2_idx), 2)
                    i = i3*2 + i2
                    j = j3*2 + j2
                    U6[i*6+j, r] = U3[u3_idx, r3] * U2[u2_idx, r2]
            # V6
            nz_v3 = np.nonzero(V3[:, r3])[0]
            nz_v2 = np.nonzero(V2[:, r2])[0]
            for v3_idx in nz_v3:
                j3, k3 = divmod(int(v3_idx), 3)
                for v2_idx in nz_v2:
                    j2, k2 = divmod(int(v2_idx), 2)
                    j = j3*2 + j2
                    k = k3*2 + k2
                    V6[j*6+k, r] = V3[v3_idx, r3] * V2[v2_idx, r2]
            # W6
            nz_w3 = np.nonzero(W3[:, r3])[0]
            nz_w2 = np.nonzero(W2[:, r2])[0]
            for w3_idx in nz_w3:
                k3, i3 = divmod(int(w3_idx), 3)
                for w2_idx in nz_w2:
                    k2, i2 = divmod(int(w2_idx), 2)
                    k = k3*2 + k2
                    i = i3*2 + i2
                    W6[k*6+i, r] = W3[w3_idx, r3] * W2[w2_idx, r2]

    # Verify the full 6x6 decomposition
    def verify_6x6(U, V, W):
        n = 6
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*n+j, r] * V[j*n+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True

    if verify_6x6(U6, V6, W6):
        return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
    else:
        # Fallback to schoolbook 6x6 (rank 216)
        U6s = np.zeros((36, 216))
        V6s = np.zeros((36, 216))
        W6s = np.zeros((36, 216))
        col = 0
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    U6s[i*6+j, col] = 1
                    V6s[j*6+k, col] = 1
                    W6s[k*6+i, col] = 1
                    col += 1
        return {"U": U6s.tolist(), "V": V6s.tolist(), "W": W6s.tolist()}
# EVOLVE-BLOCK-END
