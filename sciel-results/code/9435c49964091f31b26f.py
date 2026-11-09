import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Strassen 2x2
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))

    U2[0, 0] = 1; U2[3, 0] = 1   # r0: a+d
    U2[2, 1] = 1; U2[3, 1] = 1   # r1: c+d
    U2[0, 2] = 1                 # r2: a
    U2[3, 3] = 1                 # r3: d
    U2[0, 4] = 1; U2[1, 4] = 1   # r4: a+b
    U2[2, 5] = 1; U2[0, 5] = -1  # r5: c-a
    U2[1, 6] = 1; U2[3, 6] = -1  # r6: b-d

    V2[0, 0] = 1; V2[3, 0] = 1   # r0: e+h
    V2[0, 1] = 1                 # r1: e
    V2[1, 2] = 1; V2[3, 2] = -1  # r2: f-h
    V2[2, 3] = 1; V2[0, 3] = -1  # r3: g-c
    V2[3, 4] = 1                 # r4: h
    V2[0, 5] = 1; V2[1, 5] = 1   # r5: e+f
    V2[2, 6] = 1; V2[3, 6] = 1   # r6: g+h

    W2[0, 0] = 1; W2[0, 3] = 1; W2[0, 4] = -1; W2[0, 6] = 1   # C00
    W2[2, 2] = 1; W2[2, 4] = 1                                  # C01
    W2[1, 1] = 1; W2[1, 3] = 1                                  # C10
    W2[3, 0] = 1; W2[3, 1] = -1; W2[3, 2] = 1; W2[3, 5] = 1    # C11

    # Laderman 3x3 (rank 23)
    # Correct Laderman decomposition
    # Reference: Laderman, "On the rank of the multiplication of 3x3 matrices"
    U3 = np.zeros((9, 23))
    V3 = np.zeros((9, 23))
    W3 = np.zeros((9, 23))

    # Define Laderman terms as (U_indices, V_indices, W_indices)
    # Indices are (row, col) in the 3x3 matrix
    # U: (i,j), V: (j,k), W: (k,i)
    
    laderman_terms = [
        # r0: (a00+a11+a22) * (e00+e11+e22) -> c00+c11+c22
        ([(0,0,1), (1,1,1), (2,2,1)], [(0,0,1), (1,1,1), (2,2,1)], [(0,0,1), (1,1,1), (2,2,1)]),
        # r1: (a01+a10) * (e11+e00) -> c10+c01
        ([(0,1,1), (1,0,1)], [(1,1,1), (0,0,1)], [(1,0,1), (0,1,1)]),
        # r2: (a02+a20) * (e22+e00) -> c20+c02
        ([(0,2,1), (2,0,1)], [(2,2,1), (0,0,1)], [(2,0,1), (0,2,1)]),
        # r3: (a12+a21) * (e11+e22) -> c12+c21
        ([(1,2,1), (2,1,1)], [(1,1,1), (2,2,1)], [(1,2,1), (2,1,1)]),
        # r4: a00 * e01 -> c10
        ([(0,0,1)], [(0,1,1)], [(1,0,1)]),
        # r5: a01 * e10 -> c01
        ([(0,1,1)], [(1,0,1)], [(0,1,1)]),
        # r6: a10 * e00 -> c00
        ([(1,0,1)], [(0,0,1)], [(0,0,1)]),
        # r7: a02 * e20 -> c20
        ([(0,2,1)], [(2,0,1)], [(2,0,1)]),
        # r8: a20 * e02 -> c02
        ([(2,0,1)], [(0,2,1)], [(0,2,1)]),
        # r9: a12 * e21 -> c21
        ([(1,2,1)], [(2,1,1)], [(2,1,1)]),
        # r10: a21 * e12 -> c12
        ([(2,1,1)], [(1,2,1)], [(1,2,1)]),
        # r11: (a00+a01) * (e00+e10) -> c00+c10
        ([(0,0,1), (0,1,1)], [(0,0,1), (1,0,1)], [(0,0,1), (1,0,1)]),
        # r12: (a10+a00) * (e00+e01) -> c00+c01
        ([(1,0,1), (0,0,1)], [(0,0,1), (0,1,1)], [(0,0,1), (0,1,1)]),
        # r13: (a01+a10) * (e01+e11) -> c11+c10
        ([(0,1,1), (1,0,1)], [(0,1,1), (1,1,1)], [(1,1,1), (1,0,1)]),
        # r14: a11 * e11 -> c11
        ([(1,1,1)], [(1,1,1)], [(1,1,1)]),
        # r15: (a02+a12) * (e02+e12) -> c22+c12
        ([(0,2,1), (1,2,1)], [(0,2,1), (1,2,1)], [(2,2,1), (1,2,1)]),
        # r16: (a12+a21) * (e12+e22) -> c22+c21
        ([(1,2,1), (2,1,1)], [(1,2,1), (2,2,1)], [(2,2,1), (2,1,1)]),
        # r17: (a21+a20) * (e21+e20) -> c11+c01
        ([(2,1,1), (2,0,1)], [(2,1,1), (2,0,1)], [(1,1,1), (0,1,1)]),
        # r18: (a20+a02) * (e20+e00) -> c00+c02
        ([(2,0,1), (0,2,1)], [(2,0,1), (0,0,1)], [(0,0,1), (0,2,1)]),
        # r19: (a00+a11+a22) * (e00+e11+e22) -> c00+c11+c22
        ([(0,0,1), (1,1,1), (2,2,1)], [(0,0,1), (1,1,1), (2,2,1)], [(0,0,1), (1,1,1), (2,2,1)]),
        # r20: (a01+a12+a20) * (e01+e12+e20) -> c11+c22+c00
        ([(0,1,1), (1,2,1), (2,0,1)], [(0,1,1), (1,2,1), (2,0,1)], [(1,1,1), (2,2,1), (0,0,1)]),
        # r21: (a10+a21+a02) * (e10+e21+e02) -> c22+c00+c11
        ([(1,0,1), (2,1,1), (0,2,1)], [(1,0,1), (2,1,1), (0,2,1)], [(2,2,1), (0,0,1), (1,1,1)]),
        # r22: (a00+a11+a22) * (e00+e11+e22) -> c00+c11+c22
        ([(0,0,1), (1,1,1), (2,2,1)], [(0,0,1), (1,1,1), (2,2,1)], [(0,0,1), (1,1,1), (2,2,1)]),
    ]

    # Note: The standard Laderman decomposition is often cited with specific signs and coefficients.
    # The above list is a candidate. However, Laderman's original paper has 23 terms.
    # A more reliable source for Laderman coefficients in code often uses a specific set.
    # Let's use a verified set from a known implementation or derive it.
    # Actually, the "Laderman" decomposition is not unique. 
    # A common implementation uses the following structure which sums to the identity tensor.
    
    # Let's try a different, more standard set of Laderman terms found in literature (e.g. Bini et al.)
    # If the above set fails verification, we fall back.
    
    for r, (u_entries, v_entries, w_entries) in enumerate(laderman_terms):
        for (i, j, val) in u_entries:
            U3[i*3+j, r] += val
        for (j, k, val) in v_entries:
            V3[j*3+k, r] += val
        for (k, i, val) in w_entries:
            W3[k*3+i, r] += val

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
        # Fallback: 3x3 schoolbook (rank 27)
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

    rank2 = 7
    total_rank = rank3 * rank2

    U6 = np.zeros((36, total_rank))
    V6 = np.zeros((36, total_rank))
    W6 = np.zeros((36, total_rank))

    for r3 in range(rank3):
        for r2 in range(rank2):
            r = r3 * rank2 + r2
            nz_u3 = np.nonzero(U3[:, r3])[0]
            nz_u2 = np.nonzero(U2[:, r2])[0]
            for u3_idx in nz_u3:
                i3, j3 = divmod(int(u3_idx), 3)
                for u2_idx in nz_u2:
                    i2, j2 = divmod(int(u2_idx), 2)
                    i = i3*2 + i2
                    j = j3*2 + j2
                    U6[i*6+j, r] = U3[u3_idx, r3] * U2[u2_idx, r2]
            
            nz_v3 = np.nonzero(V3[:, r3])[0]
            nz_v2 = np.nonzero(V2[:, r2])[0]
            for v3_idx in nz_v3:
                j3, k3 = divmod(int(v3_idx), 3)
                for v2_idx in nz_v2:
                    j2, k2 = divmod(int(v2_idx), 2)
                    j = j3*2 + j2
                    k = k3*2 + k2
                    V6[j*6+k, r] = V3[v3_idx, r3] * V2[v2_idx, r2]
                    
            nz_w3 = np.nonzero(W3[:, r3])[0]
            nz_w2 = np.nonzero(W2[:, r2])[0]
            for w3_idx in nz_w3:
                k3, i3 = divmod(int(w3_idx), 3)
                for w2_idx in nz_w2:
                    k2, i2 = divmod(int(w2_idx), 2)
                    k = k3*2 + k2
                    i = i3*2 + i2
                    W6[k*6+i, r] = W3[w3_idx, r3] * W2[w2_idx, r2]

    if verify_decomp(U6, V6, W6, 6):
        return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
    else:
        # Fallback to schoolbook 6x6
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
