import numpy as np
import time

def construct():
    t0 = time.time()
    DEADLINE = 120  # seconds

    n, m, p = 4, 4, 6  # A: n x m, B: m x p, C: n x p
    # Output C index convention: C[k*n + i] where C = A @ B
    # A index: a[i*m + j], B index: b[j*p + k], C index: c[k*n + i]

    # Tensor T[i*m+j, j'*p+k', k''*n+i''] = 1 if i==i'' and j==j' and k==k''
    # (standard matrix mult tensor)

    # We will build rank-77 decomposition via:
    # Level 1: Strassen on 2x2 blocks: 7 products of (2x2) x (2x3)
    # Level 2: For each 2x2 x 2x3, use rank-11 (Strassen-7 for first 2 cols + schoolbook-4 for 3rd col)
    # Total: 7 * 11 = 77

    rank = 77
    U = np.zeros((n * m, rank))  # 16 x 77
    V = np.zeros((m * p, rank))  # 24 x 77
    W = np.zeros((p * n, rank))  # 24 x 77

    # --- Build rank-7 Strassen for 2x2 x 2x2 ---
    # A: 2x2 (indices 0..3: i*2+j), B: 2x2 (indices 0..3: j*2+k), C: 2x2 (indices 0..3: k*2+i)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))

    # M1 = (a00+a11)(b00+b11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1  # C00 += M1, C11 += M1

    # M2 = (a10+a11)b00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1  # C10 += M2, C11 -= M2

    # M3 = a00(b01-b10)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[2, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1  # C01 += M3, C11 += M3

    # M4 = a11(b10-b00)
    U2[3, 3] = 1
    V2[0, 3] = -1; V2[2, 3] = 1
    W2[0, 3] = -1; W2[2, 3] = 1  # C00 -= M4, C10 += M4

    # M5 = (a00+a01)b11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1  # C00 -= M5, C01 += M5

    # M6 = (a10+a11)(b01+b11)
    U2[2, 5] = 1; U2[3, 5] = 1
    V2[1, 5] = 1; V2[3, 5] = 1
    W2[3, 5] = 1  # C11 += M6

    # M7 = (a00-a10)(b00+b01)
    U2[0, 6] = 1; U2[2, 6] = -1
    V2[0, 6] = 1; V2[1, 6] = 1
    W2[0, 6] = -1  # C00 -= M7

    # Verify: sum_r U2[:,r] * V2[:,r] * W2[:,r]^T should equal the 2x2x2 mult tensor
    def tensor222(Um, Vm, Wm):
        # T[a, b, c] = sum_r Um[a,r] * Vm[b,r] * Wm[c,r]
        # a = i*2+j, b = j'*2+k', c = k''*2+i''
        T = np.zeros((4, 4, 4))
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    T[a, b, c] = np.sum(Um[a, :] * Vm[b, :] * Wm[c, :])
        return T

    T222 = tensor222(U2, V2, W2)
    # Check: T222[i*2+j, j'*2+k', k''*2+i''] = 1 if i==i'', j==j', k==k''
    for i in range(2):
        for j in range(2):
            for k in range(2):
                a = i * 2 + j
                b = j * 2 + k
                c = k * 2 + i
                assert abs(T222[a, b, c] - 1.0) < 1e-10, f"222 tensor mismatch at ({i},{j},{k}): {T222[a,b,c]}"
                for i2 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            if i2 != i or j2 != j or k2 != k:
                                a2 = i2 * 2 + j2
                                b2 = j2 * 2 + k2
                                c2 = k2 * 2 + i2
                                assert abs(T222[a2, b2, c2]) < 1e-10, f"222 tensor nonzero at ({i2},{j2},{k2}): {T222[a2,b2,c2]}"

    # --- Build rank-11 for 2x2 x 2x3 ---
    # A: 2x2 (4 entries: i*2+j), B: 2x3 (6 entries: j*3+k), C: 2x3 (6 entries: k*3+i)
    U23 = np.zeros((4, 11))
    V23 = np.zeros((6, 11))
    W23 = np.zeros((6, 11))

    # Part 1: Strassen on first 2 columns of B (B[:, 0:2] is 2x2), gives C[:, 0:2] (2x2)
    for r in range(7):
        U23[:, r] = U2[:, r]
        # Map V2 (2x2, index j*2+k') to V23 (2x3, index j*3+k)
        # B2[j,k'] -> B23[j, k'] for k' in {0,1}
        for j in range(2):
            for kp in range(2):
                V23[j * 3 + kp, r] = V2[j * 2 + kp, r]
        # Map W2 (2x2, index k''*2+i') to W23 (2x3, index k''*3+i')
        # C2[k'',i'] -> C23[k'', i'] for i' in {0,1}
        for k in range(2):
            for ip in range(2):
                W23[k * 3 + ip, r] = W2[k * 2 + ip, r]

    # Part 2: Schoolbook for column 2 of B (B[:, 2])
    # C0,2 = A00 * B02 + A01 * B12
    # C1,2 = A10 * B02 + A11 * B12
    # B indices: j*3+k. B02 = j=0,k=2 -> idx 2. B12 = j=1,k=2 -> idx 5.
    # C indices: k*3+i. C02 = k=0,i=2 -> idx 2. C12 = k=1,i=2 -> idx 5.
    # A indices: i*2+j. A00=0, A01=1, A10=2, A11=3.

    r0 = 7
    U23[0, r0] = 1; V23[2, r0] = 1; W23[2, r0] = 1  # A00 * B02 -> C02
    U23[1, r0 + 1] = 1; V23[5, r0 + 1] = 1; W23[2, r0 + 1] = 1  # A01 * B12 -> C02
    U23[2, r0 + 2] = 1; V23[2, r0 + 2] = 1; W23[5, r0 + 2] = 1  # A10 * B02 -> C12
    U23[3, r0 + 3] = 1; V23[5, r0 + 3] = 1; W23[5, r0 + 3] = 1  # A11 * B12 -> C12

    # Verify 2x2x2x3 tensor
    def tensor223(Um, Vm, Wm):
        # T[a, b, c] = sum_r Um[a,r] * Vm[b,r] * Wm[c,r]
        # a = i*2+j (A 2x2), b = j'*3+k' (B 2x3), c = k''*3+i'' (C 2x3)
        T = np.zeros((4, 6, 6))
        for a in range(4):
            for b in range(6):
                for c in range(6):
                    T[a, b, c] = np.sum(Um[a, :] * Vm[b, :] * Wm[c, :])
        return T

    T223 = tensor223(U23, V23, W23)
    # Check: T223[i*2+j, j'*3+k', k''*3+i''] = 1 if i==i'', j==j', k==k''
    for i in range(2):
        for j in range(2):
            for k in range(3):
                a = i * 2 + j
                b = j * 3 + k
                c = k * 3 + i
                assert abs(T223[a, b, c] - 1.0) < 1e-10, f"223 tensor mismatch at ({i},{j},{k}): {T223[a,b,c]}"

    # --- Build 4x4 x 4x6 via Strassen on 2x2 blocks ---
    # A is 4x4: blocks A00, A01, A10, A11 (each 2x2)
    #   A[i,j] with i in [0,4), j in [0,4)
    #   Block (bi, bj): rows bi*2..bi*2+1, cols bj*2..bj*2+1
    # B is 4x6: blocks B00, B01, B10, B11 (each 2x3)
    #   B[j,k] with j in [0,4), k in [0,6)
    #   Block (bj, bk): rows bj*2..bj*2+1, cols bk*3..bk*3+2
    # C is 4x6: blocks C00, C01, C10, C11 (each 2x3)
    #   C[i,k] with i in [0,4), k in [0,6)
    #   Block (ci, ck): rows ci*2..ci*2+1, cols ck*3..ck*3+2
    #   C index: k*4+i (output convention)

    # Strassen steps: (A_comb, B_comb, C_comb)
    # Each comb is dict mapping block (bi, bj) or (bj, bk) or (ci, ck) to coefficient
    steps = [
        # M1 = (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1}),
        # M2 = (A10+A11)B00
        ({(1,0): 1, (1,1): 1},
         {(0,0): 1},
         {(1,0): 1, (1,1): -1}),
        # M3 = A00(B01-B11)
        ({(0,0): 1},
         {(0,1): 1, (1,1): -1},
         {(0,1): 1, (1,1): 1}),
        # M4 = A11(B10-B00)
        ({(1,1): 1},
         {(1,0): 1, (0,0): -1},
         {(0,0): -1, (1,0): 1}),
        # M5 = (A00+A01)B11
        ({(0,0): 1, (0,1): 1},
         {(1,1): 1},
         {(0,0): -1, (0,1): 1}),
        # M6 = (A10+A11)(B01+B11)
        ({(1,0): 1, (1,1): 1},
         {(0,1): 1, (1,1): 1},
         {(1,1): 1}),
        # M7 = (A00-A10)(B00+B01)
        ({(0,0): 1, (1,0): -1},
         {(0,0): 1, (0,1): 1},
         {(0,0): -1}),
    ]

    for s_idx, (a_comb, b_comb, c_comb) in enumerate(steps):
        base_r = s_idx * 11
        for r_local in range(11):
            r_global = base_r + r_local

            # Fill U
            for (bi, bj), coeff in a_comb.items():
                for li in range(2):
                    for lj in range(2):
                        u_val = U23[li * 2 + lj, r_local]
                        if u_val != 0:
                            gi = bi * 2 + li
                            gj = bj * 2 + lj
                            g_idx = gi * 4 + gj  # A index: i*4+j
                            U[g_idx, r_global] += coeff * u_val

            # Fill V
            for (bj, bk), coeff in b_comb.items():
                for lj in range(2):
                    for lk in range(3):
                        v_val = V23[lj * 3 + lk, r_local]
                        if v_val != 0:
                            gj = bj * 2 + lj
                            gk = bk * 3 + lk
                            g_idx = gj * 6 + gk  # B index: j*6+k
                            V[g_idx, r_global] += coeff * v_val

            # Fill W
            for (ci, ck), coeff in c_comb.items():
                for li in range(2):
                    for lk in range(3):
                        w_val = W23[lk * 3 + li, r_local]
                        if w_val != 0:
                            gi = ci * 2 + li
                            gk = ck * 3 + lk
                            g_idx = gk * 4 + gi  # C index: k*4+i
                            W[g_idx, r_global] += coeff * w_val

    # Final verification: check the full 4x4x4x6 tensor
    # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Should equal 1 when i==i'', j==j', k==k'' and 0 otherwise
    # a = i*4+j, b = j'*6+k', c = k''*4+i''
    print(f"Verification started at t={time.time()-t0:.2f}s")
    ok = True
    for i in range(4):
        for j in range(4):
            for k in range(6):
                a = i * 4 + j
                b = j * 6 + k
                c = k * 4 + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-10:
                    print(f"ERROR: T[{a},{b},{c}] = {val}, expected 1")
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    if ok:
        # Check a few zero entries
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    for i2 in [x for x in range(4) if x != i]:
                        a = i * 4 + j
                        b = j * 6 + k
                        c = k * 4 + i2
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        if abs(val) > 1e-10:
                            print(f"ERROR: T[{a},{b},{c}] = {val}, expected 0")
                            ok = False
    print(f"Verification finished at t={time.time()-t0:.2f}s, ok={ok}")

    if not ok:
        raise ValueError("Tensor decomposition verification failed")

    # Convert to int arrays for cleaner output
    U = np.round(U).astype(int)
    V = np.round(V).astype(int)
    W = np.round(W).astype(int)

    return {"U": U, "V": V, "W": W}
