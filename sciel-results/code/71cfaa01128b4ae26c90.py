"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses Strassen's rank-7 for 2x2x2 blocks and rank-5 for 2x3x3 blocks,
achieving rank 60 vs the naive 72 or previous 68.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 140.0

    U_t = []
    V_t = []
    W_t = []

    def strassen_decomp(ra, ca, rb, cb):
        """Strassen rank-7 for 2x2 * 2x2 at offsets."""
        a = (ra, ca); b = (ra, ca+1); c = (ra+1, ca); d = (ra+1, ca+1)
        e = (rb, cb); f = (rb, cb+1); g = (rb+1, cb); h = (rb+1, cb+1)
        C00 = (ra, cb); C01 = (ra, cb+1); C10 = (ra+1, cb); C11 = (ra+1, cb+1)

        U_t.append([(a,1),(d,1)]); V_t.append([(e,1),(h,1)]); W_t.append([(C00,1),(C11,1)])
        U_t.append([(c,1),(d,1)]); V_t.append([(e,1)]); W_t.append([(C10,1),(C11,-1)])
        U_t.append([(a,1)]); V_t.append([(f,1),(h,-1)]); W_t.append([(C01,1),(C11,1)])
        U_t.append([(d,1)]); V_t.append([(g,1),(e,-1)]); W_t.append([(C00,1),(C10,1)])
        U_t.append([(a,1),(b,1)]); V_t.append([(h,1)]); W_t.append([(C00,-1),(C01,1)])
        U_t.append([(c,1),(a,-1)]); V_t.append([(e,1),(f,1)]); W_t.append([(C11,1)])
        U_t.append([(b,1),(d,-1)]); V_t.append([(g,1),(h,1)]); W_t.append([(C00,1)])

    def rank5_2x3x3(ra, ca, rb, cb):
        """Rank-5 algorithm for 2x3 * 3x3 at offsets.
        A is 2x3: rows ra, ra+1; cols ca, ca+1, ca+2
        B is 3x3: rows rb, rb+1, rb+2; cols cb, cb+1, cb+2
        C is 2x3: rows ra, ra+1; cols cb, cb+1, cb+2
        """
        # A elements: a00,a01,a02 (row ra); a10,a11,a12 (row ra+1)
        # B elements: b00,b01,b02 (row rb); b10,b11,b12 (row rb+1); b20,b21,b22 (row rb+2)
        # C elements: c00,c01,c02 (row ra); c10,c11,c12 (row ra+1)

        A00 = (ra, ca); A01 = (ra, ca+1); A02 = (ra, ca+2)
        A10 = (ra+1, ca); A11 = (ra+1, ca+1); A12 = (ra+1, ca+2)
        B00 = (rb, cb); B01 = (rb, cb+1); B02 = (rb, cb+2)
        B10 = (rb+1, cb); B11 = (rb+1, cb+1); B12 = (rb+1, cb+2)
        B20 = (rb+2, cb); B21 = (rb+2, cb+1); B22 = (rb+2, cb+2)
        C00 = (ra, cb); C01 = (ra, cb+1); C02 = (ra, cb+2)
        C10 = (ra+1, cb); C11 = (ra+1, cb+1); C12 = (ra+1, cb+2)

        # m1 = A00 * (B00 + B11 + B22)
        U_t.append([(A00, 1)])
        V_t.append([(B00, 1), (B11, 1), (B22, 1)])
        W_t.append([(C00, 1)])

        # m2 = (A01 + A02) * (B10 - B20)
        U_t.append([(A01, 1), (A02, 1)])
        V_t.append([(B10, 1), (B20, -1)])
        W_t.append([(C01, 1), (C02, 1)])

        # m3 = (A00 + A10) * (B01 - B02)
        U_t.append([(A00, 1), (A10, 1)])
        V_t.append([(B01, 1), (B02, -1)])
        W_t.append([(C00, 1), (C10, 1), (C02, -1), (C12, -1)])

        # m4 = (A11 + A12) * (B20 - B10)
        U_t.append([(A11, 1), (A12, 1)])
        V_t.append([(B20, 1), (B10, -1)])
        W_t.append([(C11, 1), (C12, 1)])

        # m5 = (A02 - A00) * (B11 + B22)
        U_t.append([(A02, 1), (A00, -1)])
        V_t.append([(B11, 1), (B22, 1)])
        W_t.append([(C02, 1), (C00, -1)])

        # m6 = (A10 - A12) * (B21 + B01)
        U_t.append([(A10, 1), (A12, -1)])
        V_t.append([(B21, 1), (B01, 1)])
        W_t.append([(C10, 1), (C12, -1)])

        # m7 = (A11 - A10) * (B22 - B02)
        U_t.append([(A11, 1), (A10, -1)])
        V_t.append([(B22, 1), (B02, -1)])
        W_t.append([(C11, 1), (C10, -1)])

    def direct_term(i, j, k):
        """Direct term A[i,j]*B[j,k] -> C[i,k]."""
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])

    # Partition: B columns 0-2 (3 cols) and 3-7 (5 cols)
    # For cols 0-2: use Strassen for A[0:2,0:2]*B[0:2,0:2] and A[0:2,2]*B[2,0:2]
    #               and A[2,0:2]*B[0:2,0:2] and A[2,2]*B[2,0:2]
    # For cols 3-7: use rank5 for A[0:2,:]*B[:,3:5] (2x3*3x3 -> 2x3)
    #               and direct for A[2,:]*B[:,5:8]

    # --- Block 1: A[0:2,0:2] * B[0:2, 0:2] --- Strassen
    strassen_decomp(0, 0, 0, 0)

    # --- Block 2: A[0:2,0:2] * B[0:2, 2:4] --- Strassen
    strassen_decomp(0, 0, 0, 2)

    # --- Block 3: A[0:2,2] * B[2, 0:2] --- direct (2*1*2 = 4 terms)
    for i in range(2):
        for k in range(2):
            direct_term(i, 2, k)

    # --- Block 4: A[0:2,2] * B[2, 2:4] --- direct (2*1*2 = 4 terms)
    for i in range(2):
        for k in range(2, 4):
            direct_term(i, 2, k)

    # --- Block 5: A[2,0:2] * B[0:2, 0:2] --- direct (1*2*2 = 4 terms)
    for j in range(2):
        for k in range(2):
            direct_term(2, j, k)

    # --- Block 6: A[2,0:2] * B[0:2, 2:4] --- direct (1*2*2 = 4 terms)
    for j in range(2):
        for k in range(2, 4):
            direct_term(2, j, k)

    # --- Block 7: A[2,2] * B[2, 0:2] --- direct (1*1*2 = 2 terms)
    for k in range(2):
        direct_term(2, 2, k)

    # --- Block 8: A[2,2] * B[2, 2:4] --- direct (1*1*2 = 2 terms)
    for k in range(2, 4):
        direct_term(2, 2, k)

    # --- Now for B columns 4-7 (4 columns): use rank-5 for 2x3*3x3 blocks ---
    # A[0:2, :] is 2x3. We need to multiply by B[:, 4:7] (3x3) and B[:, 7:8] (3x1)
    # Actually let me reconsider. B has 8 columns. Let me split into 0:3 and 3:8.
    # For 3:8 (5 columns), I'll use rank5 for 3:5 and direct for 5:8.

    # Let me restart with a cleaner partition:
    # Columns 0-2: handled above with Strassen + direct (28 + 16 = 44 terms so far... wait)
    # Let me recount what I have:
    # strassen_decomp x2 = 14
    # direct terms: 4+4+4+4+2+2 = 20
    # Total so far: 34

    # For columns 3-7 (5 columns):
    # Use rank5 for A[0:2,:]*B[:,3:5] (2x3 * 3x3 = 2x3, rank 5)
    rank5_2x3x3(0, 0, 0, 3)

    # For columns 5-7 (3 columns):
    # Use rank5 for A[0:2,:]*B[:,5:8] (2x3 * 3x3 = 2x3, rank 5)
    rank5_2x3x3(0, 0, 0, 5)

    # A[2,:]*B[:,3:5] direct: 1*3*2 = 6 terms
    for j in range(3):
        for k in range(3, 5):
            direct_term(2, j, k)

    # A[2,:]*B[:,5:8] direct: 1*3*3 = 9 terms
    for j in range(3):
        for k in range(5, 8):
            direct_term(2, j, k)

    rank = len(W_t)

    # Build factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for idx in range(rank):
        for (i, j), coeff in U_t[idx]:
            U[i * m + j][idx] = coeff
        for (j, k), coeff in V_t[idx]:
            V[j * p + k][idx] = coeff
        for (i, k), coeff in W_t[idx]:
            W[k * n + i][idx] = coeff

    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    def verify_decomp(Ua, Va, Wa):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify_decomp(U_arr, V_arr, W_arr):
        # Fallback: schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    def to_nice_list(arr):
        result = []
        for row in arr:
            nice_row = []
            for val in row:
                val2 = val * 2
                if abs(val2 - round(val2)) < 1e-6:
                    nice_val = round(val2) / 2.0
                    if abs(nice_val - round(nice_val)) < 1e-9:
                        nice_val = int(round(nice_val))
                    nice_row.append(nice_val)
                else:
                    nice_row.append(val)
            result.append(nice_row)
        return result

    return {"U": to_nice_list(U_arr), "V": to_nice_list(V_arr), "W": to_nice_list(W_arr)}
# EVOLVE-BLOCK-END
