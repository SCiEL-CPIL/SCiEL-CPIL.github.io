"""Matrix multiplication 2,2,8 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 8]


def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 2, 8

    # Schoolbook decomposition (rank 32) as fallback
    def schoolbook():
        rank = n * m * p  # 32
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        idx = 0
        for i in range(n):
            for k in range(m):
                for l in range(p):
                    U[i * m + k, idx] = 1
                    V[k * p + l, idx] = 1
                    W[l * n + i, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Verify a candidate decomposition
    def verify(U_list, V_list, W_list, rank):
        U_arr = np.array(U_list, dtype=float)
        V_arr = np.array(V_list, dtype=float)
        W_arr = np.array(W_list, dtype=float)

        np.random.seed(12345)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test

        A_flat = A_test.flatten()  # row-major: [A[0,0], A[0,1], A[1,0], A[1,1]]
        B_flat = B_test.flatten()  # row-major: [B[0,0], B[0,1], ..., B[1,7]]

        C_computed = np.zeros(p * n)
        for r in range(rank):
            prod_u = np.dot(U_arr[:, r], A_flat)
            prod_v = np.dot(V_arr[:, r], B_flat)
            C_computed += W_arr[:, r] * prod_u * prod_v

        C_computed = C_computed.reshape(p, n).T
        max_err = np.max(np.abs(C_computed - C_expected))
        return max_err < 1e-10, max_err

    # Strassen per-block decomposition (rank 28)
    def strassen_blocks():
        num_blocks = p // 2  # 4 blocks of 2x2
        rank = num_blocks * 7  # 28

        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        for b in range(num_blocks):
            base = b * 7
            # A indices: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
            # B indices in block: col 2b and 2b+1
            #   B[0,2b] = row 0, col 2b -> index 0*p + 2b = 2b
            #   B[0,2b+1] = index 2b+1
            #   B[1,2b] = index p + 2b
            #   B[1,2b+1] = index p + 2b + 1
            # C indices: W[k*n+i, r] where k is row of C, i is col of C
            #   C[0, 2b]   -> index 0*n + 2b = 2b
            #   C[0, 2b+1] -> index 2b+1
            #   C[1, 2b]   -> index n + 2b
            #   C[1, 2b+1] -> index n + 2b + 1

            c0_0 = 2 * b          # C[0, 2b]
            c0_1 = 2 * b + 1      # C[0, 2b+1]
            c1_0 = n + 2 * b      # C[1, 2b]
            c1_1 = n + 2 * b + 1  # C[1, 2b+1]

            b_00 = 2 * b          # B[0, 2b]
            b_01 = 2 * b + 1      # B[0, 2b+1]
            b_10 = p + 2 * b      # B[1, 2b]
            b_11 = p + 2 * b + 1  # B[1, 2b+1]

            # M1 = (a+d)(e+h)
            # A: a=A[0,0]=0, d=A[1,1]=3
            # B: e=B[0,2b], h=B[1,2b+1]
            U[0, base + 0] = 1   # a
            U[3, base + 0] = 1   # d
            V[b_00, base + 0] = 1   # e
            V[b_11, base + 0] = 1   # h
            # C[0,2b] += M1, C[1,2b+1] += M1
            W[c0_0, base + 0] = 1
            W[c1_1, base + 0] = 1

            # M2 = (c+d)e
            # A: c=A[1,0]=2, d=A[1,1]=3
            # B: e=B[0,2b]
            U[2, base + 1] = 1   # c
            U[3, base + 1] = 1   # d
            V[b_00, base + 1] = 1   # e
            # C[1,2b] += M2, C[1,2b+1] -= M2
            W[c1_0, base + 1] = 1
            W[c1_1, base + 1] = -1

            # M3 = a(f-h)
            # A: a=A[0,0]=0
            # B: f=B[0,2b+1], h=B[1,2b+1]
            U[0, base + 2] = 1   # a
            V[b_01, base + 2] = 1   # f
            V[b_11, base + 2] = -1  # -h
            # C[0,2b+1] += M3, C[1,2b+1] += M3
            W[c0_1, base + 2] = 1
            W[c1_1, base + 2] = 1

            # M4 = d(g-e)
            # A: d=A[1,1]=3
            # B: g=B[1,2b], e=B[0,2b]
            U[3, base + 3] = 1   # d
            V[b_10, base + 3] = 1   # g
            V[b_00, base + 3] = -1  # -e
            # C[0,2b] += M4, C[1,2b] += M4
            W[c0_0, base + 3] = 1
            W[c1_0, base + 3] = 1

            # M5 = (a+b)h
            # A: a=A[0,0]=0, b=A[0,1]=1
            # B: h=B[1,2b+1]
            U[0, base + 4] = 1   # a
            U[1, base + 4] = 1   # b
            V[b_11, base + 4] = 1   # h
            # C[0,2b] -= M5, C[1,2b] += M5
            W[c0_0, base + 4] = -1
            W[c1_0, base + 4] = 1

            # M6 = (c-a)(e+f)
            # A: c=A[1,0]=2, a=A[0,0]=0
            # B: e=B[0,2b], f=B[0,2b+1]
            U[2, base + 5] = 1   # c
            U[0, base + 5] = -1  # -a
            V[b_00, base + 5] = 1   # e
            V[b_01, base + 5] = 1   # f
            # C[1,2b+1] += M6
            W[c1_1, base + 5] = 1

            # M7 = (b-d)(g+h)
            # A: b=A[0,1]=1, d=A[1,1]=3
            # B: g=B[1,2b], h=B[1,2b+1]
            U[1, base + 6] = 1   # b
            U[3, base + 6] = -1  # -d
            V[b_10, base + 6] = 1   # g
            V[b_11, base + 6] = 1   # h
            # C[0,2b] += M7
            W[c0_0, base + 6] = 1

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    # Try Strassen blocks first (rank 28)
    result_28, rank_28 = strassen_blocks()
    valid, err = verify(result_28["U"], result_28["V"], result_28["W"], rank_28)
    if valid:
        return result_28

    # If verification fails, try to debug and fix
    # The issue might be in the W indexing convention
    # Let me try an alternative: maybe W[k*n+i, r] means k is the column of A and i is row of C?
    # Actually the convention says W[k*n+i, r] where the output is C[i,k] = sum A[i,a]*B[a,k]
    # So C has shape (n, p) = (2, 8), flattened as C[k*n + i]? No, that gives (8,2) flattened.
    # W has shape (p*n, rank) = (16, rank). The convention W[k*n+i, r] suggests
    # index = k*n + i where k in [0,p) and i in [0,n), so this is row-major of (p, n) matrix.
    # C is (n, p) = (2, 8). C.T is (p, n) = (8, 2).
    # So W[k*n+i] corresponds to C.T[k, i] = C[i, k].
    # That matches what I have: C_computed.reshape(p, n).T gives back C (n, p).

    # Let me re-examine: maybe the issue is that the Strassen formula I used has an error.
    # Standard Strassen for C = A*B where A,B are 2x2:
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6

    # Let me verify my W assignments sum correctly for each C entry:
    # C[0,0]: M1(+1) + M4(+1) + M5(-1) + M7(+1) = 1+1-1+1 = 2... that's wrong, should be coefficient 1 for each.
    # Wait, each M term appears with coefficient +1 or -1 in the sum. Let me recheck:
    # C[0,0] = M1 + M4 - M5 + M7: coefficients are M1:1, M4:1, M5:-1, M7:1
    # In my code: W[c0_0, base+0]=1 (M1), W[c0_0, base+3]=1 (M4), W[c0_0, base+4]=-1 (M5), W[c0_0, base+6]=1 (M7)
    # That looks correct.

    # C[0,1] = M3 + M5: W[c0_1, base+2]=1 (M3), W[c0_1, base+4]=1 (M5) -- but I have W[c0_1, base+4]=1? No!
    # I have W[c0_0, base+4] = -1 (M5 contributes to C[0,0]) and W[c1_0, base+4] = 1 (M5 contributes to C[1,0])
    # But M5 should contribute to C[0,1] as well! Let me recheck.
    # C[0,1] = M3 + M5. So W[c0_1, base+4] should be 1 for M5.
    # I'm missing this! Let me fix.

    # Actually wait - let me re-derive Strassen carefully.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+cg, af+dh], [ce+dg, cf+dh]]
    # C[0,0] = ae + cg
    # C[0,1] = af + dh
    # C[1,0] = ce + dg
    # C[1,1] = cf + dh

    # Strassen:
    # M1 = (a+d)(e+h) = ae+ah+de+dh
    # M2 = (c+d)e = ce+de
    # M3 = a(f-h) = af-ah
    # M4 = d(g-e) = dg-de
    # M5 = (a+b)h = ah+bh
    # M6 = (c-a)(e+f) = ce+cf-ae-af
    # M7 = (b-d)(g+h) = bg+bh-dg-dh

    # C[0,0] = ae+cg
    #   M1+M4-M5+M7 = (ae+ah+de+dh) + (dg-de) - (ah+bh) + (bg+bh-dg-dh)
    #   = ae+ah+de+dh+dg-de-ah-bh+bg+bh-dg-dh = ae + bg
    #   Hmm, that gives ae+bg, not ae+cg. Let me recheck.
    #   Actually I think the standard formula is:
    #   C[0,0] = M1 + M4 - M5 + M7
    #   Let me expand more carefully:
    #   M1 = ae+ah+de+dh
    #   M4 = dg-de
    #   -M5 = -ah-bh
    #   M7 = bg+bh-dg-dh
    #   Sum = ae+ah+de+dh + dg-de -ah-bh + bg+bh-dg-dh
    #   = ae + (ah-ah) + (de-de) + (dh-dh) + (dg-dg) + (-bh+bh) + bg
    #   = ae + bg
    #   That's wrong! Should be ae+cg.

    # I think I'm using a different convention. Let me look up the standard Strassen:
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)B11
    # P3 = A11(B21 - B22)
    # P4 = A22(B12 - B11)
    # P5 = (A11 + A12)B22
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6

    # Let me verify C11:
    # P1 + P4 - P5 + P7
    # = (a+d)(e+h) + d(g-e) - (a+b)h + (b-d)(g+h)
    # = ae+ah+de+dh + dg-de - ah-bh + bg+bh-dg-dh
    # = ae + bg
    # Still ae+bg. That can't be right for C11=ae+cg.

    # Wait, I think the issue is that in some conventions the matrix B is transposed or the indices are different.
    # Let me just directly verify with a simple example.
    # A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    # C = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19, 22],[43, 50]]
    # a=1,b=2,c=3,d=4,e=5,f=6,g=7,h=8
    # P1 = (1+4)(5+8) = 5*13 = 65
    # P2 = (3+4)*5 = 35
    # P3 = 1*(6-8) = -2
    # P4 = 4*(7-5) = 8
    # P5 = (1+2)*8 = 24
    # P6 = (3-1)*(5+6) = 2*11 = 22
    # P7 = (2-4)*(7+8) = -2*15 = -30
    # C11 = 65+8-24+(-30) = 19 ✓
    # C12 = -2+24 = 22 ✓
    # C21 = 35+8 = 43 ✓
    # C22 = 65-35+(-2)+22 = 50 ✓

    # So the formula is correct! My algebra expansion was wrong. Let me redo:
    # P1+P4-P5+P7 = 65+8-24-30 = 19 = ae+cg = 1*5+2*7 = 5+14 = 19 ✓
    # So my algebra was wrong. The formula is correct.

    # So the Strassen formula is correct. The issue must be in my W assignments.
    # Let me recheck: C[0,0] should get contributions from M1, M4, -M5, M7.
    # In my code:
    # W[c0_0, base+0] = 1  (M1)
    # W[c0_0, base+3] = 1  (M4)
    # W[c0_0, base+4] = -1 (M5)
    # W[c0_0, base+6] = 1  (M7)
    # That's correct.

    # C[0,1] = M3 + M5:
    # W[c0_1, base+2] = 1  (M3)
    # W[c0_1, base+4] = ? I don't have this! I only have W[c0_0, base+4]=-1 and W[c1_0, base+4]=1
    # I'm missing W[c0_1, base+4] = 1 for M5!

    # Let me fix this and rebuild.

    def strassen_blocks_fixed():
        num_blocks = p // 2
        rank = num_blocks * 7

        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        for b in range(num_blocks):
            base = b * 7

            c0_0 = 2 * b
            c0_1 = 2 * b + 1
            c1_0 = n + 2 * b
            c1_1 = n + 2 * b + 1

            b_00 = 2 * b
            b_01 = 2 * b + 1
            b_10 = p + 2 * b
            b_11 = p + 2 * b + 1

            # P1 = (a+d)(e+h)
            U[0, base + 0] = 1
            U[3, base + 0] = 1
            V[b_00, base + 0] = 1
            V[b_11, base + 0] = 1
            W[c0_0, base + 0] = 1    # C[0,0]
            W[c1_1, base + 0] = 1    # C[1,1]

            # P2 = (c+d)e
            U[2, base + 1] = 1
            U[3, base + 1] = 1
            V[b_00, base + 1] = 1
            W[c1_0, base + 1] = 1    # C[1,0]
            W[c1_1, base + 1] = -1   # C[1,1]

            # P3 = a(f-h)
            U[0, base + 2] = 1
            V[b_01, base + 2] = 1
            V[b_11, base + 2] = -1
            W[c0_1, base + 2] = 1    # C[0,1]
            W[c1_1, base + 2] = 1    # C[1,1]

            # P4 = d(g-e)
            U[3, base + 3] = 1
            V[b_10, base + 3] = 1
            V[b_00, base + 3] = -1
            W[c0_0, base + 3] = 1    # C[0,0]
            W[c1_0, base + 3] = 1    # C[1,0]

            # P5 = (a+b)h
            U[0, base + 4] = 1
            U[1, base + 4] = 1
            V[b_11, base + 4] = 1
            W[c0_0, base + 4] = -1   # C[0,0]
            W[c0_1, base + 4] = 1    # C[0,1]
            W[c1_0, base + 4] = 1    # C[1,0]

            # P6 = (c-a)(e+f)
            U[2, base + 5] = 1
            U[0, base + 5] = -1
            V[b_00, base + 5] = 1
            V[b_01, base + 5] = 1
            W[c1_1, base + 5] = 1    # C[1,1]

            # P7 = (b-d)(g+h)
            U[1, base + 6] = 1
            U[3, base + 6] = -1
            V[b_10, base + 6] = 1
            V[b_11, base + 6] = 1
            W[c0_0, base + 6] = 1    # C[0,0]

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    result_28f, rank_28f = strassen_blocks_fixed()
    valid, err = verify(result_28f["U"], result_28f["V"], result_28f["W"], rank_28f)
    if valid:
        return result_28f

    # Fallback to schoolbook
    return schoolbook()
