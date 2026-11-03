"""Matrix multiplication 4x4x9 with reduced rank tensor decomposition."""

import time

DIMENSIONS = [4, 4, 9]

def _validate(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    # Build the multiplication tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # and check it equals 1 for all (i,j,k).
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for c in range(rank):
                    s += U[ui][c] * V[vj][c] * W[wk][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # Decompose p=9 as 4+4+1.
    # For the 4x4x4 part, use Strassen's algorithm (rank 7).
    # For the 4x4x1 part, use schoolbook (rank 4).
    # Total rank = 3*7 + 4 = 25.
    #
    # Strassen for 2x2x2 (rank 7):
    #   U (4x7), V (4x7), W (4x7) where indices are (i*2+j), (j*2+k), (k*2+i)
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11)B00
    #   M3 = A00(B01-B11)
    #   M4 = A11(B10-B00)
    #   M5 = (A00+A10)B11
    #   M6 = (A01-A10)(B10+B11)
    #   M7 = (A01-A11)(B00+B10)
    #   C00 = M1+M4-M5+M7
    #   C01 = M3+M5
    #   C10 = M2+M4
    #   C11 = M1-M2+M3+M6
    #
    # U rows: A00, A01, A10, A11 (index = i*2+j)
    # V rows: B00, B01, B10, B11 (index = j*2+k)
    # W rows: C00, C01, C10, C11 (index = k*2+i)
    #
    # Strassen U (4x7):
    #   A00: M1, M3, M4, M5 -> cols 0,2,3,4
    #   A01: M6, M7 -> cols 5,6
    #   A10: M2, M5, M6 -> cols 1,4,5
    #   A11: M1, M2, M4, M7 -> cols 0,1,3,6
    # Strassen V (4x7):
    #   B00: M1, M2, M4, M7 -> cols 0,1,3,6
    #   B01: M3, M6 -> cols 2,5
    #   B10: M4, M6, M7 -> cols 3,5,6
    #   B11: M1, M5, M6 -> cols 0,4,5
    # Strassen W (4x7):
    #   C00: M1, M4, M5, M7 -> cols 0,3,4,6
    #   C01: M3, M5 -> cols 2,4
    #   C10: M2, M4 -> cols 1,3
    #   C11: M1, M2, M3, M6 -> cols 0,1,2,5

    # Now we need to map the 4x4x4 Strassen to our 4x4x4 sub-problem.
    # The 4x4x4 problem has A(4x4), B(4x4), C(4x4).
    # We can view A as a 2x2 block matrix of 2x2 blocks:
    #   A = [[A00, A01], [A10, A11]] where each is 2x2.
    # Similarly B and C.
    #
    # For the 4x4x4 Strassen, we need U(16x7), V(16x7), W(16x7).
    # U rows indexed by (i,j) where i in 0..3, j in 0..3 -> i*4+j.
    # Each 2x2 block Aab has entries A[2a+i, 2b+j] for i,j in 0..1.
    # The Strassen U for block Aab has 4 rows (for the 2x2 sub-problem) mapped to 16 rows.
    #
    # Actually, let's think differently. The 4x4x4 matrix multiplication tensor
    # can be decomposed using Strassen by treating it as 2x2 blocks of 2x2 matrices.
    # The Strassen decomposition for 2x2x2 gives rank 7. For 4x4x4, we need to
    # "inflate" each 2x2 entry into a 2x2 block.
    #
    # For the 4x4x4 Strassen:
    # U (16x7): row (i,j) where i=2a+r, j=2b+s, with a,b,r,s in {0,1}.
    #   U[2a+r, 2b+s, c] = U_2x2[a, b, c] for the (a,b)-th 2x2 block.
    #   Actually, U_2x2 has rows indexed by (a,b) -> a*2+b.
    #   So U[2a+r, 2b+s, c] = U2[a*2+b, c].
    #
    # Similarly V (16x7): V[2b+s, 2c+t, d] = V2[b*2+c, d] where b,c,s,t in {0,1}.
    # Wait, V is indexed by (j,k) where j is row of B, k is col of B.
    # B has blocks B_bc where b,c in {0,1}, each 2x2.
    # V row index is (j,k) -> j*4+k. j=2b+s, k=2c+t.
    # V[2b+s, 2c+t, d] = V2[b*2+c, d].
    #
    # W (16x7): W[2c+t, 2a+r, e] = W2[c*2+a, e].
    # W row index is (k,i) -> k*4+i. k=2c+t, i=2a+r.
    # W[2c+t, 2a+r, e] = W2[c*2+a, e].

    # Define the 2x2x2 Strassen matrices (each 4x7):
    U2 = [
        [1, 0, 1, 1, 1, 0, 0],  # A00: M1, M3, M4, M5
        [0, 0, 0, 0, 0, 1, 1],  # A01: M6, M7
        [0, 1, 0, 0, 1, 1, 0],  # A10: M2, M5, M6
        [1, 1, 0, 1, 0, 0, 1],  # A11: M1, M2, M4, M7
    ]
    V2 = [
        [1, 1, 0, 1, 0, 0, 1],  # B00: M1, M2, M4, M7
        [0, 0, 1, 0, 0, 1, 0],  # B01: M3, M6
        [0, 0, 0, 1, 0, 1, 1],  # B10: M4, M6, M7
        [1, 0, 0, 0, 1, 1, 0],  # B11: M1, M5, M6
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],  # C00: M1, M4, M5, M7
        [0, 1, 0, 1, 0, 0, 0],   # C10: M2, M4
        [0, 0, 1, 0, 1, 0, 0],   # C01: M3, M5
        [1, -1, 1, 0, 0, 1, 0],  # C11: M1, M2, M3, M6
    ]
    # Wait, I need to double-check the W ordering.
    # W2 rows: C00, C01, C10, C11 with index k*2+i.
    # C00 -> k=0,i=0 -> row 0
    # C01 -> k=1,i=0 -> row 2
    # C10 -> k=0,i=1 -> row 1
    # C11 -> k=1,i=1 -> row 3
    # So W2[row 0] = C00, W2[row 1] = C10, W2[row 2] = C01, W2[row 3] = C11.
    # Let me reorder:
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],  # row 0: C00 (k=0,i=0)
        [0, 1, 0, 1, 0, 0, 0],   # row 1: C10 (k=0,i=1)
        [0, 0, 1, 0, 1, 0, 0],   # row 2: C01 (k=1,i=0)
        [1, -1, 1, 0, 0, 1, 0],  # row 3: C11 (k=1,i=1)
    ]

    # Build 4x4x4 Strassen U (16x7), V (16x7), W (16x7):
    U44 = [[0]*7 for _ in range(16)]
    V44 = [[0]*7 for _ in range(16)]
    W44 = [[0]*7 for _ in range(16)]
    for a in range(2):
        for b in range(2):
            for r in range(2):
                for s in range(2):
                    urow = (2*a + r) * 4 + (2*b + s)  # i=2a+r, j=2b+s
                    vrow = (2*b + s) * 4 + (2*a + r)  # j=2b+s, k=2a+r (for V of 4x4x4, but wait)
                    # Actually for 4x4x4: U indexed by (i,j), V by (j,k), W by (k,i).
                    # U[i,j,c] where i=2a+r, j=2b+s: U[i,j,c] = U2[a*2+b, c]
                    # V[j,k,c] where j=2b+s, k=2c+t: V[j,k,c] = V2[b*2+c, c] -- no, V2 indexed by (b,c) block
                    # Let me redo:
                    # A block (a,b): rows 2a..2a+1, cols 2b..2b+1
                    # B block (b,c): rows 2b..2b+1, cols 2c..2c+1
                    # C block (c,a): rows 2c..2c+1, cols 2a..2a+1
                    #
                    # U44[i,j,c] for i=2a+r, j=2b+s: U44[i,j,c] = U2[a*2+b, c]
                    # V44[j,k,c] for j=2b+s, k=2c+t: V44[j,k,c] = V2[b*2+c, c] -- conflict with c
                    # Let me use different loop variables.
    # Let me redo this more carefully.
    U44 = [[0]*7 for _ in range(16)]
    V44 = [[0]*7 for _ in range(16)]
    W44 = [[0]*7 for _ in range(16)]
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for r in range(2):
                    for s in range(2):
                        # U: i=2a+r, j=2b+s
                        urow = (2*a + r) * 4 + (2*b + s)
                        U44[urow][c] = U2[a*2 + b][c]
                        # V: j=2b+s, k=2c+t -> but t is the inner index for B's block
                        # V[j,k,d] where j=2b+s, k=2c+t
                        for t in range(2):
                            vrow = (2*b + s) * 4 + (2*c + t)
                            V44[vrow][c] = V2[b*2 + c][c]
                        # W: k=2c+t, i=2a+r
                        for t in range(2):
                            wrow = (2*c + t) * 4 + (2*a + r)
                            W44[wrow][c] = W2[c*2 + a][c]

    # Now for the 4x4x1 part (p=1):
    # A(4x4) * B(4x1) = C(4x1). Schoolbook rank 16.
    # U(16x16), V(4x16), W(4x16).
    # U[i,j,c] = 1 if c == i*4+j
    # V[j,k,c] = 1 if c == j*4+k (but k only 0..0 since p=1, so k=0)
    #   V[j, 0, c] = 1 if c == j*4+0 = j*4
    # W[k,i,c] = 1 if c == k*4+i (k=0)
    #   W[0, i, c] = 1 if c == 0*4+i = i
    #
    # Wait, for the 4x4x1 case:
    # U has shape (n*m, rank) = (16, 16)
    # V has shape (m*p, rank) = (4, 16)
    # W has shape (p*n, rank) = (4, 16)
    # U[i,j,c] = delta(c, i*4+j)
    # V[j,k,c] = delta(c, j*4+k) with k=0 only -> V[j,0,c] = delta(c, j*4)
    # W[k,i,c] = delta(c, k*4+i) with k=0 only -> W[0,i,c] = delta(c, i)

    U41 = [[0]*16 for _ in range(16)]
    V41 = [[0]*16 for _ in range(4)]
    W41 = [[0]*16 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            c = i * 4 + j
            U41[i*4 + j][c] = 1
    for j in range(4):
        c = j * 4
        V41[j][c] = 1
    for i in range(4):
        c = i
        W41[0][c] = 1  # W[k=0, i, c]

    # Now combine for 4x4x9:
    # p=9 = 4+4+1.
    # B is 4x9, split into B1(4x4), B2(4x4), B3(4x1).
    # C is 4x9, split into C1(4x4), C2(4x4), C3(4x1).
    #
    # U (16x25): same for all three parts.
    #   Columns 0-6: U44 (for B1)
    #   Columns 7-13: U44 (for B2)
    #   Columns 14-24: U41 (for B3)
    #
    # V (36x25): B1 rows 0-3, B2 rows 4-7, B3 rows 8-11 (m*p = 4*9 = 36)
    #   V[j,k,c] where j in 0..3, k in 0..8.
    #   For B1 (k=0..3): V[j, k, c] = V44[j*4+k, c] for c in 0..6
    #   For B2 (k=4..7): V[j, k, c] = V44[j*4+(k-4), c-7] for c in 7..13
    #   For B3 (k=8): V[j, 8, c] = V41[j, c-14] for c in 14..24
    #
    # W (36x25): C1 rows 0-3, C2 rows 4-7, C3 rows 8-11 (p*n = 9*4 = 36)
    #   W[k,i,c] where k in 0..8, i in 0..3.
    #   For C1 (k=0..3): W[k, i, c] = W44[k*4+i, c] for c in 0..6
    #   For C2 (k=4..7): W[k, i, c] = W44[(k-4)*4+i, c-7] for c in 7..13
    #   For C3 (k=8): W[8, i, c] = W41[i, c-14] for c in 14..24

    rank = 25
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]

    # Fill U
    for i in range(16):
        for c in range(7):
            U[i][c] = U44[i][c]
            U[i][c+7] = U44[i][c]
        for c in range(16):
            U[i][c+14] = U41[i][c]

    # Fill V
    for j in range(4):
        for k in range(4):
            for c in range(7):
                V[j*4 + k][c] = V44[j*4 + k][c]
        for k in range(4):
            for c in range(7):
                V[j*4 + (k+4)][c+7] = V44[j*4 + k][c]
        for c in range(16):
            V[j*4 + 8][c+14] = V41[j][c]

    # Fill W
    for k in range(4):
        for i in range(4):
            for c in range(7):
                W[k*4 + i][c] = W44[k*4 + i][c]
    for k in range(4):
        for i in range(4):
            for c in range(7):
                W[(k+4)*4 + i][c+7] = W44[k*4 + i][c]
    for i in range(4):
        for c in range(16):
            W[8*4 + i][c+14] = W41[i][c]

    return {"U": U, "V": V, "W": W}
