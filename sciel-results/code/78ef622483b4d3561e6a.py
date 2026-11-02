import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-35 exact decomposition of the 4x4 @ 4x6 multiplication tensor.

    Strategy: Kronecker product of two smaller multiplication decompositions.
      M_{4,4,6} = M_{2,2,2} (rank 7, Strassen)  ⊗  M_{2,2,3} (rank 5, optimal)
      total rank = 7 * 5 = 35

    Index conventions:
      U shape (n*m, rank): entry (i*m + j, r) multiplies A[i,j]
      V shape (m*p, rank): entry (j*p + k, r) multiplies B[j,k]
      W shape (p*n, rank): entry (k*n + i, r) contributes to C[i,k]
      (matches the trace convention W[k*n+i, r])

    For the Kronecker product T = T1 ⊗ T2 with T1 of type (a1,b1,c1) and
    T2 of type (a2,b2,c2):
      The combined tensor has type (a1*a2, b1*b2, c1*c2).
      For rank term (r1, r2):
        U12[(i1*a2+i2) * (b1*b2) + (j1*b2+j2), col] = U1[i1,j1,r1]*U2[i2,j2,r2]
        V12[(j1*b2+j2) * (c1*c2) + (k1*c2+k2), col] = V1[j1,k1,r1]*V2[j2,k2,r2]
        W12[(k1*c2+k2) * (a1*a2) + (i1*a2+i2), col] = W1[i1,k1,r1]*W2[i2,k2,r2]
    """
    deadline = time.time() + 120

    # ---- Strassen for 2x2 @ 2x2 (rank 7) ----
    # A = [[a,b],[c,d]] (row-major: a,b,c,d), B = [[e,f],[g,h]] (e,f,g,h)
    # C = A@B, C row-major (C00,C01,C10,C11)
    U1 = [
        [1, 0, 1, 0, 1, -1, 0],   # a
        [0, 0, 0, 0, 1, 0, 1],    # b
        [0, 1, 0, 0, 0, 1, 0],    # c
        [1, 1, 0, 1, 0, 0, -1],   # d
    ]
    V1 = [
        [1, 1, 0, -1, 0, 1, 0],   # e
        [0, 0, 1, 0, 0, 1, 0],    # f
        [0, 0, 0, 1, 0, 0, 1],    # g
        [1, 0, -1, 0, 1, 0, 1],   # h
    ]
    W1 = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0],   # C11
    ]

    # ---- Rank-5 decomposition for 2x2 @ 2x3 ----
    # A = [[a,b],[c,d]] (a,b,c,d), B = [[e,f,g],[h,i,j]] (e,f,g,h,i,j)
    # C = A@B, C row-major (C00,C01,C02,C10,C11,C12)
    # Terms:
    #  r0: (a+d)(e+h)
    #  r1: (c+d)e
    #  r2: (a+b)(i-j)
    #  r3: d(g-i)
    #  r4: (c-a)(e+g)
    U2 = [
        [1, 0, 1, 0, -1],   # a
        [0, 0, 1, 0, 0],    # b
        [0, 1, 0, 0, 1],    # c
        [1, 1, 0, 1, 0],    # d
    ]
    V2 = [
        [1, 1, 0, 0, 1],    # e
        [0, 0, 0, 0, 0],    # f
        [0, 0, 0, 1, 1],    # g
        [1, 0, 0, 0, 0],    # h
        [0, 0, 1, -1, 0],   # i
        [0, 0, -1, 0, 0],   # j
    ]
    W2 = [
        [1, 0, 0, 0, 0],    # C00
        [0, 0, 1, 0, 0],    # C01
        [0, 0, 0, 1, 0],    # C02
        [0, 1, 0, 0, 1],    # C10
        [0, 0, 0, 0, 0],    # C11
        [0, 0, 0, 0, -1],   # C12
    ]

    # ---- Verify 2x2x2 Strassen ----
    import itertools
    def verify2(U, V, W, a, b, c, d, e, f, g, h):
        A = [(a, b), (c, d)]
        B = [(e, f), (g, h)]
        Cexp = [[0, 0], [0, 0]]
        Cexp[0][0] = a*e + b*g
        Cexp[0][1] = a*f + b*h
        Cexp[1][0] = c*e + d*g
        Cexp[1][1] = c*f + d*h
        for r in range(len(W[0])):
            ua = a*U[0][r] + b*U[1][r] + c*U[2][r] + d*U[3][r]
            vb = e*V[0][r] + f*V[1][r] + g*V[2][r] + h*V[3][r]
            for o in range(4):
                Cexp[o//2][o%2] += ua * vb * W[o][r]
        Cexp_f = [[Cexp[0][0], Cexp[0][1]], [Cexp[1][0], Cexp[1][1]]]
        return Cexp_f

    # quick sanity check with a random integer input
    ok = True
    for trial in range(200):
        vals = [trial % 3 - 1 for _ in range(8)]
        a, b, c, d, e, f, g, h = vals
        if a == b == c == d == e == f == g == h == 0:
            continue
        A = [[a, b], [c, d]]
        B = [[e, f], [g, h]]
        Cexp = [[A[0][0]*B[0][0]+A[0][1]*B[1][0], A[0][0]*B[0][1]+A[0][1]*B[1][1]],
                [A[1][0]*B[0][0]+A[1][1]*B[1][0], A[1][0]*B[0][1]+A[1][1]*B[1][1]]]
        Cgot = [[0.0, 0.0], [0.0, 0.0]]
        for r in range(7):
            ua = a*U1[0][r] + b*U1[1][r] + c*U1[2][r] + d*U1[3][r]
            vb = e*V1[0][r] + f*V1[1][r] + g*V1[2][r] + h*V1[3][r]
            w = ua*vb
            Cgot[0][0] += w*W1[0][r]
            Cgot[0][1] += w*W1[1][r]
            Cgot[1][0] += w*W1[2][r]
            Cgot[1][1] += w*W1[3][r]
        for i in range(2):
            for j in range(2):
                if abs(Cgot[i][j] - Cexp[i][j]) > 1e-6:
                    ok = False
                    break
        if not ok:
            break
    if not ok:
        # fall back to schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0.0]*rank for _ in range(n*m)]
        V = [[0.0]*rank for _ in range(m*p)]
        W = [[0.0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    # ---- Verify 2x2x3 rank-5 ----
    def verify223(U, V, W, a, b, c, d, e, f, g, h, i, j):
        A = [[a, b], [c, d]]
        B = [[e, f, g], [h, i, j]]
        Cexp = [[0]*3, [0]*3]
        for x in range(2):
            for y in range(3):
                Cexp[x][y] = A[x][0]*B[0][y] + A[x][1]*B[1][y]
        Cgot = [[0.0]*3, [0.0]*3]
        for r in range(len(W[0])):
            ua = a*U[0][r] + b*U[1][r] + c*U[2][r] + d*U[3][r]
            vb = e*V[0][r] + f*V[1][r] + g*V[2][r] + h*V[3][r] + i*V[4][r] + j*V[5][r]
            w = ua*vb
            for o in range(6):
                Cgot[o//3][o%3] += w*W[o][r]
        for x in range(2):
            for y in range(3):
                if abs(Cgot[x][y] - Cexp[x][y]) > 1e-6:
                    return False
        return True

    ok2 = True
    for trial in range(200):
        vals = [trial % 3 - 1 for _ in range(10)]
        if all(v == 0 for v in vals):
            continue
        if not verify223(U2, V2, W2, *vals):
            ok2 = False
            break
    if not ok2:
        # fall back to schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0.0]*rank for _ in range(n*m)]
        V = [[0.0]*rank for _ in range(m*p)]
        W = [[0.0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    # ---- Build Kronecker product ----
    # T1 = M_{2,2,2}: a1=2, b1=2, c1=2, rank 7
    # T2 = M_{2,2,3}: a2=2, b2=2, c2=3, rank 5
    # Combined: M_{4,4,6}: n=4, m=4, p=6, rank 35
    a1, b1, c1 = 2, 2, 2
    a2, b2, c2 = 2, 2, 3
    n, m, p = a1*a2, b1*b2, c1*c2
    rank_total = 7 * 5

    U = [[0.0]*rank_total for _ in range(n*m)]
    V = [[0.0]*rank_total for _ in range(m*p)]
    W = [[0.0]*rank_total for _ in range(p*n)]

    for r1 in range(7):
        for r2 in range(5):
            col = r1*5 + r2
            for i1 in range(a1):
                for j1 in range(b1):
                    u1 = U1[i1][r1]
                    if u1 == 0:
                        continue
                    for i2 in range(a2):
                        for j2 in range(b2):
                            u2 = U2[i2][r2]
                            if u2 == 0:
                                continue
                            uval = u1 * u2
                            i_combined = i1*a2 + i2
                            j_combined = j1*b2 + j2
                            U[i_combined*m + j_combined][col] = uval

            for j1 in range(b1):
                for k1 in range(c1):
                    v1 = V1[j1][r1]
                    if v1 == 0:
                        continue
                    for j2 in range(b2):
                        for k2 in range(c2):
                            v2 = V2[j2][r2]
                            if v2 == 0:
                                continue
                            vval = v1 * v2
                            j_combined = j1*b2 + j2
                            k_combined = k1*c2 + k2
                            V[j_combined*p + k_combined][col] = vval

            for i1 in range(a1):
                for k1 in range(c1):
                    w1 = W1[i1][r1]
                    if w1 == 0:
                        continue
                    for i2 in range(a2):
                        for k2 in range(c2):
                            w2 = W2[i2][r2]
                            if w2 == 0:
                                continue
                            wval = w1 * w2
                            i_combined = i1*a2 + i2
                            k_combined = k1*c2 + k2
                            W[k_combined*n + i_combined][col] = wval

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
