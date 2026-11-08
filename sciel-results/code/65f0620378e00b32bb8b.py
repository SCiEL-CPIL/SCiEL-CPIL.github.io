import time

def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        from fractions import Fraction
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 coefficients
    # Rows correspond to the 7 multiplications.
    # Cols in U correspond to (i, a) where i in 0..1, a in 0..1. Index i*2+a.
    # Cols in V correspond to (a, k) where a in 0..1, k in 0..1. Index a*2+k.
    # Cols in W correspond to (k, i) where k in 0..1, i in 0..1. Index k*2+i.
    
    # Standard Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # U (A side):
    # M1: A11(1,1) + A22(2,2) -> [1, 0, 1, 0] (if indices 00,01,10,11) -> [1, 0, 0, 1] in row-major i*2+a?
    # i=0,a=0 (00), i=0,a=1 (01), i=1,a=0 (10), i=1,a=1 (11)
    # M1: A11(00) + A22(11) -> [1, 0, 0, 1]
    # M2: A21(10) + A22(11) -> [0, 0, 1, 1]
    # M3: A11(00) -> [1, 0, 0, 0]
    # M4: A22(11) -> [0, 0, 0, 1]
    # M5: A11(00) + A12(01) -> [1, 1, 0, 0]
    # M6: A21(10) - A11(00) -> [-1, 0, 1, 0]
    # M7: A12(01) - A22(11) -> [0, 1, 0, -1]
    
    S_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    # V (B side):
    # Indices a*2+k. a in 0..1, k in 0..1.
    # 00 (a0k0), 01 (a0k1), 10 (a1k0), 11 (a1k1)
    # M1: B11(00) + B22(11) -> [1, 0, 0, 1]
    # M2: B11(00) -> [1, 0, 0, 0]
    # M3: B21(10) - B22(11) -> [0, 0, 1, -1]
    # M4: B12(01) - B11(00) -> [-1, 1, 0, 0]  Wait, M4 = A22(B12 - B11). B12 is a0k1? No.
    # B is 2x2. Rows a, Cols k.
    # B11 is a=0,k=0. B12 is a=0,k=1. B21 is a=1,k=0. B22 is a=1,k=1.
    # M4: B12 - B11 -> B(0,1) - B(0,0) -> [ -1, 1, 0, 0]
    # M5: B22(11) -> [0, 0, 0, 1]
    # M6: B11(00) + B12(01) -> [1, 1, 0, 0]
    # M7: B21(10) + B22(11) -> [0, 0, 1, 1]
    
    S_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 1, -1],# M3
        [-1, 1, 0, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # W (C side):
    # Indices k*2+i. k in 0..1, i in 0..1.
    # 00 (k0i0), 01 (k0i1), 10 (k1i0), 11 (k1i1)
    # C = 
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 + M7
    
    # W rows for each multiplication M_r contribute to C entries.
    # We need W[r][k*2+i] such that sum_r U[r][i,a] V[r][a,k] W[r][k,i] = C[i,k]
    
    # M1 contributes to C11, C22
    # C11 (i=0,k=0): +1 -> W[0][0] = 1
    # C22 (i=1,k=1): +1 -> W[0][3] = 1
    # Others 0.
    # W[0] = [1, 0, 0, 1]
    
    # M2 contributes to C21
    # C21 (i=1,k=0): +1 -> W[1][2] = 1
    # W[1] = [0, 0, 1, 0]
    
    # M3 contributes to C12, C22
    # C12 (i=0,k=1): +1 -> W[2][1] = 1
    # C22 (i=1,k=1): -1 -> W[2][3] = -1
    # W[2] = [0, 1, 0, -1]
    
    # M4 contributes to C11, C21
    # C11 (i=0,k=0): -1 -> W[3][0] = -1
    # C21 (i=1,k=0): +1 -> W[3][2] = 1
    # W[3] = [-1, 0, 1, 0]
    
    # M5 contributes to C11, C12
    # C11 (i=0,k=0): +1 -> W[4][0] = 1
    # C12 (i=0,k=1): +1 -> W[4][1] = 1
    # W[4] = [1, 1, 0, 0]
    
    # M6 contributes to C22
    # C22 (i=1,k=1): +1 -> W[5][3] = 1
    # W[5] = [0, 0, 0, 1]
    
    # M7 contributes to C11, C22
    # C11 (i=0,k=0): +1 -> W[6][0] = 1
    # C22 (i=1,k=1): +1 -> W[6][3] = 1
    # W[6] = [1, 0, 0, 1]
    
    S_W = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 1]  # M7
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i * 2 + a]
                        v_val = S_V[r][a * 2 + k]
                        w_val = S_W[r][k * 2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Rank 44 construction
    rank_total = 44
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]

    col = 0

    # Part 1: A_left (2x2) * B_top (2x8) using 4 Strassen blocks
    # A_left is rows 0,1 of A. Cols 0,1 of A.
    # B_top is rows 0,1 of B. Cols 0..7 of B.
    # We split B_top into 4 blocks of 2x2: cols 0-1, 2-3, 4-5, 6-7.
    
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            # Fill U
            # U rows are i*m+a. i in 0..1, a in 0..1 (for the 2x2 block).
            # Global A index: i*3 + a.
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U[li * m + la][col] = u_val

            # Fill V
            # V rows are a*p+k. a in 0..1 (for the 2x2 block). k in 0..1 (local).
            # Global B index: a*8 + (k0 + lk).
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V[la * p + (k0 + lk)][col] = v_val

            # Fill W
            # W rows are k*n+i. k in 0..1 (local). i in 0..1.
            # Global C index: (k0 + lk)*2 + li.
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W[(k0 + lk) * n + li][col] = w_val

            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Part 2: A_right (2x1) * B_bottom (1x8) schoolbook
    # A_right is col 2 of A.
    # B_bottom is row 2 of B.
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}
    if _validate(candidate):
        return candidate
    return _schoolbook()
