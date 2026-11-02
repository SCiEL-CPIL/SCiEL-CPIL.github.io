"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal current_rank
        r = current_rank
        for (i, j), val in u_coeffs.items():
            U[i * m + j][r] = val
        for (j, k), val in v_coeffs.items():
            V[j * p + k][r] = val
        for (k, i), val in w_coeffs.items():
            W[k * n + i][r] = val
        current_rank += 1

    # Strassen 2x2x2 terms
    # m1 = (a00+a11)(b00+b11)
    def strassen_m1(rows, cols, j_start):
        # rows: tuple of 2 rows
        # cols: tuple of 2 cols
        # j_start: start of j index (0 or 2)
        i0, i1 = rows
        k0, k1 = cols
        j0, j1 = j_start, j_start + 1
        
        u = {(i0, j0): 1, (i0, j1): 0, (i1, j0): 0, (i1, j1): 1}
        v = {(j0, k0): 1, (j0, k1): 1, (j1, k0): 1, (j1, k1): 1}
        w = {(k0, i0): 1, (k1, i0): 0, (k0, i1): 0, (k1, i1): 1}
        # c00 += m1, c11 += m1
        # m1 contributes to c00 and c11
        # In Strassen: c00 = m1 + m4 - m5 + m7; c11 = m1 - m2 + m3 + m6
        # So m1 adds 1 to c00 and 1 to c11
        w = {(k0, i0): 1, (k1, i1): 1}
        add_term(u, v, w)

    def strassen_m2(rows, cols, j_start):
        i0, i1 = rows
        k0, k1 = cols
        j0, j1 = j_start, j_start + 1
        # m2 = (a10+a11)b00
        u = {(i1, j0): 1, (i1, j1): 1}
        v = {(j0, k0): 1}
        # c10 += m2, c11 -= m2
        w = {(k0, i1): 1, (k1, i1): -1}
        add_term(u, v, w)

    def strassen_m3(rows, cols, j_start):
        i0, i1 = rows
        k0, k1 = cols
        j0, j1 = j_start, j_start + 1
        # m3 = a00(b01-b11)
        u = {(i0, j0): 1}
        v = {(j0, k1): 1, (j1, k1): -1}
        # c01 += m3, c11 += m3
        w = {(k1, i0): 1, (k1, i1): 1}
        add_term(u, v, w)

    def strassen_m4(rows, cols, j_start):
        i0, i1 = rows
        k0, k1 = cols
        j0, j1 = j_start, j_start + 1
        # m4 = a11(b10-b00)
        u = {(i1, j1): 1}
        v = {(j1, k0): 1, (j0, k0): -1}
        # c00 += m4, c10 += m4
        w = {(k0, i0): 1, (k0, i1): 1}
        add_term(u, v, w)

    def strassen_m5(rows, cols, j_start):
        i0, i1 = rows
        k0, k1 = cols
        j0, j1 = j_start, j_start + 1
        # m5 = (a00+a01)b11
        u = {(i0, j0): 1, (i0, j1): 1}
        v = {(j1, k1): 1}
        # c00 -= m5, c01 += m5
        w = {(k0, i0): -1, (k1, i0): 1}
        add_term(u, v, w)

    def strassen_m6(rows, cols, j_start):
        i0, i1 = rows
        k0, k1 = cols
        j0, j1 = j_start, j_start + 1
        # m6 = (a10-a00)(b00+b01)
        u = {(i1, j0): 1, (i0, j0): -1}
        v = {(j0, k0): 1, (j0, k1): 1}
        # c11 += m6
        w = {(k1, i1): 1}
        add_term(u, v, w)

    def strassen_m7(rows, cols, j_start):
        i0, i1 = rows
        k0, k1 = cols
        j0, j1 = j_start, j_start + 1
        # m7 = (a01-a11)(b10+b11)
        u = {(i0, j1): 1, (i1, j1): -1}
        v = {(j1, k0): 1, (j1, k1): 1}
        # c00 += m7
        w = {(k0, i0): 1}
        add_term(u, v, w)

    def strassen_block(rows, cols, j_start):
        strassen_m1(rows, cols, j_start)
        strassen_m2(rows, cols, j_start)
        strassen_m3(rows, cols, j_start)
        strassen_m4(rows, cols, j_start)
        strassen_m5(rows, cols, j_start)
        strassen_m6(rows, cols, j_start)
        strassen_m7(rows, cols, j_start)

    # Schoolbook 1x2x2 terms
    def schoolbook_1x2x2(row, cols, j_start):
        # C[k] = sum_j A[row, j] B[j, k]
        # j in {j0, j1}
        # k in {k0, k1}
        i = row
        k0, k1 = cols
        j0, j1 = j_start, j_start + 1
        
        # Term 1: j0, k0
        u = {(i, j0): 1}
        v = {(j0, k0): 1}
        w = {(k0, i): 1}
        add_term(u, v, w)
        
        # Term 2: j0, k1
        u = {(i, j0): 1}
        v = {(j0, k1): 1}
        w = {(k1, i): 1}
        add_term(u, v, w)
        
        # Term 3: j1, k0
        u = {(i, j1): 1}
        v = {(j1, k0): 1}
        w = {(k0, i): 1}
        add_term(u, v, w)
        
        # Term 4: j1, k1
        u = {(i, j1): 1}
        v = {(j1, k1): 1}
        w = {(k1, i): 1}
        add_term(u, v, w)

    # Block (0,1) rows, (0,1) cols
    strassen_block((0, 1), (0, 1), 0)
    strassen_block((0, 1), (0, 1), 2)
    
    # Block (0,1) rows, (2,3) cols
    strassen_block((0, 1), (2, 3), 0)
    strassen_block((0, 1), (2, 3), 2)
    
    # Block (2) row, (0,1) cols
    schoolbook_1x2x2(2, (0, 1), 0)
    schoolbook_1x2x2(2, (0, 1), 2)
    
    # Block (2) row, (2,3) cols
    schoolbook_1x2x2(2, (2, 3), 0)
    schoolbook_1x2x2(2, (2, 3), 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
