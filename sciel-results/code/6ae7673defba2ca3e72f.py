import time

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    C[i,j] = sum_k A[i,k] * B[k,j]
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Convention: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    Tensor: T[i*4+j, j*4+k, k*4+i] = sum_r U[i*4+j,r] * V[j*4+k,r] * W[k*4+i,r] = 1
    """
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s kill

    n, m, p = 4, 4, 4
    rank = 49

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def get_U_index(row, col):
        return row * m + col

    def get_V_index(row, col):
        return row * p + col

    def get_W_index(row, col):
        return col * n + row

    def a_elem(bi, bj, r, c):
        return (2 * bi + r, 2 * bj + c)

    def b_elem(bi, bj, r, c):
        return (2 * bi + r, 2 * bj + c)

    def c_elem(bi, bj, r, c):
        return (2 * bi + r, 2 * bj + c)

    L_terms = []
    R_terms = []
    contrib_terms = []

    # Term 0: P1 = (A11 + A22)(B11 + B22)
    L0 = [[{} for _ in range(2)] for _ in range(2)]
    R0 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            ar, ac = a_elem(0, 0, r, c)
            dr, dc = a_elem(1, 1, r, c)
            L0[r][c] = {(ar, ac): 1, (dr, dc): 1}
            er, ec = b_elem(0, 0, r, c)
            hr, hc = b_elem(1, 1, r, c)
            R0[r][c] = {(er, ec): 1, (hr, hc): 1}
    contrib0 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = c_elem(0, 0, r, c)
            contrib0[(r, c)] = [(cr, cc, 1)]
            cr2, cc2 = c_elem(1, 1, r, c)
            contrib0[(r, c)].append((cr2, cc2, 1))
    L_terms.append(L0)
    R_terms.append(R0)
    contrib_terms.append(contrib0)

    # Term 1: P2 = (A21 + A22)B11
    L1 = [[{} for _ in range(2)] for _ in range(2)]
    R1 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            ar, ac = a_elem(1, 0, r, c)
            dr, dc = a_elem(1, 1, r, c)
            L1[r][c] = {(ar, ac): 1, (dr, dc): 1}
            er, ec = b_elem(0, 0, r, c)
            R1[r][c] = {(er, ec): 1}
    contrib1 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = c_elem(1, 0, r, c)
            contrib1[(r, c)] = [(cr, cc, 1)]
            cr2, cc2 = c_elem(1, 1, r, c)
            contrib1[(r, c)].append((cr2, cc2, -1))
    L_terms.append(L1)
    R_terms.append(R1)
    contrib_terms.append(contrib1)

    # Term 2: P3 = A11(B12 - B22)
    L2 = [[{} for _ in range(2)] for _ in range(2)]
    R2 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            ar, ac = a_elem(0, 0, r, c)
            L2[r][c] = {(ar, ac): 1}
            fr, fc = b_elem(0, 1, r, c)
            hr, hc = b_elem(1, 1, r, c)
            R2[r][c] = {(fr, fc): 1, (hr, hc): -1}
    contrib2 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = c_elem(0, 1, r, c)
            contrib2[(r, c)] = [(cr, cc, 1)]
            cr2, cc2 = c_elem(1, 1, r, c)
            contrib2[(r, c)].append((cr2, cc2, 1))
    L_terms.append(L2)
    R_terms.append(R2)
    contrib_terms.append(contrib2)

    # Term 3: P4 = A22(B21 - B11)
    L3 = [[{} for _ in range(2)] for _ in range(2)]
    R3 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            dr, dc = a_elem(1, 1, r, c)
            L3[r][c] = {(dr, dc): 1}
            gr, gc = b_elem(1, 0, r, c)
            er, ec = b_elem(0, 0, r, c)
            R3[r][c] = {(gr, gc): 1, (er, ec): -1}
    contrib3 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = c_elem(0, 0, r, c)
            contrib3[(r, c)] = [(cr, cc, 1)]
            cr2, cc2 = c_elem(1, 0, r, c)
            contrib3[(r, c)].append((cr2, cc2, 1))
    L_terms.append(L3)
    R_terms.append(R3)
    contrib_terms.append(contrib3)

    # Term 4: P5 = (A11 + A12)B22
    L4 = [[{} for _ in range(2)] for _ in range(2)]
    R4 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            ar, ac = a_elem(0, 0, r, c)
            br, bc = a_elem(0, 1, r, c)
            L4[r][c] = {(ar, ac): 1, (br, bc): 1}
            hr, hc = b_elem(1, 1, r, c)
            R4[r][c] = {(hr, hc): 1}
    contrib4 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = c_elem(0, 0, r, c)
            contrib4[(r, c)] = [(cr, cc, -1)]
            cr2, cc2 = c_elem(0, 1, r, c)
            contrib4[(r, c)].append((cr2, cc2, 1))
    L_terms.append(L4)
    R_terms.append(R4)
    contrib_terms.append(contrib4)

    # Term 5: P6 = (A21 - A11)(B11 + B12)
    L5 = [[{} for _ in range(2)] for _ in range(2)]
    R5 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            cr, cc = a_elem(1, 0, r, c)
            ar, ac = a_elem(0, 0, r, c)
            L5[r][c] = {(cr, cc): 1, (ar, ac): -1}
            er, ec = b_elem(0, 0, r, c)
            fr, fc = b_elem(0, 1, r, c)
            R5[r][c] = {(er, ec): 1, (fr, fc): 1}
    contrib5 = {}
    for r in range(2):
        for c in range(2):
            cr2, cc2 = c_elem(1, 1, r, c)
            contrib5[(r, c)] = [(cr2, cc2, 1)]
    L_terms.append(L5)
    R_terms.append(R5)
    contrib_terms.append(contrib5)

    # Term 6: P7 = (A12 - A22)(B21 + B22)
    L6 = [[{} for _ in range(2)] for _ in range(2)]
    R6 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            br, bc = a_elem(0, 1, r, c)
            dr, dc = a_elem(1, 1, r, c)
            L6[r][c] = {(br, bc): 1, (dr, dc): -1}
            gr, gc = b_elem(1, 0, r, c)
            hr, hc = b_elem(1, 1, r, c)
            R6[r][c] = {(gr, gc): 1, (hr, hc): 1}
    contrib6 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = c_elem(0, 0, r, c)
            contrib6[(r, c)] = [(cr, cc, 1)]
    L_terms.append(L6)
    R_terms.append(R6)
    contrib_terms.append(contrib6)

    # Inner Strassen sub-contributions
    sub_contrib = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]

    def add_to_dict(target, source, sign=1):
        for k, v in source.items():
            target[k] = target.get(k, 0) + sign * v

    for t in range(7):
        if time.time() - start_time > DEADLINE:
            break
        L_t = L_terms[t]
        R_t = R_terms[t]
        contrib_t = contrib_terms[t]

        a = L_t[0][0]
        b = L_t[0][1]
        c = L_t[1][0]
        d = L_t[1][1]
        e = R_t[0][0]
        f = R_t[0][1]
        g = R_t[1][0]
        h = R_t[1][1]

        sub_products = []

        # M1: (a+d)(e+h)
        x1 = {}
        add_to_dict(x1, a, 1)
        add_to_dict(x1, d, 1)
        y1 = {}
        add_to_dict(y1, e, 1)
        add_to_dict(y1, h, 1)
        sub_products.append((x1, y1))

        # M2: (c+d)e
        x2 = {}
        add_to_dict(x2, c, 1)
        add_to_dict(x2, d, 1)
        y2 = dict(e)
        sub_products.append((x2, y2))

        # M3: a(f-h)
        x3 = dict(a)
        y3 = {}
        add_to_dict(y3, f, 1)
        add_to_dict(y3, h, -1)
        sub_products.append((x3, y3))

        # M4: d(g-e)
        x4 = dict(d)
        y4 = {}
        add_to_dict(y4, g, 1)
        add_to_dict(y4, e, -1)
        sub_products.append((x4, y4))

        # M5: (a+b)h
        x5 = {}
        add_to_dict(x5, a, 1)
        add_to_dict(x5, b, 1)
        y5 = dict(h)
        sub_products.append((x5, y5))

        # M6: (c-a)(e+f)
        x6 = {}
        add_to_dict(x6, c, 1)
        add_to_dict(x6, a, -1)
        y6 = {}
        add_to_dict(y6, e, 1)
        add_to_dict(y6, f, 1)
        sub_products.append((x6, y6))

        # M7: (b-d)(g+h)
        x7 = {}
        add_to_dict(x7, b, 1)
        add_to_dict(x7, d, -1)
        y7 = {}
        add_to_dict(y7, g, 1)
        add_to_dict(y7, h, 1)
        sub_products.append((x7, y7))

        for k in range(7):
            if time.time() - start_time > DEADLINE:
                break
            x_k, y_k = sub_products[k]
            rank_idx = t * 7 + k

            for (row, col), coeff in x_k.items():
                if coeff != 0:
                    U[get_U_index(row, col)][rank_idx] += coeff

            for (row, col), coeff in y_k.items():
                if coeff != 0:
                    V[get_V_index(row, col)][rank_idx] += coeff

            for (pr, pc, sub_coeff) in sub_contrib[k]:
                if (pr, pc) in contrib_t:
                    for (out_row, out_col, block_coeff) in contrib_t[(pr, pc)]:
                        W[get_W_index(out_row, out_col)][rank_idx] += sub_coeff * block_coeff

    return {"U": U, "V": V, "W": W}
