"""Structural redesign: tiled Strassen decomposition of the 3x4x7 product tensor."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def _strassen_terms(n, m, p, i0, j0, k0):
    """7 rank-1 terms for a 2x2x2 Strassen block (A rows i0,i0+1; B cols j0,j0+1; C rows k0,k0+1)."""
    i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
    A00, A01, A10, A11 = i0 * m + j0, i0 * m + j1, i1 * m + j0, i1 * m + j1
    B00, B01, B10, B11 = j0 * p + k0, j0 * p + k1, j1 * p + k0, j1 * p + k1
    C00, C01, C10, C11 = k0 * n + i0, k0 * n + i1, k1 * n + i0, k1 * n + i1
    # m1=(a00+a11)(b00+b11); m2=(a10+a11)b00; m3=a00(b01-b11); m4=a11(b10-b00)
    # m5=(a00+a01)b11; m6=(a10-a00)(b00+b10); m7=(a01-a11)(b01+b11)
    # c00=m1+m4-m5-m7; c01=m3+m5; c10=m2+m4; c11=m1-m2+m3+m6
    return [
        ([(A00, 1), (A11, 1)], [(B00, 1), (B11, 1)], [(C00, 1), (C11, 1)]),   # m1
        ([(A10, 1), (A11, 1)], [(B00, 1)], [(C10, 1)]),                        # m2
        ([(A00, 1)], [(B01, 1), (B11, -1)], [(C01, 1)]),                       # m3
        ([(A11, 1)], [(B10, 1), (B00, -1)], [(C00, 1), (C10, 1)]),             # m4
        ([(A00, 1), (A01, 1)], [(B11, 1)], [(C00, -1)]),                       # m5
        ([(A10, 1), (A00, -1)], [(B00, 1), (B10, 1)], [(C11, 1)]),             # m6
        ([(A01, 1), (A11, -1)], [(B01, 1), (B11, 1)], [(C00, -1)]),            # m7
    ]


def _school_terms(n, m, p, i0, j0, k0, ia, jb, kc):
    """Schoolbook terms for a ia x jb x kc block."""
    terms = []
    for a in range(ia):
        for b in range(jb):
            for c in range(kc):
                terms.append(([((i0 + a) * m + (j0 + b), 1)],
                              [((j0 + b) * p + (k0 + c), 1)],
                              [((k0 + c) * n + (i0 + a), 1)]))
    return terms


def _assemble(all_terms, n, m, p):
    rank = len(all_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for col, (u, v, w) in enumerate(all_terms):
        for idx, val in u:
            U[idx][col] = val
        for idx, val in v:
            V[idx][col] = val
        for idx, val in w:
            W[idx][col] = val
    return U, V, W


def _valid(U, V, W, n, m, p):
    """Exact check: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] == 1 for all i,j,k."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                ui = U[i * m + j]
                vk = V[j * p + k]
                wk = W[k * n + i]
                for c in range(rank):
                    s += ui[c] * vk[c] * wk[c]
                if s != 1:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS

    # Tiling: rows {0,1}|{2}, cols {0,1}|{2,3}, outputs {0,1}|{2,3}|{4,5}|{6}.
    i_groups = [(0, 2), (2, 1)]
    j_groups = [(0, 2), (2, 2)]
    k_groups = [(0, 2), (2, 2), (4, 2), (6, 1)]

    terms = []
    for (i0, ia) in i_groups:
        for (j0, jb) in j_groups:
            for (k0, kc) in k_groups:
                if ia == 2 and jb == 2 and kc == 2:
                    terms.extend(_strassen_terms(n, m, p, i0, j0, k0))
                else:
                    terms.extend(_school_terms(n, m, p, i0, j0, k0, ia, jb, kc))

    U, V, W = _assemble(terms, n, m, p)
    if _valid(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Safe fallback: pure schoolbook (always valid).
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append(([(i * m + j, 1)], [(j * p + k, 1)], [(k * n + i, 1)]))
    U, V, W = _assemble(terms, n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
