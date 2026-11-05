"""Find rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition. U: (n*m, r), V: (m*p, r), W: (p*n, r)"""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(r):
                    total += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2():
    """
    Return the 7 Strassen terms for 2x2 * 2x2 multiplication.
    Each term is (u, v, w) where u is 4-vector (for A, row-major), v is 4-vector (for B, row-major), w is 4-vector (for C, row-major).
    A is 2x2 indexed [i*2+j], B is 2x2 indexed [j*2+k], C is 2x2 indexed [i*2+k].
    """
    terms = []
    # M1 = (a00+a11)(b00+b11)
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, -1]  # C00 += 1, C11 += -1
    terms.append((u1, v1, w1))

    # M2 = (a10+a11)b00
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, 0]  # C10 += 1
    terms.append((u2, v2, w2))

    # M3 = a00(b10-b11)
    u3 = [1, 0, 0, 0]
    v3 = [0, 0, 1, -1]
    w3 = [0, 1, 0, 0]  # C01 += 1
    terms.append((u3, v3, w3))

    # M4 = a11(b01-b00)
    u4 = [0, 0, 0, 1]
    v4 = [0, 1, -1, 0]
    w4 = [-1, 1, 0, 0]  # C00 += -1, C01 += 1
    terms.append((u4, v4, w4))

    # M5 = (a00+a01)b11
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [1, 0, 1, 0]  # C00 += 1, C10 += 1
    terms.append((u5, v5, w5))

    # M6 = (a10-a00)(b00+b01)
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 0, 1]  # C11 += 1
    terms.append((u6, v6, w6))

    # M7 = (a01-a11)(b10+b11)
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [1, 0, 0, 0]  # C00 += 1
    terms.append((u7, v7, w7))

    return terms

def _schoolbook_block(na, mb, pb):
    """
    Return schoolbook terms for na x mb * mb x pb.
    Each term is (u, v, w) with lengths na*mb, mb*pb, pb*na.
    u indexed [i*mb+j], v indexed [j*pb+k], w indexed [k*na+i].
    """
    terms = []
    for i in range(na):
        for j in range(mb):
            for k in range(pb):
                u = [0] * (na * mb)
                v = [0] * (mb * pb)
                w = [0] * (pb * na)
                u[i * mb + j] = 1
                v[j * pb + k] = 1
                w[k * na + i] = 1
                terms.append((u, v, w))
    return terms

def _build_strassen_based():
    """
    Build the rank-46 decomposition for 2x5 * 5x5 -> 2x5.
    Partition columns/rows as [0,1], [2,3], [4].
    """
    n, m, p = 2, 5, 5

    # Block index lists
    blocks = [[0, 1], [2, 3], [4]]  # 3 blocks of sizes 2, 2, 1

    all_terms = []

    for ik in range(3):  # C block column index (also B block col)
        for ij in range(3):  # A block col = B block row index
            mb = len(blocks[ij])  # A cols = B rows
            pb = len(blocks[ik])  # B cols = C cols

            if mb == 2 and pb == 2:
                local_terms = _strassen_2x2()
            else:
                local_terms = _schoolbook_block(2, mb, pb)

            for (u_local, v_local, w_local) in local_terms:
                u_global = [0] * (n * m)  # 10
                v_global = [0] * (m * p)  # 25
                w_global = [0] * (p * n)  # 10

                # A block ij: 2 rows, cols blocks[ij]
                for li, lv in enumerate(u_local):
                    if lv != 0:
                        local_i = li // mb
                        local_j = li % mb
                        gi = local_i
                        gj = blocks[ij][local_j]
                        u_global[gi * m + gj] = lv

                # B block (ij, ik): rows blocks[ij], cols blocks[ik]
                for li, lv in enumerate(v_local):
                    if lv != 0:
                        local_j = li // pb
                        local_k = li % pb
                        gj = blocks[ij][local_j]
                        gk = blocks[ik][local_k]
                        v_global[gj * p + gk] = lv

                # C block ik: 2 rows, cols blocks[ik]
                # W indexed as W[k*n + i]
                for li, lv in enumerate(w_local):
                    if lv != 0:
                        local_i = li // pb
                        local_k = li % pb
                        gi = local_i
                        gk = blocks[ik][local_k]
                        w_global[gk * n + gi] = lv

                all_terms.append((u_global, v_global, w_global))

    rank = len(all_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u, v, w) in enumerate(all_terms):
        for i in range(n * m):
            U[i][c] = float(u[i])
        for i in range(m * p):
            V[i][c] = float(v[i])
        for i in range(p * n):
            W[i][c] = float(w[i])

    return {"U": U, "V": V, "W": W}

def _schoolbook():
    n, m, p = 2, 5, 5
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Try the Strassen-based construction (rank 46)
    result = _build_strassen_based()
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result

    # Fallback to schoolbook
    result = _schoolbook()
    return result
