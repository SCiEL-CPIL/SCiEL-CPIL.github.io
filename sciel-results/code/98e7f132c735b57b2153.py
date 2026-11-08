import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 4, 7
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))

    def strassen_2x2_block(i_off, j_off, k_off):
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        
        # M1 = (A11+A22)*(B11+B22)
        u[i_off * m + j_off] = 1
        u[(i_off+1) * m + (j_off+1)] = 1
        v[j_off * p + k_off] = 1
        v[(j_off+1) * p + (k_off+1)] = 1
        w[k_off * n + i_off] = 1
        w[(k_off+1) * n + (i_off+1)] = 1
        add_term(u, v, w)

        # M2 = (A21+A22)*B11
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = 1
        u[(i_off+1) * m + (j_off+1)] = 1
        v[j_off * p + k_off] = 1
        w[k_off * n + (i_off+1)] = 1
        w[(k_off+1) * n + (i_off+1)] = -1
        add_term(u, v, w)

        # M3 = A11*(B12-B22)
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        v[j_off * p + (k_off+1)] = 1
        v[(j_off+1) * p + (k_off+1)] = -1
        w[(k_off+1) * n + i_off] = 1
        w[(k_off+1) * n + (i_off+1)] = 1
        add_term(u, v, w)

        # M4 = A22*(B21-B11)
        u = [0] * (n * m)
        u[(i_off+1) * m + (j_off+1)] = 1
        v[(j_off+1) * p + k_off] = 1
        v[j_off * p + k_off] = -1
        w[k_off * n + i_off] = 1
        w[k_off * n + (i_off+1)] = 1
        add_term(u, v, w)

        # M5 = (A11+A12)*B22
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        u[i_off * m + (j_off+1)] = 1
        v[(j_off+1) * p + (k_off+1)] = 1
        w[(k_off+1) * n + i_off] = 1
        w[k_off * n + i_off] = -1
        add_term(u, v, w)

        # M6 = (A21-A11)*(B11+B12)
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = 1
        u[i_off * m + j_off] = -1
        v[j_off * p + k_off] = 1
        v[j_off * p + (k_off+1)] = 1
        w[(k_off+1) * n + (i_off+1)] = 1
        add_term(u, v, w)

        # M7 = (A12-A22)*(B21+B22)
        u = [0] * (n * m)
        u[i_off * m + (j_off+1)] = 1
        u[(i_off+1) * m + (j_off+1)] = -1
        v[(j_off+1) * p + k_off] = 1
        v[(j_off+1) * p + (k_off+1)] = 1
        w[k_off * n + i_off] = 1
        add_term(u, v, w)

    def strassen_4x4_head():
        for i_off in range(0, 4, 2):
            for j_off in range(0, 4, 2):
                for k_off in range(0, 4, 2):
                    strassen_2x2_block(i_off, j_off, k_off)

    def tail_4x4x2():
        for i_off in range(0, 4, 2):
            for j_off in range(0, 4, 2):
                strassen_2x2_block(i_off, j_off, 4)

    def tail_4x4x1():
        for i in range(4):
            for j in range(4):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + 6] = 1
                w = [0] * (p * n)
                w[6 * n + i] = 1
                add_term(u, v, w)

    strassen_4x4_head()
    tail_4x4x2()
    tail_4x4x1()
    
    if time.time() > deadline:
        return {"U": [], "V": [], "W": []}

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            if u[r] != 0:
                U[r][c] = u[r]
        for r in range(m * p):
            if v[r] != 0:
                V[r][c] = v[r]
        for r in range(p * n):
            if w[r] != 0:
                W[r][c] = w[r]

    return {"U": U, "V": V, "W": W}
