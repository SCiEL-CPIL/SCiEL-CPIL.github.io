import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    DEADLINE = 140

    n, m, p = DIMENSIONS
    terms = []

    def add_strassen_block(i0, j0, k0):
        a11_u = i0 * m + j0
        a12_u = i0 * m + (j0 + 1)
        a21_u = (i0 + 1) * m + j0
        a22_u = (i0 + 1) * m + (j0 + 1)
        
        b11_v = j0 * p + k0
        b12_v = j0 * p + (k0 + 1)
        b21_v = (j0 + 1) * p + k0
        b22_v = (j0 + 1) * p + (k0 + 1)
        
        c11_w = k0 * n + i0
        c12_w = (k0 + 1) * n + i0
        c21_w = k0 * n + (i0 + 1)
        c22_w = (k0 + 1) * n + (i0 + 1)
        
        u1 = np.zeros(n * m); u1[a11_u] = 1.0; u1[a22_u] = 1.0
        v1 = np.zeros(m * p); v1[b11_v] = 1.0; v1[b22_v] = 1.0
        w1 = np.zeros(p * n); w1[c11_w] = 1.0; w1[c22_w] = 1.0
        terms.append((u1, v1, w1))
        
        u2 = np.zeros(n * m); u2[a21_u] = 1.0; u2[a22_u] = 1.0
        v2 = np.zeros(m * p); v2[b11_v] = 1.0
        w2 = np.zeros(p * n); w2[c21_w] = 1.0; w2[c22_w] = -1.0
        terms.append((u2, v2, w2))
        
        u3 = np.zeros(n * m); u3[a11_u] = 1.0
        v3 = np.zeros(m * p); v3[b12_v] = 1.0; v3[b22_v] = -1.0
        w3 = np.zeros(p * n); w3[c12_w] = 1.0; w3[c22_w] = 1.0
        terms.append((u3, v3, w3))
        
        u4 = np.zeros(n * m); u4[a22_u] = 1.0
        v4 = np.zeros(m * p); v4[b21_v] = 1.0; v4[b11_v] = -1.0
        w4 = np.zeros(p * n); w4[c11_w] = 1.0; w4[c21_w] = 1.0
        terms.append((u4, v4, w4))
        
        u5 = np.zeros(n * m); u5[a11_u] = 1.0; u5[a12_u] = 1.0
        v5 = np.zeros(m * p); v5[b22_v] = 1.0
        w5 = np.zeros(p * n); w5[c11_w] = -1.0; w5[c12_w] = 1.0
        terms.append((u5, v5, w5))
        
        u6 = np.zeros(n * m); u6[a21_u] = 1.0; u6[a11_u] = -1.0
        v6 = np.zeros(m * p); v6[b11_v] = 1.0; v6[b12_v] = 1.0
        w6 = np.zeros(p * n); w6[c22_w] = 1.0
        terms.append((u6, v6, w6))
        
        u7 = np.zeros(n * m); u7[a12_u] = 1.0; u7[a22_u] = -1.0
        v7 = np.zeros(m * p); v7[b21_v] = 1.0; v7[b22_v] = 1.0
        w7 = np.zeros(p * n); w7[c11_w] = 1.0
        terms.append((u7, v7, w7))

    def add_schoolbook_row(i_row, j0, k0, j_range, k_range):
        for j in range(j0, j0 + j_range):
            for k in range(k0, k0 + k_range):
                u_idx = i_row * m + j
                v_idx = j * p + k
                w_idx = k * n + i_row
                u_vec = np.zeros(n * m); u_vec[u_idx] = 1.0
                v_vec = np.zeros(m * p); v_vec[v_idx] = 1.0
                w_vec = np.zeros(p * n); w_vec[w_idx] = 1.0
                terms.append((u_vec, v_vec, w_vec))

    # Rows 0-1: 4 Strassen 2x2x2 blocks
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            if time.time() - start_time > DEADLINE:
                break
            add_strassen_block(i0=0, j0=j0, k0=k0)

    # Row 2: schoolbook 1x4 x 4x8
    add_schoolbook_row(2, 0, 0, 4, 8)

    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for r in range(rank):
        u_vec, v_vec, w_vec = terms[r]
        U[:, r] = u_vec
        V[:, r] = v_vec
        W[:, r] = w_vec

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
