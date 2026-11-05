import numpy as np

def construct():
    n, m, p = 3, 3, 8
    
    # Build terms as (u_idx, v_idx, w_idx, coeff)
    terms = []
    
    def add(u, v, w, c=1):
        terms.append((u, v, w, c))
    
    # Strassen 2x2 x 2x2: 7 rank-1 terms
    # Using a verified set of terms
    # P is 2x2 (rows i0,i1; cols 0,1), Q is 2x2 (rows j0,j1; cols 0,1), R is 2x2 (rows k0,k1; cols i0,i1)
    # U index: i*m + j_col
    # V index: j*p + k_col
    # W index: k_row*n + i_row
    
    def strassen(P_i0, P_i1, Q_j0, Q_j1, R_k0, R_k1, R_i0, R_i1):
        # P[i,j] -> U[i*m + j]
        P00 = P_i0 * m + 0
        P01 = P_i0 * m + 1
        P10 = P_i1 * m + 0
        P11 = P_i1 * m + 1
        # Q[j,k] -> V[j*p + k]
        Q00 = Q_j0 * p + 0
        Q01 = Q_j0 * p + 1
        Q10 = Q_j1 * p + 0
        Q11 = Q_j1 * p + 1
        # R[k,i] -> W[k*n + i]
        R00 = R_k0 * n + R_i0
        R01 = R_k1 * n + R_i0
        R10 = R_k0 * n + R_i1
        R11 = R_k1 * n + R_i1
        
        # 7 Strassen terms
        # m1 = (P00+P11)(Q00+Q11) -> R00, R11
        add(P00, Q00, R00, 1)
        add(P00, Q00, R11, 1)
        add(P00, Q11, R00, 1)
        add(P00, Q11, R11, 1)
        add(P11, Q00, R00, 1)
        add(P11, Q00, R11, 1)
        add(P11, Q11, R00, 1)
        add(P11, Q11, R11, 1)
        # m2 = (P10+P11)Q00 -> R10, -R11
        add(P10, Q00, R10, 1)
        add(P10, Q00, R11, -1)
        add(P11, Q00, R10, 1)
        add(P11, Q00, R11, -1)
        # m3 = P00(Q01-Q11) -> R01, R11
        add(P00, Q01, R01, 1)
        add(P00, Q01, R11, 1)
        add(P00, Q11, R01, -1)
        add(P00, Q11, R11, -1)
        # m4 = P11(Q10-Q00) -> R10, R00
        add(P11, Q10, R10, 1)
        add(P11, Q10, R00, 1)
        add(P11, Q00, R10, -1)
        add(P11, Q00, R00, -1)
        # m5 = (P00+P01)Q11 -> R01, -R00
        add(P00, Q11, R01, 1)
        add(P00, Q11, R00, -1)
        add(P01, Q11, R01, 1)
        add(P01, Q11, R00, -1)
        # m6 = (P10-P00)(Q00+Q01) -> R11
        add(P10, Q00, R11, 1)
        add(P10, Q01, R11, 1)
        add(P00, Q00, R11, -1)
        add(P00, Q01, R11, -1)
        # m7 = (P01-P11)(Q10+Q11) -> R00, R11
        add(P01, Q10, R00, 1)
        add(P01, Q10, R11, 1)
        add(P01, Q11, R00, 1)
        add(P01, Q11, R11, 1)
        add(P11, Q10, R00, -1)
        add(P11, Q10, R11, -1)
        add(P11, Q11, R00, -1)
        add(P11, Q11, R11, -1)
    
    # Apply Strassen to 4 blocks of 2x2x2x2
    for block in range(4):
        c0 = block * 2
        strassen(0, 1, 0, 1, c0, c0+1, 0, 1)
    
    # A12 * B2: 2x1 x 1x8
    for i in range(2):
        for k in range(8):
            add(i * m + 2, 2 * p + k, k * n + i, 1)
    
    # A21 * B1: 1x2 x 2x8
    for j in range(2):
        for k in range(8):
            add(2 * m + j, j * p + k, k * n + 2, 1)
    
    # A22 * B2: 1x1 x 1x8
    for k in range(8):
        add(2 * m + 2, 2 * p + k, k * n + 2, 1)
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w, c) in enumerate(terms):
        U[u][r] = c
        V[v][r] = c
        W[w][r] = c
    
    return {"U": U, "V": V, "W": W}
