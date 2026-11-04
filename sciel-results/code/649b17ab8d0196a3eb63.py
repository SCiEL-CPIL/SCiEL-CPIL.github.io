import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 27 block multiplications * 7 Strassen multiplications = 189
    rank = 189
    
    # U: (n*m, rank) = (36, 189)
    # V: (m*p, rank) = (36, 189)
    # W: (p*n, rank) = (36, 189)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen coefficients for 2x2 multiplication
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Coeffs for A part (a,b,c,d)
    # Indices: 0:a, 1:b, 2:c, 3:d
    A_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6 (c-a) -> -1*a + 1*c
        [0, 1, 0, -1]  # M7 (b-d) -> 1*b - 1*d
    ]
    
    # Coeffs for B part (e,f,g,h)
    # Indices: 0:e, 1:f, 2:g, 3:h
    B_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3 (f-h)
        [0, 0, 1, -1], # M4 (g-e)
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6 (e+f)
        [0, 0, 1, 1]  # M7 (g+h)
    ]
    
    # Coeffs for C part (p,q,r,s) from M1..M7
    # C = [[p, q], [r, s]]
    # p = M1 + M4 - M5 + M7
    # q = M3 + M5
    # r = M2 + M4
    # s = M1 - M2 + M3 + M6
    # Indices: 0:p, 1:q, 2:r, 3:s
    # Rows are M1..M7, Cols are p,q,r,s
    C_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    col = 0
    # Iterate over block indices I, J, K
    # I: block row of A, block row of C
    # J: block col of A, block row of B
    # K: block col of B, block col of C
    for I in range(3):
        for J in range(3):
            for K in range(3):
                for m_idx in range(7):
                    # Set U coefficients
                    # A block is at (I, J). Local (u, v).
                    # Global i = 2*I + u, j = 2*J + v.
                    # U index: i * m + j
                    for u in range(2):
                        for v in range(2):
                            # Local index in A_coeffs: u*2 + v
                            # 0:00, 1:01, 2:10, 3:11
                            local_idx = u * 2 + v
                            coeff = A_coeffs[m_idx][local_idx]
                            if coeff != 0:
                                i_global = 2 * I + u
                                j_global = 2 * J + v
                                U[i_global * m + j_global, col] = coeff
                    
                    # Set V coefficients
                    # B block is at (J, K). Local (v, w).
                    # Global j = 2*J + v, k = 2*K + w.
                    # V index: j * p + k
                    for v in range(2):
                        for w in range(2):
                            local_idx = v * 2 + w
                            coeff = B_coeffs[m_idx][local_idx]
                            if coeff != 0:
                                j_global = 2 * J + v
                                k_global = 2 * K + w
                                V[j_global * p + k_global, col] = coeff
                    
                    # Set W coefficients
                    # C block is at (I, K). Local (u, w).
                    # Global i = 2*I + u, k = 2*K + w.
                    # W index: k * n + i
                    for u in range(2):
                        for w in range(2):
                            local_idx = u * 2 + w
                            coeff = C_coeffs[m_idx][local_idx]
                            if coeff != 0:
                                i_global = 2 * I + u
                                k_global = 2 * K + w
                                W[k_global * n + i_global, col] = coeff
                    
                    col += 1
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
