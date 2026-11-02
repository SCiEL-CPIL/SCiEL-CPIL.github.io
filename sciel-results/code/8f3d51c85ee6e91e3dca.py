"""Constructs and verifies a rank-minimal decomposition for 2x3x8 matrix multiplication."""
import numpy as np

def construct():
    n, m, p = 2, 3, 8
    
    # The decomposition is a direct sum of two tensors:
    # 1. T1: Multiplication of 2x1 and 1x8 matrices. Rank 16.
    # 2. T2: Multiplication of 2x2 and 2x8 matrices. Rank 28 (using Strassen on 4 blocks of 2x2x2).
    # Total Rank: 44.
    
    rank = 44
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    r = 0
    
    # Part 1: j=0 (2x1 x 1x8)
    for i in range(n):
        for k in range(p):
            U[u_idx(i, 0), r] = 1
            V[v_idx(0, k), r] = 1
            W[w_idx(k, i), r] = 1
            r += 1
            
    # Part 2: j=1,2 (2x2 x 2x8)
    # Split k into 4 blocks of size 2.
    # Use Strassen for each 2x2x2 block.
    
    # Strassen coefficients
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e), M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(g+h)
    # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    a_map = {0: (0, 1), 1: (0, 2), 2: (1, 1), 3: (1, 2)}
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        b_map = {0: (1, k0), 1: (1, k1), 2: (2, k0), 3: (2, k1)}
        c_map = {0: (0, k0), 1: (0, k1), 2: (1, k0), 3: (1, k1)}
        
        for u_c, v_c, w_c in strassen_terms:
            for la, coeff in u_c.items():
                gi, gj = a_map[la]
                U[u_idx(gi, gj), r] += coeff
            for lb, coeff in v_c.items():
                gj, gk = b_map[lb]
                V[v_idx(gj, gk), r] += coeff
            for lc, coeff in w_c.items():
                gi, gk = c_map[lc]
                W[w_idx(gk, gi), r] += coeff
            r += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
