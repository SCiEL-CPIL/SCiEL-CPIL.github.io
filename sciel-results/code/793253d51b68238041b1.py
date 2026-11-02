"""Optimized construction for 2x3x8 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8

    # Decomposition strategy:
    # 1. j=0 (A is 2x1, B is 1x8): 16 terms (each output entry is a unique product)
    # 2. j=1,2 (A is 2x2, B is 2x8): Split B into 4 blocks of 2 columns each.
    #    Use Strassen's algorithm (rank 7) for each 2x2x2 block multiplication.
    #    4 blocks * 7 terms = 28 terms.
    # Total rank = 16 + 28 = 44.
    
    rank = 44
    U = np.zeros((n * m, rank), dtype=int)  # 6 x 44
    V = np.zeros((m * p, rank), dtype=int)  # 24 x 44
    W = np.zeros((p * n, rank), dtype=int)  # 16 x 44
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    r = 0
    
    # Part 1: j=0 (middle dimension split: m1=1)
    # C[i,k] = A[i,0] * B[0,k]
    for i in range(n):
        for k in range(p):
            U[u_idx(i, 0), r] = 1
            V[v_idx(0, k), r] = 1
            W[w_idx(k, i), r] = 1
            r += 1
    
    # Part 2: j=1,2 (middle dimension split: m2=2)
    # A is 2x2 (rows i=0,1; cols j=1,2)
    # B is 2x8 (rows j=1,2; cols k=0..7)
    # Split into 4 blocks of 2 columns each
    
    # Strassen 2x2x2 terms:
    # M1 = (a+d)(e+h), W: C00=1, C11=1
    # M2 = (c+d)e,     W: C10=1, C11=-1
    # M3 = a(f-h),     W: C01=1, C11=1
    # M4 = d(g-e),     W: C00=1, C10=1
    # M5 = (a+b)h,     W: C00=-1, C01=1
    # M6 = (c-a)(e+f), W: C11=1
    # M7 = (b-d)(g+h), W: C00=1
    
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Local A mapping: 0->(0,1), 1->(0,2), 2->(1,1), 3->(1,2)
    a_map = {0: (0, 1), 1: (0, 2), 2: (1, 1), 3: (1, 2)}
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Local B mapping: 0->(1,k0), 1->(1,k1), 2->(2,k0), 3->(2,k1)
        b_map = {0: (1, k0), 1: (1, k1), 2: (2, k0), 3: (2, k1)}
        
        # Local C mapping: 0->(0,k0), 1->(0,k1), 2->(1,k0), 3->(1,k1)
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
# EVOLVE-BLOCK-END
