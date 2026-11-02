"""Optimized construction for 2x3x8 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    
    # Rank calculation:
    # Split m=3 into m1=1 and m2=2.
    # Part 1: 2x1x8 multiplication. Rank = 2*8 = 16.
    # Part 2: 2x2x8 multiplication.
    #   2x2x8 is 4 blocks of 2x2x2.
    #   Each 2x2x2 uses Strassen (rank 7).
    #   Total rank for Part 2 = 4 * 7 = 28.
    # Total rank = 16 + 28 = 44.
    
    rank = 44
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    r = 0
    
    # --- Part 1: 2x1x8 ---
    # A is 2x1 (j=0), B is 1x8 (j=0), C is 2x8.
    # C[i,k] = A[i,0] * B[0,k]
    for i in range(n):
        for k in range(p):
            U[u_idx(i, 0), r] = 1
            V[v_idx(0, k), r] = 1
            W[w_idx(k, i), r] = 1
            r += 1
            
    # --- Part 2: 2x2x8 ---
    # A is 2x2 (j=1,2), B is 2x8 (j=1,2), C is 2x8.
    # Split B and C into 4 blocks of width 2.
    
    # Strassen coefficients for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h) -> C00+=1, C11+=1
    # M2 = (c+d)e    -> C10+=1, C11-=1
    # M3 = a(f-h)    -> C01+=1, C11+=1
    # M4 = d(g-e)    -> C00+=1, C10+=1
    # M5 = (a+b)h    -> C00-=1, C01+=1
    # M6 = (c-a)(e+f)-> C11+=1
    # M7 = (b-d)(g+h)-> C00+=1
    
    # Local indices:
    # A: 0:a, 1:b, 2:c, 3:d
    # B: 0:e, 1:f, 2:g, 3:h
    # C: 0:C00, 1:C01, 2:C10, 3:C11
    
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),       # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),       # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),       # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),       # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),       # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),       # M7
    ]
    
    # Mapping local A indices to global (i, j)
    # a=A[0,1], b=A[0,2], c=A[1,1], d=A[1,2]
    a_map = {0: (0, 1), 1: (0, 2), 2: (1, 1), 3: (1, 2)}
    
    for b_idx in range(4):
        k0 = 2 * b_idx
        k1 = 2 * b_idx + 1
        
        # Mapping local B indices to global (j, k)
        # e=B[1,k0], f=B[1,k1], g=B[2,k0], h=B[2,k1]
        b_map = {0: (1, k0), 1: (1, k1), 2: (2, k0), 3: (2, k1)}
        
        # Mapping local C indices to global (k, i) -> (i, k) for W index
        # C00=C[0,k0], C01=C[0,k1], C10=C[1,k0], C11=C[1,k1]
        c_map = {0: (k0, 0), 1: (k1, 0), 2: (k0, 1), 3: (k1, 1)}
        
        for u_c, v_c, w_c in strassen_terms:
            for loc_a, coeff in u_c.items():
                gi, gj = a_map[loc_a]
                U[u_idx(gi, gj), r] += coeff
            
            for loc_b, coeff in v_c.items():
                gj, gk = b_map[loc_b]
                V[v_idx(gj, gk), r] += coeff
            
            for loc_c, coeff in w_c.items():
                gk, gi = c_map[loc_c]
                W[w_idx(gk, gi), r] += coeff
            
            r += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
