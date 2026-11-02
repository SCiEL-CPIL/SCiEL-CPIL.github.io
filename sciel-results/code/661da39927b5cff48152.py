"""Optimized construction for 2x3x8 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    
    # New approach: Split the output dimension p=8 into 4 blocks of size 2.
    # Each block corresponds to a 2x3x2 matrix multiplication.
    # We use a rank-11 decomposition for each 2x3x2 block.
    # Total rank = 4 * 11 = 44.
    # This is a different approach from the previous middle-dimension split.
    
    rank = 44
    U = np.zeros((n * m, rank), dtype=int)  # 6 x 44
    V = np.zeros((m * p, rank), dtype=int)  # 24 x 44
    W = np.zeros((p * n, rank), dtype=int)  # 16 x 44
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    r = 0
    
    # Define the rank-11 decomposition for 2x3x2.
    # A (2x3): a,b,c / d,e,f
    # B (3x2): g,h / i,j / k,l
    # C (2x2): C00, C01 / C10, C11
    
    # Decomposition:
    # 1. Strassen for top-left 2x2 of A (a,b,d,e) and top-left 2x2 of B (g,h,i,j)
    #    This computes C00_partial, C01_partial, C10_partial, C11_partial
    #    using 7 terms.
    # 2. Outer product of A_right (c,f) and B_bottom (k,l)
    #    This computes C00_extra, C01_extra, C10_extra, C11_extra
    #    using 4 terms.
    
    # Strassen terms for 2x2x2 (local indices 0:a, 1:b, 2:d, 3:e for A; 0:g, 1:h, 2:i, 3:j for B)
    # Note: Standard Strassen uses (a,b,c,d) for A and (e,f,g,h) for B.
    # Here A is [[a,b],[d,e]], B is [[g,h],[i,j]].
    # M1 = (a+e)(g+j)
    # M2 = (d+e)g
    # M3 = a(h-j)
    # M4 = e(i-g)
    # M5 = (a+b)j
    # M6 = (d-a)(g+h)
    # M7 = (b-e)(i+j)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_2x2 = [
        # (U_coeffs {a:0, b:1, d:2, e:3}, V_coeffs {g:0, h:1, i:2, j:3}, W_coeffs {C00:0, C01:1, C10:2, C11:3})
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),       # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),       # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),       # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),       # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),       # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),       # M7
    ]
    
    # Extra terms for c,f and k,l
    # P1 = ck
    # P2 = cl
    # P3 = fk
    # P4 = fl
    extra_terms = [
        ({4: 1}, {4: 1}, {0: 1}), # c * k -> C00
        ({4: 1}, {5: 1}, {1: 1}), # c * l -> C01
        ({5: 1}, {4: 1}, {2: 1}), # f * k -> C10
        ({5: 1}, {5: 1}, {3: 1}), # f * l -> C11
    ]
    
    # Total 11 terms per block.
    # Local A indices: 0:a, 1:b, 2:d, 3:e, 4:c, 5:f
    # Local B indices: 0:g, 1:h, 2:i, 3:j, 4:k, 5:l
    # Local C indices: 0:C00, 1:C01, 2:C10, 3:C11
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Map local A to global
        # a: i=0, j=0
        # b: i=0, j=1
        # c: i=0, j=2
        # d: i=1, j=0
        # e: i=1, j=1
        # f: i=1, j=2
        a_map = {
            0: (0, 0),
            1: (0, 1),
            2: (1, 0),
            3: (1, 1),
            4: (0, 2),
            5: (1, 2)
        }
        
        # Map local B to global for this block
        # g: j=0, k=k0
        # h: j=0, k=k1
        # i: j=1, k=k0
        # j: j=1, k=k1
        # k: j=2, k=k0
        # l: j=2, k=k1
        b_map = {
            0: (0, k0),
            1: (0, k1),
            2: (1, k0),
            3: (1, k1),
            4: (2, k0),
            5: (2, k1)
        }
        
        # Map local C to global for this block
        # C00: i=0, k=k0
        # C01: i=0, k=k1
        # C10: i=1, k=k0
        # C11: i=1, k=k1
        c_map = {
            0: (0, k0),
            1: (0, k1),
            2: (1, k0),
            3: (1, k1)
        }
        
        # Apply Strassen terms
        for u_c, v_c, w_c in strassen_2x2:
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
            
        # Apply extra terms
        for u_c, v_c, w_c in extra_terms:
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
