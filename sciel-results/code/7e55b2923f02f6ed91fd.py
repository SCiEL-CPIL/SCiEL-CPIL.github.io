import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, val in u_coeffs.items():
            if val != 0:
                U[idx][col] += val
        for idx, val in v_coeffs.items():
            if val != 0:
                V[idx][col] += val
        for idx, val in w_coeffs.items():
            if val != 0:
                W[idx][col] += val
        col += 1

    # Strassen 2x2x2 coefficients
    # u: {0:1, 3:1} etc. 
    # Indices for 2x2 A: 0,1,2,3 -> (0,0), (0,1), (1,0), (1,1)
    # Indices for 2x2 B: 0,1,2,3 -> (0,0), (0,1), (1,0), (1,1)
    # Indices for 2x2 C: 0,1,2,3 -> (0,0), (0,1), (1,0), (1,1)
    
    strassen_terms = [
        # m1: (x00+x11)(y00+y11) -> z00+1, z11+1
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        # m2: (x10+x11)y00 -> z10+1, z11-1
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        # m3: x00(y01-y11) -> z01+1, z11+1
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        # m4: x11(y10-y00) -> z00+1, z10+1
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        # m5: (x00+x01)y11 -> z00-1, z01+1
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        # m6: (x10-x00)(y00+y01) -> z11+1
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        # m7: (x01-x11)(y10+y11) -> z00+1
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1})
    ]
    
    # Standard 2x2x1 coefficients
    # u: 4 entries, v: 2 entries, w: 2 entries
    # z0 = x00*y0 + x01*y1
    # z1 = x10*y0 + x11*y1
    standard_terms_2x2x1 = [
        ({0:1}, {0:1}, {0:1}), # x00*y0 -> z0
        ({1:1}, {1:1}, {0:1}), # x01*y1 -> z0
        ({2:1}, {0:1}, {1:1}), # x10*y0 -> z1
        ({3:1}, {1:1}, {1:1})  # x11*y1 -> z1
    ]

    for q in range(2): # Block in m (0, 1)
        # A block indices: i in 0..1, j_local in 0..1
        # Global A index: i*m + (2*q + j_local)
        # B block indices: j_local in 0..1, k_global
        # Global B index: (2*q + j_local)*p + k_global
        # C indices: i in 0..1, k_global
        # Global C index: k_global*n + i
        
        for s in range(3): # Segment in p (0, 1, 2)
            if s < 2:
                # 2x2x2 Strassen
                # k_local in 0..1
                # k_global = 2*s + k_local
                for u_t, v_t, w_t in strassen_terms:
                    u_global = {}
                    v_global = {}
                    w_global = {}
                    
                    for idx, val in u_t.items():
                        # idx is local A index 0..3
                        # local i = idx // 2, local j = idx % 2
                        li = idx // 2
                        lj = idx % 2
                        gi = li
                        gj = 2*q + lj
                        u_global[gi * m + gj] = val
                    
                    for idx, val in v_t.items():
                        # idx is local B index 0..3
                        # local j = idx // 2, local k = idx % 2
                        lj = idx // 2
                        lk = idx % 2
                        gj = 2*q + lj
                        gk = 2*s + lk
                        v_global[gj * p + gk] = val
                    
                    for idx, val in w_t.items():
                        # idx is local C index 0..3
                        # local i = idx // 2, local k = idx % 2
                        li = idx // 2
                        lk = idx % 2
                        gi = li
                        gk = 2*s + lk
                        w_global[gk * n + gi] = val
                    
                    add_term(u_global, v_global, w_global)
            else:
                # 2x2x1 Standard
                # k_local is 0 (only 1 column)
                # k_global = 4
                for u_t, v_t, w_t in standard_terms_2x2x1:
                    u_global = {}
                    v_global = {}
                    w_global = {}
                    
                    for idx, val in u_t.items():
                        li = idx // 2
                        lj = idx % 2
                        gi = li
                        gj = 2*q + lj
                        u_global[gi * m + gj] = val
                    
                    for idx, val in v_t.items():
                        # idx is local B index 0..1 (row)
                        lj = idx
                        gk = 4
                        gj = 2*q + lj
                        v_global[gj * p + gk] = val
                    
                    for idx, val in w_t.items():
                        # idx is local C index 0..1 (row)
                        li = idx
                        gk = 4
                        gi = li
                        w_global[gk * n + gi] = val
                    
                    add_term(u_global, v_global, w_global)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
