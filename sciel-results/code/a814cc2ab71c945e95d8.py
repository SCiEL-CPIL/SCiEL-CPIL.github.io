import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Inner Strassen vectors for 2x2
    # u_l, v_l, w_l are lists of 4 elements
    u_inner = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    v_inner = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    w_inner = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [0, 1, -1, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Outer Strassen coefficients for A blocks (indices 0:00, 1:01, 2:10, 3:11)
    a_outer = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    # Outer Strassen coefficients for B blocks
    b_outer = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Outer Strassen coefficients for C blocks
    # c[block][k]
    # Block 0: C00, Block 1: C01, Block 2: C10, Block 3: C11
    c_outer = [
        [1, 0, 0, 1, -1, 0, 1], # C00 from M1, M4, M5, M7
        [0, 0, 1, 0, 1, 0, 0],   # C01 from M3, M5
        [0, 1, 0, 1, 0, 0, 0],   # C10 from M2, M4
        [1, -1, 1, 0, 0, 1, 0]   # C11 from M1, M2, M3, M6
    ]
    
    # Mapping for C block elements to global W index
    # w = k * 4 + i
    # Block 0 (C00): i in 0,1; k in 0,1
    # Local (r,c) -> global i=r, k=c -> w = c*4 + r
    map_c00 = [0, 4, 1, 5] # indices for r=0c=0, r=0c=1, r=1c=0, r=1c=1
    # Block 1 (C01): i in 0,1; k in 2,3
    # Local (r,c) -> global i=r, k=2+c -> w = (2+c)*4 + r
    map_c01 = [8, 12, 9, 13]
    # Block 2 (C10): i in 2,3; k in 0,1
    # Local (r,c) -> global i=2+r, k=c -> w = c*4 + (2+r)
    map_c10 = [2, 6, 3, 7]
    # Block 3 (C11): i in 2,3; k in 2,3
    # Local (r,c) -> global i=2+r, k=2+c -> w = (2+c)*4 + (2+r)
    map_c11 = [10, 14, 11, 15]
    
    c_maps = [map_c00, map_c01, map_c10, map_c11]
    
    for k in range(7):
        for l in range(7):
            r_idx = k * 7 + l
            
            # Update U
            for b in range(4):
                coeff_a = a_outer[k][b]
                if coeff_a != 0:
                    base_u = b * 4
                    for e in range(4):
                        coeff_inner = u_inner[l][e]
                        if coeff_inner != 0:
                            U[base_u + e][r_idx] += coeff_a * coeff_inner
            
            # Update V
            for b in range(4):
                coeff_b = b_outer[k][b]
                if coeff_b != 0:
                    base_v = b * 4
                    for e in range(4):
                        coeff_inner = v_inner[l][e]
                        if coeff_inner != 0:
                            V[base_v + e][r_idx] += coeff_b * coeff_inner
            
            # Update W
            for block in range(4):
                coeff_c = c_outer[block][k]
                if coeff_c != 0:
                    for e in range(4):
                        coeff_inner = w_inner[l][e]
                        if coeff_inner != 0:
                            w_idx = c_maps[block][e]
                            W[w_idx][r_idx] += coeff_c * coeff_inner
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
