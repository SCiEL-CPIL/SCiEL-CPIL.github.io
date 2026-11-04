import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # U size: n*m = 8
    # V size: m*p = 16
    # W size: p*n = 8
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for U (X), V (Y), W (Z)
    # Order of X: x00, x01, x10, x11
    # Order of Y: y00, y01, y10, y11
    # Order of Z: z00, z01, z10, z11
    
    # m1
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1]
    
    # m2
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, -1]
    
    # m3
    u3 = [1, 0, 0, 0]
    v3 = [0, 0, 1, -1]
    w3 = [0, 1, 0, 1]
    
    # m4
    u4 = [0, 0, 0, 1]
    v4 = [-1, 1, 0, 0]
    w4 = [1, 0, 1, 0]
    
    # m5
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [-1, 1, 0, 0]
    
    # m6
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 0, 1]
    
    # m7
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [1, 0, 0, 0]
    
    strassen_coeffs = [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7)
    ]
    
    # Define index mappings
    # A indices: i*4 + j
    # A_L: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # A_R: (0,2)->2, (0,3)->3, (1,2)->6, (1,3)->7
    
    # B indices: j*4 + k
    # B_TL: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # B_TR: (0,2)->2, (0,3)->3, (1,2)->6, (1,3)->7
    # B_BL: (2,0)->8, (2,1)->9, (3,0)->12, (3,1)->13
    # B_BR: (2,2)->10, (2,3)->11, (3,2)->14, (3,3)->15
    
    # W indices: k*2 + i
    # C_L (k=0,1):
    # z00 (C00) -> 0*2+0=0
    # z01 (C01) -> 1*2+0=2
    # z10 (C10) -> 0*2+1=1
    # z11 (C11) -> 1*2+1=3
    # C_R (k=2,3):
    # z00 (C02) -> 2*2+0=4
    # z01 (C03) -> 3*2+0=6
    # z10 (C12) -> 2*2+1=5
    # z11 (C13) -> 3*2+1=7
    
    mappings = [
        {
            "name": "P1", # A_L * B_TL -> C_L
            "u_idx": [0, 1, 4, 5],
            "v_idx": [0, 1, 4, 5],
            "w_idx": [0, 2, 1, 3]
        },
        {
            "name": "P2", # A_R * B_BL -> C_L
            "u_idx": [2, 3, 6, 7],
            "v_idx": [8, 9, 12, 13],
            "w_idx": [0, 2, 1, 3]
        },
        {
            "name": "P3", # A_L * B_TR -> C_R
            "u_idx": [0, 1, 4, 5],
            "v_idx": [2, 3, 6, 7],
            "w_idx": [4, 6, 5, 7]
        },
        {
            "name": "P4", # A_R * B_BR -> C_R
            "u_idx": [2, 3, 6, 7],
            "v_idx": [10, 11, 14, 15],
            "w_idx": [4, 6, 5, 7]
        }
    ]
    
    col = 0
    for mp in mappings:
        for uc, vc, wc in strassen_coeffs:
            for l in range(4):
                if uc[l] != 0:
                    U[mp["u_idx"][l]][col] += uc[l]
                if vc[l] != 0:
                    V[mp["v_idx"][l]][col] += vc[l]
                if wc[l] != 0:
                    W[mp["w_idx"][l]][col] += wc[l]
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
