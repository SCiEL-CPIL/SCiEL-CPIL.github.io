"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # Block size 2, so 3x3 blocks
    s = 2
    k_blocks = n // s # 3
    
    rank = k_blocks * k_blocks * k_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # M_s = sum_u A_u * sum_v B_v
    # We define A_coeff[s][u] and B_coeff[s][v]
    # u, v are local indices 0..3 (00, 01, 10, 11)
    # local index mapping: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    
    A_coeff = [
        {0: 1, 3: 1},       # M0: a00 + a11
        {2: 1, 3: 1},       # M1: a10 + a11
        {0: 1},             # M2: a00
        {3: 1},             # M3: a11
        {0: 1, 2: 1},       # M4: a00 + a10
        {3: 1, 0: -1},      # M5: a11 - a00
        {1: 1, 3: -1}       # M6: a01 - a11
    ]
    
    B_coeff = [
        {0: 1, 3: 1},       # M0: b00 + b11
        {0: 1},             # M1: b00
        {2: 1, 3: -1},      # M2: b10 - b11
        {1: 1, 0: -1},      # M3: b01 - b00
        {3: 1},             # M4: b11
        {0: 1, 2: 1},       # M5: b00 + b10
        {3: 1, 1: 1}        # M6: b11 + b01
    ]
    
    # Coeff[s][u][w] for Z_uw = sum_s Coeff[s][u][w] * M_s
    # Z is 2x2, so u, w in {0,1}
    # local index for Z: u*2 + w ? No, Z_uw is row u, col w.
    # We need to map Z_uw to C index.
    # C_ki where k=2K+w, i=2I+u.
    # So for a given s, and Z element (u,w), the W index is (2K+w)*n + (2I+u).
    
    W_coeff = [
        # s=0
        {(0,0): 1, (1,1): 1},
        # s=1
        {(1,0): 1, (1,1): -1},
        # s=2
        {(0,1): 1, (1,1): 1},
        # s=3
        {(0,0): 1, (1,0): 1},
        # s=4
        {(0,0): -1, (0,1): 1},
        # s=5
        {(1,1): 1},
        # s=6
        {(0,0): 1}
    ]
    
    col = 0
    for I in range(k_blocks):
        for J in range(k_blocks):
            for K in range(k_blocks):
                for s_idx in range(7):
                    # Process column col
                    # A part
                    for u_idx, coeff in A_coeff[s_idx].items():
                        # u_idx is 0..3
                        u = u_idx // 2
                        v = u_idx % 2
                        i = I * s + u
                        j = J * s + v
                        U[i * m + j][col] += coeff
                    
                    # B part
                    for v_idx, coeff in B_coeff[s_idx].items():
                        # v_idx is 0..3
                        v = v_idx // 2
                        w = v_idx % 2
                        j = J * s + v
                        k = K * s + w
                        V[j * p + k][col] += coeff
                    
                    # W part
                    for (u, w), coeff in W_coeff[s_idx].items():
                        # u, w are 0,1
                        i = I * s + u
                        k = K * s + w
                        W[k * n + i][col] += coeff
                    
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
