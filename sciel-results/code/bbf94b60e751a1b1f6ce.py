import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # A coefficients: [m1, m2, m3, m4, m5, m6, m7]
    # Each is a dict or list of (i, j, coeff)
    # We map local indices 0,1 to global.
    
    # Local A indices for left block: i in 0,1; j in 0,1
    # Local A indices for right block: i in 0,1; j in 2,3
    
    # Local B indices for top block: j in 0,1; k in 0,1 (local)
    # Local B indices for bot block: j in 2,3; k in 0,1 (local)
    
    # Strassen multipliers for A (left)
    # m1: (0,0)+ (1,1)
    # m2: (1,0)+ (1,1)
    # m3: (0,0)
    # m4: (1,1)
    # m5: (0,0)+ (0,1)
    # m6: (1,0)- (0,0)
    # m7: (0,1)- (1,1)
    
    A_left_coeffs = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)]
    ]
    
    # A_right coeffs are same but j offset by 2
    A_right_coeffs = [
        [(0, 2, 1), (1, 3, 1)],
        [(1, 2, 1), (1, 3, 1)],
        [(0, 2, 1)],
        [(1, 3, 1)],
        [(0, 2, 1), (0, 3, 1)],
        [(1, 2, 1), (0, 2, -1)],
        [(0, 3, 1), (1, 3, -1)]
    ]
    
    # B coeffs (local k 0,1)
    # m1: (0,0)+ (1,1)
    # m2: (0,0)
    # m3: (0,1)- (1,1)
    # m4: (1,0)- (0,0)
    # m5: (1,1)
    # m6: (0,0)+ (0,1)
    # m7: (1,0)+ (1,1)
    
    B_top_coeffs = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)]
    ]
    
    B_bot_coeffs = [
        [(2, 0, 1), (3, 1, 1)],
        [(2, 0, 1)],
        [(2, 1, 1), (3, 1, -1)],
        [(3, 0, 1), (2, 0, -1)],
        [(3, 1, 1)],
        [(2, 0, 1), (2, 1, 1)],
        [(3, 0, 1), (3, 1, 1)]
    ]
    
    # C coeffs (local k 0,1)
    # z00 = m1 + m4 - m5 + m7
    # z01 = m3 + m5
    # z10 = m2 + m4
    # z11 = m1 - m2 + m3 + m6
    
    # We need to map each m_s to contributions to C[i, k_local]
    # C_local indices: i in 0,1; k_local in 0,1
    
    C_coeffs = [
        # m1
        [(0, 0, 1), (1, 1, 1)],
        # m2
        [(1, 0, -1), (1, 1, 1)], # Wait, z11 has -m2. z10 has +m2.
        # Let's list contributions per m_s to C entries
        # m1: C00 += 1, C11 += 1
        # m2: C10 += 1, C11 += -1
        # m3: C01 += 1, C11 += 1
        # m4: C00 += 1, C10 += 1
        # m5: C00 += -1, C01 += 1
        # m6: C11 += 1
        # m7: C00 += 1
        # Let's re-verify
        # z00 = m1 + m4 - m5 + m7
        # z01 = m3 + m5
        # z10 = m2 + m4
        # z11 = m1 - m2 + m3 + m6
        
        # So:
        # m1: (0,0,1), (1,1,1)
        # m2: (1,0,1), (1,1,-1)
        # m3: (0,1,1), (1,1,1)
        # m4: (0,0,1), (1,0,1)
        # m5: (0,0,-1), (0,1,1)
        # m6: (1,1,1)
        # m7: (0,0,1)
    ]
    
    C_coeffs = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    for t in range(4):
        # P_t: A_left * B_top
        for s in range(7):
            c_idx = t * 14 + s
            # U
            for i, j, coeff in A_left_coeffs[s]:
                U[i * m + j][c_idx] += coeff
            # V
            for j, k_local, coeff in B_top_coeffs[s]:
                k_global = t * 2 + k_local
                V[j * p + k_global][c_idx] += coeff
            # W
            for i, k_local, coeff in C_coeffs[s]:
                k_global = t * 2 + k_local
                W[k_global * n + i][c_idx] += coeff
                
        # Q_t: A_right * B_bot
        for s in range(7):
            c_idx = t * 14 + 7 + s
            # U
            for i, j, coeff in A_right_coeffs[s]:
                U[i * m + j][c_idx] += coeff
            # V
            for j, k_local, coeff in B_bot_coeffs[s]:
                k_global = t * 2 + k_local
                V[j * p + k_global][c_idx] += coeff
            # W
            for i, k_local, coeff in C_coeffs[s]:
                k_global = t * 2 + k_local
                W[k_global * n + i][c_idx] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
