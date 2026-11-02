"""
Rank-minimal tensor decomposition for 2x4 by 4x8 matrix multiplication.

Uses Strassen's algorithm on 2x2 blocks to achieve rank 56.
A (2x4) is split into two 2x2 blocks A1, A2.
B (4x8) is split into top (2x8) and bottom (2x8).
B_top is split into four 2x2 blocks.
B_bottom is split into four 2x2 blocks.
Total 8 blocks of 2x2 * 2x2 multiplication.
Each 2x2 * 2x2 uses Strassen (rank 7).
Total rank = 8 * 7 = 56.
"""

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]  # shape (8, 56)
    V = [[0] * rank for _ in range(m * p)]  # shape (32, 56)
    W = [[0] * rank for _ in range(p * n)]  # shape (16, 56)
    
    # Strassen coefficients for U, V, W
    # Each entry is a list of (index_offset, coefficient)
    # Indices are local (0..1)
    
    strassen_terms = [
        # P1
        {
            'U': [((0,0), 1), ((1,1), 1)],
            'V': [((0,0), 1), ((1,1), 1)],
            'W': [((0,0), 1), ((1,1), 1)] # (x, v)
        },
        # P2
        {
            'U': [((1,0), 1), ((1,1), 1)],
            'V': [((0,0), 1)],
            'W': [((1,0), 1), ((1,1), -1)]
        },
        # P3
        {
            'U': [((0,0), 1)],
            'V': [((1,0), 1), ((1,1), -1)],
            'W': [((0,1), 1), ((1,1), 1)]
        },
        # P4
        {
            'U': [((1,1), 1)],
            'V': [((0,1), 1), ((0,0), -1)],
            'W': [((0,0), 1), ((1,0), 1)]
        },
        # P5
        {
            'U': [((0,0), 1), ((1,0), 1)],
            'V': [((1,1), 1)],
            'W': [((0,0), -1), ((0,1), 1)]
        },
        # P6
        {
            'U': [((1,0), 1), ((0,0), -1)],
            'V': [((0,0), 1), ((0,1), 1)],
            'W': [((1,1), 1)]
        },
        # P7
        {
            'U': [((0,1), 1), ((1,1), -1)],
            'V': [((1,0), 1), ((1,1), 1)],
            'W': [((0,0), 1)]
        }
    ]
    
    # 8 blocks
    for b in range(8):
        if b < 4:
            # A1 (j offset 0), B_top (j offset 0)
            j_off_A = 0
            j_off_B = 0
            k_off = 2 * b
        else:
            # A2 (j offset 2), B_bottom (j offset 2)
            j_off_A = 2
            j_off_B = 2
            k_off = 2 * (b - 4)
            
        for t in range(7):
            r = b * 7 + t
            term = strassen_terms[t]
            
            # Fill U
            for (local_x, local_y), coeff in term['U']:
                global_i = local_x
                global_j = j_off_A + local_y
                idx = global_i * m + global_j
                U[idx][r] = coeff
                
            # Fill V
            for (local_u, local_v), coeff in term['V']:
                global_j = j_off_B + local_u
                global_k = k_off + local_v
                idx = global_j * p + global_k
                V[idx][r] = coeff
                
            # Fill W
            for (local_x, local_v), coeff in term['W']:
                global_i = local_x
                global_k = k_off + local_v
                idx = global_k * n + global_i
                W[idx][r] = coeff
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
