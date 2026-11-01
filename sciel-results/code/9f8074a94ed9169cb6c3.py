"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for M1..M7
    # U coeffs for A:
    # M1: a00+a11 -> [1, 0, 0, 1]
    # M2: a10+a11 -> [0, 0, 1, 1]
    # M3: a00     -> [1, 0, 0, 0]
    # M4: a11     -> [0, 0, 0, 1]
    # M5: a00+a01 -> [1, 1, 0, 0]
    # M6: a10-a00 -> [-1, 0, 1, 0]
    # M7: a01-a11 -> [0, 1, 0, -1]
    U_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    # V coeffs for B block (2x2):
    # M1: b00+b11
    # M2: b00
    # M3: b01-b11
    # M4: b10-b00
    # M5: b11
    # M6: b00+b01
    # M7: b10+b11
    # Indices in 2x2 block: 00, 10, 01, 11 (row-major? No, V index is j*p+k)
    # j=0, k=0 -> 0
    # j=1, k=0 -> 6
    # j=0, k=1 -> 1
    # j=1, k=1 -> 7
    # So for block k, offset is 2k.
    # b00: 2k
    # b10: 6+2k
    # b01: 2k+1
    # b11: 7+2k
    V_indices = [
        [0, 6, 1, 7], # placeholders, will be added with offset
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]
    V_coeffs = [
        [1, 0, 0, 1], # b00, b10, b01, b11
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    
    # W coeffs for C block (2x2):
    # C00, C10, C01, C11 (order in W: 4k, 4k+1, 4k+2, 4k+3)
    # M1: C00=1, C11=1
    # M2: C10=1
    # M3: C01=1, C11=-1
    # M4: C00=-1, C10=1
    # M5: C01=1, C11=1
    # M6: C00=-1
    # M7: C00=1, C11=-1
    W_coeffs = [
        [1, 0, 0, 1], # C00, C10, C01, C11
        [0, 1, 0, 0],
        [0, 0, 1, -1],
        [-1, 1, 0, 0],
        [0, 0, 1, 1],
        [-1, 0, 0, 0],
        [1, 0, 0, -1]
    ]
    
    for k in range(3):
        for t in range(7):
            col = 7 * k + t
            
            # Fill U
            for i in range(4):
                if U_coeffs[t][i] != 0:
                    U[i][col] += U_coeffs[t][i]
            
            # Fill V
            offset = 2 * k
            # Indices relative to block start:
            # b00: 0, b10: 6, b01: 1, b11: 7
            # Wait, V index is j*p + k.
            # Block k uses cols 2k, 2k+1.
            # j=0, k=2k -> 2k
            # j=1, k=2k -> 6+2k
            # j=0, k=2k+1 -> 2k+1
            # j=1, k=2k+1 -> 6+2k+1 = 7+2k
            v_idx_00 = 2 * k
            v_idx_10 = 6 + 2 * k
            v_idx_01 = 2 * k + 1
            v_idx_11 = 7 + 2 * k
            v_indices = [v_idx_00, v_idx_10, v_idx_01, v_idx_11]
            
            for idx_in_block, coeff in enumerate(V_coeffs[t]):
                if coeff != 0:
                    V[v_indices[idx_in_block]][col] += coeff
            
            # Fill W
            w_offset = 4 * k
            # C00: 4k, C10: 4k+1, C01: 4k+2, C11: 4k+3
            w_indices = [w_offset, w_offset + 1, w_offset + 2, w_offset + 3]
            for idx_in_block, coeff in enumerate(W_coeffs[t]):
                if coeff != 0:
                    W[w_indices[idx_in_block]][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
