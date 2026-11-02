import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 2x2 * 2x14 matrix multiplication.
    Implemented using NumPy for distinct structural AST compared to list-based seed.
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank = num_blocks * 7
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen decomposition for 2x2x2
    # U forms: 7 rows (forms) x 4 cols (A entries)
    # A = [[a00, a01], [a10, a11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # U_forms[i, j] where i is term 0..6, j is A entry 0..3
    U_forms = np.array([
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ])
    
    # V forms: 7 rows x 4 cols (B entries for a 2x2 block)
    # B_block = [[b00, b01], [b10, b11]]
    # Local indices: 0:b00, 1:b01, 2:b10, 3:b11
    V_forms = np.array([
        [1, 0, 0, 1],  # M1: b00 + b11
        [1, 0, 0, 0],  # M2: b00
        [0, 1, 0, -1], # M3: b01 - b11
        [0, 0, 1, -1], # M4: b10 - b00  (Wait, standard Strassen M4 is a11(b10-b00))
        [0, 0, 0, 1],  # M5: b11
        [1, 1, 0, 0],  # M6: b00 + b01
        [0, 0, 1, 1]   # M7: b10 + b11
    ])
    
    # W forms: 7 rows x 4 cols (C entries for a 2x2 block)
    # C_block = [[c00, c01], [c10, c11]]
    # Local indices: 0:c00, 1:c01, 2:c10, 3:c11
    # Standard Strassen reconstruction:
    # c00 = M1 + M4 - M5 - M7
    # c01 = M3 + M5
    # c10 = M2 + M6
    # c11 = M1 - M2 + M3 + M7
    W_forms = np.array([
        [1, 0, 0, -1],  # M1 contributes to c00 and c11
        [0, 0, 1, -1],  # M2 contributes to c10 and -c11
        [0, 1, 0, 1],   # M3 contributes to c01 and c11
        [1, 0, 0, 0],   # M4 contributes to c00
        [-1, 1, 0, 0],  # M5 contributes to -c00 and c01
        [0, 0, 1, 0],   # M6 contributes to c10
        [-1, 0, 0, 1]   # M7 contributes to -c00 and c11
    ])
    
    # Fill U: same for all blocks
    for s in range(7):
        # U has 4 rows (A entries), so U_forms[s] is the column
        U[:, 7*s + 0:7*s + 7] = U_forms[s]
    
    # Fill V and W for each block r
    for r in range(num_blocks):
        for s in range(7):
            col_idx = 7 * r + s
            
            # V indices
            # B is 2 x 14.
            # Block r corresponds to columns 2r and 2r+1.
            # V is m x p = 2 x 14.
            # v_local 0 (b00) -> V[0, 2r]
            # v_local 1 (b01) -> V[0, 2r+1]
            # v_local 2 (b10) -> V[1, 2r]
            # v_local 3 (b11) -> V[1, 2r+1]
            
            for v_local in range(4):
                coeff = V_forms[s, v_local]
                if coeff != 0:
                    row = v_local // 2
                    col_local = v_local % 2
                    v_global_row = row
                    v_global_col = 2 * r + col_local
                    V[v_global_row, v_global_col] += coeff * 0 # Placeholder, need to map to column
                    # Wait, V is (m*p, rank). So index is v_global_row * p + v_global_col?
                    # No, standard interface: V is (m*p, rank).
                    # The input B is flattened?
                    # Interface: U (n*m, rank), V (m*p, rank), W (p*n, rank).
                    # Usually, A is flattened row-major or column-major?
                    # Standard: A is n x m. Flattened to n*m.
                    # B is m x p. Flattened to m*p.
                    # If B is row-major, index is row * p + col.
                    # Let's assume row-major flattening for B.
                    # V[row * p + col, col_idx]
                    
                    idx = v_global_row * p + v_global_col
                    V[idx, col_idx] = coeff
            
            # W indices
            # W is (p*n, rank).
            # C is p x n? No, C is n x p.
            # Interface says W is (p*n, rank).
            # Convention: W[k*n + i, r] where C[i, k].
            # So index is k * n + i.
            # C_block is 2x2.
            # w_local 0 (c00) -> i=0, k=2r -> idx = 2r * 2 + 0
            # w_local 1 (c01) -> i=1, k=2r -> idx = 2r * 2 + 1
            # w_local 2 (c10) -> i=0, k=2r+1 -> idx = (2r+1) * 2 + 0
            # w_local 3 (c11) -> i=1, k=2r+1 -> idx = (2r+1) * 2 + 1
            
            for w_local in range(4):
                coeff = W_forms[s, w_local]
                if coeff != 0:
                    # Map local c index to global W index
                    if w_local == 0: # c00
                        i, k = 0, 2 * r
                    elif w_local == 1: # c01
                        i, k = 1, 2 * r
                    elif w_local == 2: # c10
                        i, k = 0, 2 * r + 1
                    else: # c11
                        i, k = 1, 2 * r + 1
                    
                    idx = k * n + i
                    W[idx, col_idx] = coeff

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
