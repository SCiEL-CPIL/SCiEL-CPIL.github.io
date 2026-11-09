import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 decomposition matrices
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Indices: 00->0, 01->1, 10->2, 11->3
    # W2 row index: k*2+i (col*2+row)
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],  # 00
        [0, 0, 0, 0, 1, 0, 1],   # 01
        [0, 1, 0, 0, 0, 1, 0],   # 10
        [1, 1, 0, 1, 0, 0, -1]   # 11
    ]
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],  # 00
        [0, 0, 1, 0, 0, 1, 0],   # 01
        [0, 0, 0, 1, 0, 0, 1],   # 10
        [1, 0, -1, 0, 1, 0, 1]   # 11
    ]
    
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],  # 00 (k=0, i=0)
        [0, 0, 1, 0, 1, 0, 0],   # 01 (k=1, i=0) -> Wait.
        # C_01 is row 0, col 1. k=1, i=0. Index 1*2+0=2.
        # My W2 row 1 was for C_01?
        # Let's re-verify W2 rows.
        # Row 0: C_00 (k=0, i=0). Index 0.
        # Row 1: C_01 (k=1, i=0). Index 2?
        # The list index in W2 should correspond to the flattened index.
        # But I am accessing W2 by block indices I, J.
        # In the 2x2 case, W2[row_idx] where row_idx = k*2+i.
        # So W2[0] is C_00.
        # W2[1] is C_10? No.
        # k=0, i=1 -> C_10. Index 1.
        # k=1, i=0 -> C_01. Index 2.
        # k=1, i=1 -> C_11. Index 3.
        
        # So W2 list order:
        # 0: C_00
        # 1: C_10
        # 2: C_01
        # 3: C_11
        
        # Let's re-check my W2 definition above.
        # I defined:
        # Row 0: C_00
        # Row 1: C_01  <-- ERROR in previous thought
        # Row 2: C_10
        # Row 3: C_11
        
        # I need to map W2 list index to k*2+i.
        # List index 0 -> k=0, i=0 (C_00)
        # List index 1 -> k=0, i=1 (C_10)
        # List index 2 -> k=1, i=0 (C_01)
        # List index 3 -> k=1, i=1 (C_11)
    ]
    
    # Correct W2 based on k*2+i ordering:
    # C_00 (0): m1 + m4 - m5 + m7 -> [1, 0, 0, 1, -1, 0, 1]
    # C_10 (1): m2 + m4           -> [0, 1, 0, 1, 0, 0, 0]
    # C_01 (2): m3 + m5           -> [0, 0, 1, 0, 1, 0, 0]
    # C_11 (3): m1 - m2 + m3 + m6 -> [1, -1, 1, 0, 0, 1, 0]
    
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],  # 0: C_00
        [0, 1, 0, 1, 0, 0, 0],   # 1: C_10
        [0, 0, 1, 0, 1, 0, 0],   # 2: C_01
        [1, -1, 1, 0, 0, 1, 0]   # 3: C_11
    ]

    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill U and V
    for I in range(2):
        for J in range(2):
            block_idx = I * 2 + J
            for i in range(2):
                for j in range(2):
                    inner_idx = i * 2 + j
                    u_idx = (I * 2 + i) * 4 + (J * 2 + j)
                    v_idx = (I * 2 + i) * 4 + (J * 2 + j)
                    
                    for r in range(7):
                        ub = U2[block_idx][r]
                        vb = V2[block_idx][r]
                        if ub == 0 and vb == 0:
                            continue
                        for s in range(7):
                            c = r * 7 + s
                            ui = U2[inner_idx][s]
                            vi = V2[inner_idx][s]
                            if ui == 0 and vi == 0:
                                continue
                            U[u_idx][c] = ub * ui
                            V[v_idx][c] = vb * vi

    # Fill W
    for I in range(2):
        for J in range(2):
            # W2 block index is J*2 + I (col*2+row)
            block_idx = J * 2 + I
            for i in range(2):
                for j in range(2):
                    # W2 inner index is j*2 + i (col*2+row)
                    inner_idx = j * 2 + i
                    w_idx = (J * 2 + j) * 4 + (I * 2 + i)
                    
                    for r in range(7):
                        wb = W2[block_idx][r]
                        if wb == 0:
                            continue
                        for s in range(7):
                            c = r * 7 + s
                            wi = W2[inner_idx][s]
                            if wi == 0:
                                continue
                            W[w_idx][c] = wb * wi

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
