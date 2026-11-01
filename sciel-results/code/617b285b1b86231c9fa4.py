"""Matrix multiplication 4x4x4: rank-minimal exact tensor decomposition via recursive Strassen."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Build a rank-minimal exact tensor decomposition for 4x4 matrix multiplication
    using recursive Strassen: apply Strassen at the 4x4 block level (7 products),
    and within each 2x2 sub-product apply Strassen again (7 scalar mults each),
    giving 7*7 = 49 total rank terms.
    
    The decomposition is generated programmatically using a recursive function
    that maps hierarchical Strassen indices to global tensor coordinates,
    ensuring exact integer arithmetic.
    """
    import time
    start_time = time.time()
    DEADLINE = 145  # seconds (safety margin before 150s kill)

    n, m, p = DIMENSIONS  # 4, 4, 4
    size = 4  # matrix dimension

    # 2x2 Strassen decomposition as (u_vec, v_vec, w_vec) triples
    # u_vec: coefficients on [a00, a01, a10, a11]
    # v_vec: coefficients on [b00, b01, b10, b11]
    # w_vec: coefficients on [c00, c01, c10, c11]
    STRASSEN_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # s1 = (a00+a11)(b00+b11)
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # s2 = (a10+a11)b00
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # s3 = a00(b01-b11)
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),  # s4 = a11(b10-b00)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # s5 = (a00+a01)b11
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # s6 = (a10-a00)(b00+b01)
        ([0, 1, -1, 1], [0, 0, 1, 1], [1, 0, 0, 0]),  # s7 = (a01-a11)(b10+b11)
    ]

    # Block-level Strassen for 4x4 (split into 2x2 blocks)
    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11
    # For each of the 7 block products S_i = X_i * Y_i:
    # X_i: coefficients on (A00, A01, A10, A11)
    # Y_i: coefficients on (B00, B01, B10, B11)
    # C_i: dict mapping C-block index -> coefficient
    BLOCK_X = [
        [1, 0, 0, 1],   # S1: A00 + A11
        [0, 0, 1, 1],   # S2: A10 + A11
        [1, 0, 0, 0],   # S3: A00
        [0, 0, 0, 1],   # S4: A11
        [1, 1, 0, 0],   # S5: A00 + A01
        [-1, 0, 1, 0],  # S6: A10 - A00
        [0, 1, -1, 1],  # S7: A01 - A11
    ]
    BLOCK_Y = [
        [1, 0, 0, 1],   # S1: B00 + B11
        [1, 0, 0, 0],   # S2: B00
        [0, 1, 0, -1],  # S3: B01 - B11
        [-1, 0, 1, 0],  # S4: B10 - B00
        [0, 0, 0, 1],   # S5: B11
        [1, 1, 0, 0],   # S6: B00 + B01
        [0, 0, 1, 1],   # S7: B10 + B11
    ]
    BLOCK_C = [
        {0: 1, 3: 1},      # S1 -> C00(+1), C11(+1)
        {2: 1, 3: -1},     # S2 -> C10(+1), C11(-1)
        {1: 1, 3: 1},      # S3 -> C01(+1), C11(+1)
        {0: 1, 2: 1},      # S4 -> C00(+1), C10(+1)
        {0: -1, 1: 1},     # S5 -> C00(-1), C01(+1)
        {3: 1},            # S6 -> C11(+1)
        {0: 1},            # S7 -> C00(+1)
    ]

    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]

    def block_elem_index(block_idx, elem_idx):
        """
        Map (block_idx, elem_idx) to global flat index.
        block_idx: 0,1,2,3 for blocks in 2x2 arrangement
        elem_idx: 0,1,2,3 for elements within 2x2 block
        Returns row*4 + col in the 4x4 matrix.
        """
        bi = block_idx // 2  # block row: 0 or 1
        bj = block_idx % 2   # block col: 0 or 1
        br = elem_idx // 2   # element row within block: 0 or 1
        bc = elem_idx % 2    # element col within block: 0 or 1
        global_row = bi * 2 + br
        global_col = bj * 2 + bc
        return global_row * 4 + global_col

    # Build the 49-term decomposition
    for i in range(7):  # block product index
        for j in range(7):  # 2x2 Strassen term index
            r = i * 7 + j
            u_vec, v_vec, w_vec = STRASSEN_2x2[j]
            
            # U: contributions from A-blocks
            for ab in range(4):
                coeff_x = BLOCK_X[i][ab]
                if coeff_x == 0:
                    continue
                for e in range(4):
                    u_elem = u_vec[e]
                    if u_elem == 0:
                        continue
                    idx = block_elem_index(ab, e)
                    U[idx][r] += coeff_x * u_elem
            
            # V: contributions from B-blocks
            for bb in range(4):
                coeff_y = BLOCK_Y[i][bb]
                if coeff_y == 0:
                    continue
                for e in range(4):
                    v_elem = v_vec[e]
                    if v_elem == 0:
                        continue
                    idx = block_elem_index(bb, e)
                    V[idx][r] += coeff_y * v_elem
            
            # W: contributions from C-blocks
            for cb, coeff_c in BLOCK_C[i].items():
                if coeff_c == 0:
                    continue
                for e in range(4):
                    w_elem = w_vec[e]
                    if w_elem == 0:
                        continue
                    idx = block_elem_index(cb, e)
                    W[idx][r] += coeff_c * w_elem

    # Verify the decomposition
    if time.time() - start_time < DEADLINE - 3:
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: for all i,j,k, sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] = 1
        T_check = np.zeros((4, 4, 4), dtype=float)
        for r in range(rank):
            u_r = U_np[:, r]
            v_r = V_np[:, r]
            w_r = W_np[:, r]
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        T_check[i, j, k] += u_r[i*4+j] * v_r[j*4+k] * w_r[k*4+i]
        
        expected = np.ones((4, 4, 4))
        max_err = np.max(np.abs(T_check - expected))
        if max_err > 1e-9:
            # Fallback: schoolbook rank 64
            rank_fb = 64
            U_fb = [[0] * rank_fb for _ in range(16)]
            V_fb = [[0] * rank_fb for _ in range(16)]
            W_fb = [[0] * rank_fb for _ in range(16)]
            col = 0
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        U_fb[i*4+j][col] = 1
                        V_fb[j*4+k][col] = 1
                        W_fb[k*4+i][col] = 1
                        col += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
