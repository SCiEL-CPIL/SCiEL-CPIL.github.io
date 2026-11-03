"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # Part 1: A[0:2, :] * B[:, :] -> C[0:2, :]
    # Split A[0:2, :] into A[0:2, 0:2] and A[0:2, 2:4].
    # Split B[:, :] into B[0:2, :] and B[2:4, :].
    # C[0:2, :] = A[0:2, 0:2] * B[0:2, :] + A[0:2, 2:4] * B[2:4, :]
    # Each term is 2x2 * 2x8.
    # 2x2 * 2x8 can be decomposed into 4 * (2x2 * 2x2).
    # Each 2x2 * 2x2 uses 7 Strassen multiplications.
    # So 4 * 7 = 28 multiplications per 2x2 * 2x8 block.
    # Two such blocks (for A[0:2, 0:2] and A[0:2, 2:4]) -> 28 + 28 = 56.
    
    # Part 2: A[2:3, :] * B[:, :] -> C[2:3, :]
    # This is 1x4 * 4x8.
    # C[2, k] = sum_{j=0}^3 A[2, j] B[j, k].
    # This requires 4 * 8 = 32 schoolbook multiplications.
    
    # Total rank = 56 + 32 = 88.
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_rank_unit(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            raise Exception("Rank overflow")
        for idx, val in u_coeffs.items():
            U[idx][col] += val
        for idx, val in v_coeffs.items():
            V[idx][col] += val
        for idx, val in w_coeffs.items():
            W[idx][col] += val
        col += 1
        
    def add_strassen_block(a_r1, a_c1, b_r1, b_c1, c_r1, c_c1):
        """
        Adds Strassen decomposition for A[a_r1:a_r1+2, a_c1:a_c1+2] * B[b_r1:b_r1+2, b_c1:b_c1+2] -> C[c_r1:c_r1+2, c_c1:c_c1+2]
        """
        u_a = a_r1 * m + a_c1
        u_b = a_r1 * m + a_c1 + 1
        u_c = (a_r1 + 1) * m + a_c1
        u_d = (a_r1 + 1) * m + a_c1 + 1
        
        v_e = b_r1 * p + b_c1
        v_f = b_r1 * p + b_c1 + 1
        v_g = (b_r1 + 1) * p + b_c1
        v_h = (b_r1 + 1) * p + b_c1 + 1
        
        w_c11 = b_c1 * n + c_r1
        w_c12 = (b_c1 + 1) * n + c_r1
        w_c21 = b_c1 * n + (c_r1 + 1)
        w_c22 = (b_c1 + 1) * n + (c_r1 + 1)
        
        # M1 = (a+d)(e+h)
        add_rank_unit({u_a: 1, u_d: 1}, {v_e: 1, v_h: 1}, {w_c11: 1, w_c22: 1})
        # M2 = (c+d)e
        add_rank_unit({u_c: 1, u_d: 1}, {v_e: 1}, {w_c21: 1, w_c22: -1})
        # M3 = a(f-h)
        add_rank_unit({u_a: 1}, {v_f: 1, v_h: -1}, {w_c12: 1, w_c22: 1})
        # M4 = d(g-e)
        add_rank_unit({u_d: 1}, {v_g: 1, v_e: -1}, {w_c11: 1, w_c21: 1})
        # M5 = (a+b)h
        add_rank_unit({u_a: 1, u_b: 1}, {v_h: 1}, {w_c11: -1, w_c12: 1})
        # M6 = (c-a)(e+f)
        add_rank_unit({u_c: 1, u_a: -1}, {v_e: 1, v_f: 1}, {w_c22: 1})
        # M7 = (b-d)(g+h)
        add_rank_unit({u_b: 1, u_d: -1}, {v_g: 1, v_h: 1}, {w_c11: 1})

    # Part 1: A[0:2, 0:2] * B[0:2, :]
    # B[0:2, :] is 2x8. We split it into four 2x2 blocks along the output dimension (columns of B, which are rows of C? No, B is 4x8, so rows are 0..3, cols 0..7).
    # B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6:8].
    for k in range(0, 8, 2):
        add_strassen_block(0, 0, 0, k, 0, k)
        
    # Part 1b: A[0:2, 2:4] * B[2:4, :]
    # B[2:4, 0:2], B[2:4, 2:4], B[2:4, 4:6], B[2:4, 6:8].
    for k in range(0, 8, 2):
        add_strassen_block(0, 2, 2, k, 0, k)

    # Part 2: A[2:3, :] * B[:, :]
    # A[2, 0] * B[0, :] + A[2, 1] * B[1, :] + A[2, 2] * B[2, :] + A[2, 3] * B[3, :]
    for j in range(4):
        for k in range(8):
            u_idx = 2 * m + j
            v_idx = j * p + k
            w_idx = k * n + 2
            add_rank_unit({u_idx: 1}, {v_idx: 1}, {w_idx: 1})

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
