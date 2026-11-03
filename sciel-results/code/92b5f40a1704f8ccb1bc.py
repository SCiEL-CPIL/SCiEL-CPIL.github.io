"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
_START_TIME = time.time()
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strategy:
    # C = A * B where A is 2x4, B is 4x5, C is 2x5.
    # Split A into A1 (cols 0,1) and A2 (cols 2,3).
    # Split B into B1 (rows 0,1) and B2 (rows 2,3).
    # C = A1*B1 + A2*B2.
    # Each term is a 2x2 matrix (A_block) times a 2x5 matrix (B_block).
    
    # Decomposition of 2x2 * 2x5:
    # Split B_block (2x5) into B_left (2x2, cols 0,1), B_mid (2x2, cols 2,3), B_right (2x1, col 4).
    # C_block = A_block * B_left + A_block * B_mid + A_block * B_right.
    
    # 1. A_block * B_left (2x2 * 2x2): Use Strassen (rank 7).
    # 2. A_block * B_mid (2x2 * 2x2): Use Strassen (rank 7).
    # 3. A_block * B_right (2x2 * 2x1): Use Schoolbook (rank 4).
    #    C_0 = A_00 B_0 + A_01 B_1
    #    C_1 = A_10 B_0 + A_11 B_1
    #    Terms: (A00,B0), (A01,B1), (A10,B0), (A11,B1).
    
    # Total rank per 2x2 * 2x5 block: 7 + 7 + 4 = 18.
    # Total rank for 2x4 * 4x5: 18 + 18 = 36.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    col = 0

    def add_rank1_term(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
        col += 1

    def add_strassen_block(a_base_i, a_base_j, b_base_i, b_base_j, c_base_i, c_base_j):
        # a_base_i, a_base_j: global start index of A block (2x2)
        # b_base_i, b_base_j: global start index of B block (2x2)
        # c_base_i, c_base_j: global start index of C block (2x2)
        
        # Local indices 0,1,2,3 for 2x2 matrix
        # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        
        def map_a(local):
            return (a_base_i + local // 2) * m + (a_base_j + local % 2)
        def map_b(local):
            return (b_base_i + local // 2) * p + (b_base_j + local % 2)
        def map_c(local):
            # W index is k*n + i.
            # C local (i, j) -> global i = c_base_i + i, k = c_base_j + j
            # Index = k * n + i
            i = c_base_i + local // 2
            k = c_base_j + local % 2
            return k * n + i

        # M1 = (A00 + A11) * (B00 + B11)
        # C00 += M1, C11 += M1
        u = {map_a(0): 1, map_a(3): 1}
        v = {map_b(0): 1, map_b(3): 1}
        w = {map_c(0): 1, map_c(3): 1}
        add_rank1_term(u, v, w)

        # M2 = (A10 + A11) * B00
        # C10 += M2, C11 -= M2
        u = {map_a(2): 1, map_a(3): 1}
        v = {map_b(0): 1}
        w = {map_c(2): 1, map_c(3): -1}
        add_rank1_term(u, v, w)

        # M3 = A00 * (B01 - B11)
        # C01 += M3, C11 += M3
        u = {map_a(0): 1}
        v = {map_b(1): 1, map_b(3): -1}
        w = {map_c(1): 1, map_c(3): 1}
        add_rank1_term(u, v, w)

        # M4 = A11 * (B10 - B00)
        # C00 += M4, C10 += M4
        u = {map_a(3): 1}
        v = {map_b(2): 1, map_b(0): -1}
        w = {map_c(0): 1, map_c(2): 1}
        add_rank1_term(u, v, w)

        # M5 = (A00 + A01) * B11
        # C00 -= M5, C01 += M5
        u = {map_a(0): 1, map_a(1): 1}
        v = {map_b(3): 1}
        w = {map_c(0): -1, map_c(1): 1}
        add_rank1_term(u, v, w)

        # M6 = (A10 - A00) * (B00 + B01)
        # C11 += M6
        u = {map_a(2): 1, map_a(0): -1}
        v = {map_b(0): 1, map_b(1): 1}
        w = {map_c(3): 1}
        add_rank1_term(u, v, w)

        # M7 = (A01 - A11) * (B10 + B11)
        # C00 += M7
        u = {map_a(1): 1, map_a(3): -1}
        v = {map_b(2): 1, map_b(3): 1}
        w = {map_c(0): 1}
        add_rank1_term(u, v, w)

    def add_schoolbook_2x2_1(a_base_i, a_base_j, b_base_i, b_base_j, c_base_i, c_base_j):
        # B is 2x1. Local B indices 0, 1.
        # C is 2x1. Local C indices 0, 1.
        
        def map_a(local):
            return (a_base_i + local // 2) * m + (a_base_j + local % 2)
        def map_b(local):
            # B is 2x1. p=5.
            # Global B index: row * p + col
            return (b_base_i + local) * p + b_base_j
        def map_c(local):
            # C is 2x1.
            # Global C index: k * n + i
            i = c_base_i + local
            k = c_base_j
            return k * n + i

        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        
        # Term 1: A00 * B0 -> C0
        u = {map_a(0): 1}
        v = {map_b(0): 1}
        w = {map_c(0): 1}
        add_rank1_term(u, v, w)

        # Term 2: A01 * B1 -> C0
        u = {map_a(1): 1}
        v = {map_b(1): 1}
        w = {map_c(0): 1}
        add_rank1_term(u, v, w)

        # Term 3: A10 * B0 -> C1
        u = {map_a(2): 1}
        v = {map_b(0): 1}
        w = {map_c(1): 1}
        add_rank1_term(u, v, w)

        # Term 4: A11 * B1 -> C1
        u = {map_a(3): 1}
        v = {map_b(1): 1}
        w = {map_c(1): 1}
        add_rank1_term(u, v, w)

    # Block 1: A1 (rows 0,1; cols 0,1), B1 (rows 0,1; cols 0,4)
    # A1 base: i=0, j=0
    # B1 base: i=0, j=0
    # C base: i=0, j=0
    
    # A1 * B1[0:2, 0:2] (cols 0,1)
    add_strassen_block(0, 0, 0, 0, 0, 0)
    # A1 * B1[0:2, 2:4] (cols 2,3)
    add_strassen_block(0, 0, 0, 2, 0, 2)
    # A1 * B1[0:2, 4:5] (col 4)
    add_schoolbook_2x2_1(0, 0, 0, 4, 0, 4)
    
    # Block 2: A2 (rows 0,1; cols 2,3), B2 (rows 2,3; cols 0,4)
    # A2 base: i=0, j=2
    # B2 base: i=2, j=0
    # C base: i=0, j=0
    
    # A2 * B2[2:4, 0:2] (cols 0,1)
    add_strassen_block(0, 2, 2, 0, 0, 0)
    # A2 * B2[2:4, 2:4] (cols 2,3)
    add_strassen_block(0, 2, 2, 2, 0, 2)
    # A2 * B2[2:4, 4:5] (col 4)
    add_schoolbook_2x2_1(0, 2, 2, 4, 0, 4)
    
    # Verify the decomposition is correct
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # Compute C from tensor decomposition
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            total = 0.0
            for j in range(m):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total += sum(U_np[u_idx, r] * V_np[v_idx, r] * W_np[w_idx, r] for r in range(rank))
            C_computed[i, k] = total
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # If verification fails, fall back to schoolbook rank 40
        rank = 40
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
