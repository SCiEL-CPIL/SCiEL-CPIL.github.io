"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Strategy: Decompose 4x4 * 4x5 into:
    #   4x4 * 4x4 (first 4 columns) + 4x4 * 4x1 (last column)
    # Use rank-49 Strassen for 4x4*4x4, rank-16 schoolbook for 4x4*4x1
    # Total rank = 49 + 16 = 65
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c
    
    col = 0
    
    # --- Part 1: 4x4 * 4x4 using Strassen (rank 49) ---
    # We'll use a two-level Strassen decomposition.
    # Level 1: Split 4x4 into 2x2 blocks, use 7 multiplications of 2x2*2x2
    # Level 2: Each 2x2*2x2 uses Strassen's 7 scalar multiplications
    # Total: 7 * 7 = 49 terms
    
    # Strassen for 2x2 * 2x2:
    # Given P = [[p00, p01], [p10, p11]], Q = [[q00, q01], [q10, q11]]
    # 7 products:
    # M0 = (p00 + p11) * (q00 + q11)
    # M1 = (p10 + p11) * q00
    # M2 = p00 * (q01 - q11)
    # M3 = p11 * (q10 - q00)
    # M4 = (p00 + p01) * q11
    # M5 = (p10 - p00) * (q00 + q01)
    # M6 = (p01 - p11) * (q10 + q11)
    # Result:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # For each of the 7 Strassen products at level 1, we need to express
    # the linear combinations of A and B blocks, then for each scalar
    # multiplication within that, express the linear combinations of A and B elements.
    
    # Block structure for 4x4:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # B (4x4 part) = [[B00, B01], [B10, B11]] where each is 2x2
    # B00: rows 0-1, cols 0-1
    # B01: rows 0-1, cols 2-3
    # B10: rows 2-3, cols 0-1
    # B11: rows 2-3, cols 2-3
    
    # Level 1 Strassen products:
    # S0 = (A00 + A11) * (B00 + B11)
    # S1 = (A10 + A11) * B00
    # S2 = A00 * (B01 - B11)
    # S3 = A11 * (B10 - B00)
    # S4 = (A00 + A01) * B11
    # S5 = (A10 - A00) * (B00 + B01)
    # S6 = (A01 - A11) * (B10 + B11)
    
    # Result C (4x4):
    # C00 = S0 + S3 - S4 + S6  (top-left 2x2)
    # C01 = S2 + S4             (top-right 2x2)
    # C10 = S1 + S3             (bottom-left 2x2)
    # C11 = S0 - S1 + S2 + S5  (bottom-right 2x2)
    
    # For each level-1 product Si, we apply level-2 Strassen.
    # Each Si is a product of two 2x2 matrices (linear combinations of A and B blocks).
    
    # Helper: get the elements of a 2x2 block of A
    # block_idx: 0=A00, 1=A01, 2=A10, 3=A11
    # local (r,c) in 2x2 -> global (i,j)
    def a_block_elem(block_idx, r, c):
        if block_idx == 0:  # A00
            return (r, c)
        elif block_idx == 1:  # A01
            return (r, 2 + c)
        elif block_idx == 2:  # A10
            return (2 + r, c)
        elif block_idx == 3:  # A11
            return (2 + r, 2 + c)
    
    # Helper: get the elements of a 2x2 block of B (for 4x4 part, cols 0-3)
    def b_block_elem(block_idx, r, c):
        if block_idx == 0:  # B00
            return (r, c)
        elif block_idx == 1:  # B01
            return (r, 2 + c)
        elif block_idx == 2:  # B10
            return (2 + r, c)
        elif block_idx == 3:  # B11
            return (2 + r, 2 + c)
    
    # Helper: get the elements of a 2x2 block of C (for 4x4 part, cols 0-3)
    # C is 4x5, but we're only using cols 0-3 for this part
    def c_block_elem(block_idx, r, c):
        if block_idx == 0:  # C00
            return (r, c)
        elif block_idx == 1:  # C01
            return (r, 2 + c)
        elif block_idx == 2:  # C10
            return (2 + r, c)
        elif block_idx == 3:  # C11
            return (2 + r, 2 + c)
    
    # Level 1 Strassen: define the 7 products
    # Each product is (P) * (Q) where P and Q are 2x2 matrices
    # P is a linear combination of A blocks, Q is a linear combination of B blocks
    
    # For each level-1 product, define:
    # p_blocks: list of (block_idx, coeff) for A
    # q_blocks: list of (block_idx, coeff) for B
    # c_contrib: dict mapping c_block_idx -> coeff for how this product contributes to C
    
    level1_products = [
        # S0 = (A00 + A11) * (B00 + B11)
        {
            'p_blocks': [(0, 1), (3, 1)],
            'q_blocks': [(0, 1), (3, 1)],
            'c_contrib': {0: 1, 3: 1}  # C00 += S0, C11 += S0
        },
        # S1 = (A10 + A11) * B00
        {
            'p_blocks': [(2, 1), (3, 1)],
            'q_blocks': [(0, 1)],
            'c_contrib': {2: 1, 3: -1}  # C10 += S1, C11 -= S1
        },
        # S2 = A00 * (B01 - B11)
        {
            'p_blocks': [(0, 1)],
            'q_blocks': [(1, 1), (3, -1)],
            'c_contrib': {1: 1, 3: 1}  # C01 += S2, C11 += S2
        },
        # S3 = A11 * (B10 - B00)
        {
            'p_blocks': [(3, 1)],
            'q_blocks': [(2, 1), (0, -1)],
            'c_contrib': {0: 1, 2: 1}  # C00 += S3, C10 += S3
        },
        # S4 = (A00 + A01) * B11
        {
            'p_blocks': [(0, 1), (1, 1)],
            'q_blocks': [(3, 1)],
            'c_contrib': {0: -1, 1: 1}  # C00 -= S4, C01 += S4
        },
        # S5 = (A10 - A00) * (B00 + B01)
        {
            'p_blocks': [(2, 1), (0, -1)],
            'q_blocks': [(0, 1), (1, 1)],
            'c_contrib': {3: 1}  # C11 += S5
        },
        # S6 = (A01 - A11) * (B10 + B11)
        {
            'p_blocks': [(1, 1), (3, -1)],
            'q_blocks': [(2, 1), (3, 1)],
            'c_contrib': {0: 1}  # C00 += S6
        },
    ]
    
    # Level 2 Strassen: 7 products for 2x2 * 2x2
    # For a 2x2 matrix P = [[p00, p01], [p10, p11]] and Q = [[q00, q01], [q10, q11]]
    # The 7 products and their contributions to the 2x2 result:
    
    level2_strassen = [
        # M0 = (p00 + p11) * (q00 + q11)
        {
            'p_elems': [(0, 1), (3, 1)],  # p00, p11
            'q_elems': [(0, 1), (3, 1)],  # q00, q11
            'c_contrib': {0: 1, 3: 1}  # result[0,0] += M0, result[1,1] += M0
        },
        # M1 = (p10 + p11) * q00
        {
            'p_elems': [(2, 1), (3, 1)],  # p10, p11
            'q_elems': [(0, 1)],          # q00
            'c_contrib': {2: 1, 3: -1}   # result[1,0] += M1, result[1,1] -= M1
        },
        # M2 = p00 * (q01 - q11)
        {
            'p_elems': [(0, 1)],          # p00
            'q_elems': [(1, 1), (3, -1)], # q01, q11
            'c_contrib': {1: 1, 3: 1}    # result[0,1] += M2, result[1,1] += M2
        },
        # M3 = p11 * (q10 - q00)
        {
            'p_elems': [(3, 1)],          # p11
            'q_elems': [(2, 1), (0, -1)], # q10, q00
            'c_contrib': {0: 1, 2: 1}    # result[0,0] += M3, result[1,0] += M3
        },
        # M4 = (p00 + p01) * q11
        {
            'p_elems': [(0, 1), (1, 1)],  # p00, p01
            'q_elems': [(3, 1)],          # q11
            'c_contrib': {0: -1, 1: 1}   # result[0,0] -= M4, result[0,1] += M4
        },
        # M5 = (p10 - p00) * (q00 + q01)
        {
            'p_elems': [(2, 1), (0, -1)], # p10, p00
            'q_elems': [(0, 1), (1, 1)],  # q00, q01
            'c_contrib': {3: 1}           # result[1,1] += M5
        },
        # M6 = (p01 - p11) * (q10 + q11)
        {
            'p_elems': [(1, 1), (3, -1)], # p01, p11
            'q_elems': [(2, 1), (3, 1)],  # q10, q11
            'c_contrib': {0: 1}           # result[0,0] += M6
        },
    ]
    
    # Now generate all 49 terms
    for l1_idx, l1_prod in enumerate(level1_products):
        for l2_idx, l2_prod in enumerate(level2_strassen):
            u_list = []
            v_list = []
            w_list = []
            
            # Build u_list: linear combination of A elements
            for p_block_idx, p_block_coeff in l1_prod['p_blocks']:
                for p_elem_idx, p_elem_coeff in l2_prod['p_elems']:
                    # p_elem_idx: 0=p00, 1=p01, 2=p10, 3=p11
                    r = p_elem_idx // 2
                    c = p_elem_idx % 2
                    gi, gj = a_block_elem(p_block_idx, r, c)
                    coeff = p_block_coeff * p_elem_coeff
                    u_list.append((gi, gj, coeff))
            
            # Build v_list: linear combination of B elements
            for q_block_idx, q_block_coeff in l1_prod['q_blocks']:
                for q_elem_idx, q_elem_coeff in l2_prod['q_elems']:
                    # q_elem_idx: 0=q00, 1=q01, 2=q10, 3=q11
                    r = q_elem_idx // 2
                    c = q_elem_idx % 2
                    gj, gk = b_block_elem(q_block_idx, r, c)
                    coeff = q_block_coeff * q_elem_coeff
                    v_list.append((gj, gk, coeff))
            
            # Build w_list: linear combination of C elements
            # The result of this level-2 product contributes to a 2x2 block of the level-1 result
            # That level-1 result contributes to specific C blocks with specific coefficients
            for l2_c_block, l2_c_coeff in l2_prod['c_contrib'].items():
                for l1_c_block, l1_c_coeff in l1_prod['c_contrib'].items():
                    # Wait, this is wrong. The level-2 result is a 2x2 matrix that IS the level-1 product result.
                    # The level-1 product result contributes to C blocks.
                    # So for each element of the level-2 result, we need to map it to the corresponding C element.
                    pass
            
            # Let me redo the w_list construction.
            # The level-2 Strassen produces a 2x2 result R = [[r00, r01], [r10, r11]]
            # This R is the value of the level-1 product Si.
            # Si contributes to C blocks as specified in l1_prod['c_contrib'].
            # So R[0,0] contributes to C blocks in l1_prod['c_contrib'] with the same local position.
            
            # Actually, the level-1 product Si is a 2x2 matrix. Its elements are:
            # Si[0,0], Si[0,1], Si[1,0], Si[1,1]
            # These contribute to C as:
            # For each c_block_idx, c_coeff in l1_prod['c_contrib']:
            #   C_block[c_block_idx][0,0] += c_coeff * Si[0,0]
            #   C_block[c_block_idx][0,1] += c_coeff * Si[0,1]
            #   C_block[c_block_idx][1,0] += c_coeff * Si[1,0]
            #   C_block[c_block_idx][1,1] += c_coeff * Si[1,1]
            
            # The level-2 Strassen tells us how each Mj contributes to Si:
            # Si[0,0] = sum of Mj * coeff for j where c_contrib has 0
            # Si[0,1] = sum of Mj * coeff for j where c_contrib has 1
            # Si[1,0] = sum of Mj * coeff for j where c_contrib has 2
            # Si[1,1] = sum of Mj * coeff for j where c_contrib has 3
            
            # So for this specific l2_prod (which is Mj), it contributes to Si elements as per l2_prod['c_contrib'].
            # Then each Si element contributes to C as per l1_prod['c_contrib'].
            
            # Let me build w_list properly:
            w_list = []
            for si_elem_idx, si_coeff in l2_prod['c_contrib'].items():
                # si_elem_idx: 0=Si[0,0], 1=Si[0,1], 2=Si[1,0], 3=Si[1,1]
                si_r = si_elem_idx // 2
                si_c = si_elem_idx % 2
                for l1_c_block, l1_c_coeff in l1_prod['c_contrib'].items():
                    gi, gj = c_block_elem(l1_c_block, si_r, si_c)
                    coeff = l1_c_coeff * si_coeff
                    w_list.append((gj, gi, coeff))  # W[k*n+i], so (k, i) -> (gj, gi)
            
            add_term(col, u_list, v_list, w_list)
            col += 1
    
    # --- Part 2: 4x4 * 4x1 using schoolbook (rank 16) ---
    # A is 4x4, B_last is 4x1 (column 4 of the 4x5 matrix)
    # C_last is 4x1 (column 4 of the 4x5 result)
    # C[i,4] = sum_j A[i,j] * B[j,4]
    # So for each (i, j): term with U[i,j]=1, V[j,4]=1, W[4,i]=1
    
    for i in range(n):
        for j in range(m):
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]  # B[j,4]
            w_list = [(4, i, 1)]  # W[4*n+i] = W[4*4+i] = W[16+i]
            add_term(col, u_list, v_list, w_list)
            col += 1
    
    # Verify col == rank
    assert col == rank, f"Expected {rank} terms, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
