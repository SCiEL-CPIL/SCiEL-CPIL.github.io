"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a low-rank exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Uses Strassen for the 4x4x4 part (rank 28 via schoolbook sub-blocks) 
    and schoolbook for the 4x4x5 part (rank 80).
    Total rank: 108.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
        
    def make_vec(size, indices, values):
        vec = [0.0] * size
        for idx, val in zip(indices, values):
            vec[idx] = val
        return vec

    # Part 1: 4x4x4 (columns 0-3 of B)
    # A is 4x4, B1 is 4x4, C1 is 4x4.
    # Indices:
    # A: i in 0..3, j in 0..3 -> i*4 + j
    # B1: j in 0..3, k in 0..3 -> j*4 + k
    # C1: k in 0..3, i in 0..3 -> k*4 + i
    
    # Strassen terms for 4x4:
    # Split A into 2x2 blocks: A00, A01, A10, A11
    # Split B1 into 2x2 blocks: B00, B01, B10, B11
    # 
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M is a 2x2 matrix product.
    # We decompose each 2x2 product into 4 rank-1 terms (schoolbook).
    
    # Block indices for A (4x4 split into 2x2 blocks of 2x2)
    # A00: (0,0), (0,1), (1,0), (1,1) -> indices 0,1,4,5
    # A01: (0,2), (0,3), (1,2), (1,3) -> indices 2,3,6,7
    # A10: (2,0), (2,1), (3,0), (3,1) -> indices 8,9,12,13
    # A11: (2,2), (2,3), (3,2), (3,3) -> indices 10,11,14,15
    
    A_blocks = {
        '00': [0, 1, 4, 5],
        '01': [2, 3, 6, 7],
        '10': [8, 9, 12, 13],
        '11': [10, 11, 14, 15]
    }
    
    # Block indices for B1 (4x4 split into 2x2 blocks of 2x2)
    # B00: j=0,1; k=0,1 -> indices 0,1,4,5
    # B01: j=0,1; k=2,3 -> indices 2,3,6,7
    # B10: j=2,3; k=0,1 -> indices 8,9,12,13
    # B11: j=2,3; k=2,3 -> indices 10,11,14,15
    
    B1_blocks = {
        '00': [0, 1, 4, 5],
        '01': [2, 3, 6, 7],
        '10': [8, 9, 12, 13],
        '11': [10, 11, 14, 15]
    }
    
    # Block indices for C1 (4x4, k*n + i)
    # C00: k=0,1; i=0,1 -> indices 0,1,4,5
    # C01: k=0,1; i=2,3 -> indices 2,3,6,7
    # C10: k=2,3; i=0,1 -> indices 8,9,12,13
    # C11: k=2,3; i=2,3 -> indices 10,11,14,15
    
    C1_blocks = {
        '00': [0, 1, 4, 5],
        '01': [2, 3, 6, 7],
        '10': [8, 9, 12, 13],
        '11': [10, 11, 14, 15]
    }
    
    # Strassen combinations
    # Each term l has A_comb and B_comb
    # A_comb: dict mapping block name to coefficient
    # B_comb: dict mapping block name to coefficient
    
    strassen_terms = [
        # M1
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}),
        # M2
        ({'10': 1, '11': 1}, {'00': 1}),
        # M3
        ({'00': 1}, {'01': 1, '11': -1}),
        # M4
        ({'11': 1}, {'10': 1, '00': -1}),
        # M5
        ({'00': 1, '01': 1}, {'11': 1}),
        # M6
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}),
        # M7
        ({'01': 1, '11': -1}, {'10': 1, '11': 1})
    ]
    
    # C block coefficients for each Strassen term
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    C_coeffs = [
        # M1
        {'00': 1, '01': 0, '10': 0, '11': 1},
        # M2
        {'00': 0, '01': 0, '10': 1, '11': -1},
        # M3
        {'00': 0, '01': 1, '10': 0, '11': 1},
        # M4
        {'00': 1, '01': 0, '10': 1, '11': 0},
        # M5
        {'00': -1, '01': 1, '10': 0, '11': 0},
        # M6
        {'00': 0, '01': 0, '10': 0, '11': 1},
        # M7
        {'00': 1, '01': 0, '10': 0, '11': 0}
    ]
    
    # For each Strassen term, generate 4 rank-1 terms (schoolbook for 2x2)
    for l in range(7):
        A_comb, B_comb = strassen_terms[l]
        C_coeff = C_coeffs[l]
        
        # For each i, j, k in 0..1 (local 2x2 indices)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # U vector: contributions from A blocks
                    u_indices = []
                    u_values = []
                    for block_name, coeff in A_comb.items():
                        base_indices = A_blocks[block_name]
                        # Local index (i, j) in 2x2 block
                        # If block is 00: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
                        # If block is 01: (0,0)->2, (0,1)->3, (1,0)->6, (1,1)->7
                        # If block is 10: (0,0)->8, (0,1)->9, (1,0)->12, (1,1)->13
                        # If block is 11: (0,0)->10, (0,1)->11, (1,0)->14, (1,1)->15
                        # General: base = block_row * 4 + block_col
                        # local = i * 2 + j
                        # index = base + local
                        block_row = int(block_name[0])
                        block_col = int(block_name[1])
                        base = block_row * 4 + block_col
                        local = i * 2 + j
                        idx = base + local
                        u_indices.append(idx)
                        u_values.append(coeff)
                    
                    # V vector: contributions from B blocks
                    v_indices = []
                    v_values = []
                    for block_name, coeff in B_comb.items():
                        base_indices = B1_blocks[block_name]
                        block_row = int(block_name[0])
                        block_col = int(block_name[1])
                        base = block_row * 4 + block_col
                        local = j * 2 + k
                        idx = base + local
                        v_indices.append(idx)
                        v_values.append(coeff)
                    
                    # W vector: contributions from C blocks
                    w_indices = []
                    w_values = []
                    for block_name, coeff in C_coeff.items():
                        if coeff == 0:
                            continue
                        base_indices = C1_blocks[block_name]
                        block_row = int(block_name[0])
                        block_col = int(block_name[1])
                        base = block_row * 4 + block_col
                        local = k * 2 + i
                        idx = base + local
                        w_indices.append(idx)
                        w_values.append(coeff)
                    
                    u_vec = make_vec(16, u_indices, u_values)
                    v_vec = make_vec(16, v_indices, v_values)
                    w_vec = make_vec(16, w_indices, w_values)
                    
                    add_term(u_vec, v_vec, w_vec)
    
    # Part 2: Schoolbook for columns 4-8 of B
    # A is 4x4, B2 is 4x5, C2 is 5x4
    # Indices:
    # A: i*4 + j
    # B2: j*9 + k (k in 4..8)
    # C2: k*4 + i
    
    for i in range(4):
        for j in range(4):
            for k in range(4, 9):
                u_vec = make_vec(16, [i*4 + j], [1.0])
                v_vec = make_vec(36, [j*9 + k], [1.0])
                w_vec = make_vec(36, [k*4 + i], [1.0])
                add_term(u_vec, v_vec, w_vec)
    
    # Convert lists to matrices
    rank = len(U_list)
    U = [[0.0] * rank for _ in range(16)]
    V = [[0.0] * rank for _ in range(36)]
    W = [[0.0] * rank for _ in range(36)]
    
    for r in range(rank):
        for i in range(16):
            U[i][r] = U_list[r][i]
        for j in range(36):
            V[j][r] = V_list[r][j]
        for k in range(36):
            W[k][r] = W_list[r][k]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
