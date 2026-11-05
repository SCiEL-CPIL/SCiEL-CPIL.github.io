import numpy as np
import time

def construct():
    """
    Constructs a rank-85 exact tensor decomposition for 4x5 * 5x5 matrix multiplication.
    Uses recursive Strassen for the core 4x4x4 block (rank 49) and schoolbook for the tails.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 4, 5, 5
    
    # Total rank: 49 (core) + 16 (tail1) + 16 (tail2) + 4 (tail3) = 85
    rank = 85
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to add a term
    def add_term(i_idx, j_idx, j_prime, k_prime, w_i, w_k, col):
        """
        Adds a rank-1 term defined by:
        U[i_idx, j_idx, col] = 1
        V[j_prime, k_prime, col] = 1
        W[k_prime, w_i, col] = 1
        """
        U[i_idx * m + j_idx, col] = 1
        V[j_prime * p + k_prime, col] = 1
        W[k_prime * n + w_i, col] = 1

    # --- Part 1: Core 4x4x4 Strassen (Rank 49) ---
    # Indices for A: i in 0..3, j in 0..3
    # Indices for B: j in 0..3, k in 0..3
    # Indices for C: i in 0..3, k in 0..3
    
    # Strassen 2x2x2 factors (rank 7)
    # We apply this to 2x2 blocks of the 4x4 matrices.
    # 7 terms for 2x2x2.
    # 4x4 can be seen as 4 2x2 blocks.
    # The 7 Strassen formulas are applied to the 2x2 block structure.
    
    # Standard Strassen 2x2x2 coefficients
    # u, v, w are vectors of size 4 (2x2 matrices)
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    s_u = [
        [1, 0, 1, 0], # T1: a00+a11
        [0, 0, 1, 1], # T2: a10+a11
        [1, 0, 0, 0], # T3: a00
        [0, 0, 0, 1], # T4: a11
        [1, 1, 0, 0], # T5: a00+a01
        [1, 0, -1, 0],# T6: a10-a00
        [0, 1, 0, -1] # T7: a01-a11
    ]
    s_v = [
        [1, 0, 1, 0], # T1: b00+b11
        [1, 0, 0, 0], # T2: b00
        [0, 1, 0, -1],# T3: b01-b11
        [0, 0, 1, -1],# T4: b10-b00
        [0, 0, 0, 1], # T5: b11
        [1, 1, 0, 0], # T6: b00+b01
        [0, 0, 1, 1]  # T7: b10+b11
    ]
    s_w = [
        [1, 0, 1, 0], # T1: c00+c11
        [0, 0, 1, -1],# T2: c10-c11? No.
        # Let's use the standard mapping:
        # C00 = M1 - M4 + M6
        # C01 = M5 + M7
        # C10 = M2 + M4
        # C11 = M1 - M3 + M5
        # 
        # M1: a00+a11, b00+b11 -> w: [1,0,0,1] (C00, C11)
        # M2: a10+a11, b00 -> w: [0,0,1,0] (C10)
        # M3: a00, b01-b11 -> w: [0,1,0,-1] (C01, C11)
        # M4: a11, b10-b00 -> w: [1,0,-1,0] (C00, C10)
        # M5: a00+a01, b11 -> w: [-1,1,0,0] (C00, C01)
        # M6: a10-a00, b00+b01 -> w: [0,0,0,1] (C11)
        # M7: a01-a11, b10+b11 -> w: [0,1,0,0] (C01)
        
    ]
    
    # Correct W vectors for Strassen 2x2:
    # M1: C00 += 1, C11 += 1 => [1, 0, 0, 1]
    # M2: C10 += 1 => [0, 0, 1, 0]
    # M3: C01 += 1, C11 -= 1 => [0, 1, 0, -1]
    # M4: C00 -= 1, C10 += 1 => [1, 0, -1, 0]
    # M5: C00 -= 1, C01 += 1 => [-1, 1, 0, 0]
    # M6: C11 += 1 => [0, 0, 0, 1]
    # M7: C01 += 1 => [0, 1, 0, 0]
    
    s_w = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, -1],
        [1, 0, -1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0]
    ]
    
    col = 0
    # The 4x4 matrix is composed of four 2x2 blocks:
    # A: A00, A01, A10, A11
    # B: B00, B01, B10, B11
    # C: C00, C01, C10, C11
    
    # We apply the 7 Strassen terms to the blocks.
    # For each of the 7 terms, we distribute the coefficients to the 4 blocks.
    # Since the blocks are disjoint, we can just add the terms for each block combination.
    # Actually, the standard 4x4 Strassen uses 7 products of 2x2 matrices.
    # Each 2x2 product uses 7 scalar multiplications.
    # Total 49 scalar multiplications.
    
    # We define the block structure for the 7 terms.
    # Term 1: (A00+A11)(B00+B11)
    # Term 2: (A10+A11)B00
    # Term 3: A00(B01-B11)
    # Term 4: A11(B10-B00)
    # Term 5: (A00+A01)B11
    # Term 6: (A10-A00)(B00+B01)
    # Term 7: (A01-A11)(B10+B11)
    
    # We will iterate through the 7 block-level terms.
    # For each block-level term, we use the 7 scalar-level Strassen factors.
    
    block_terms = [
        # (A_blocks, B_blocks, C_blocks)
        # A_blocks: {block_idx: coeff}
        # Block indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    for b_term_idx, (a_blocks, b_blocks, c_blocks) in enumerate(block_terms):
        if time.time() - start_time > deadline:
            break
        for s_idx in range(7):
            if time.time() - start_time > deadline:
                break
            # Distribute u
            for a_blk, a_coeff in a_blocks.items():
                if a_coeff == 0: continue
                for local_idx, u_val in enumerate(s_u[s_idx]):
                    if u_val == 0: continue
                    lr = local_idx // 2
                    lc = local_idx % 2
                    gi = 2 * (a_blk // 2) + lr
                    gj = 2 * (a_blk % 2) + lc
                    U[gi * 5 + gj, col] += a_coeff * u_val
            
            # Distribute v
            for b_blk, b_coeff in b_blocks.items():
                if b_coeff == 0: continue
                for local_idx, v_val in enumerate(s_v[s_idx]):
                    if v_val == 0: continue
                    lr = local_idx // 2
                    lc = local_idx % 2
                    gj = 2 * (b_blk // 2) + lr
                    gk = 2 * (b_blk % 2) + lc
                    V[gj * 5 + gk, col] += b_coeff * v_val
            
            # Distribute w
            for c_blk, c_coeff in c_blocks.items():
                if c_coeff == 0: continue
                for local_idx, w_val in enumerate(s_w[s_idx]):
                    if w_val == 0: continue
                    lr = local_idx // 2
                    lc = local_idx % 2
                    gi = 2 * (c_blk // 2) + lr
                    gk = 2 * (c_blk % 2) + lc
                    W[gk * 4 + gi, col] += c_coeff * w_val
            
            col += 1

    # --- Part 2: Tail A1 (4x4) * B12 (4x1) -> C12 (4x1) ---
    # A[i, j] for i in 0..3, j in 0..3
    # B[j, 4] for j in 0..3
    # C[i, 4] for i in 0..3
    for i in range(4):
        if time.time() - start_time > deadline: break
        for j in range(4):
            U[i*5 + j, col] = 1
            V[j*5 + 4, col] = 1
            W[4*4 + i, col] = 1
            col += 1

    # --- Part 3: Tail A2 (4x1) * B21 (1x4) -> C21 (4x4) ---
    # A[i, 4] for i in 0..3
    # B[4, k] for k in 0..3
    # C[i, k] for i in 0..3, k in 0..3
    for i in range(4):
        if time.time() - start_time > deadline: break
        for k in range(4):
            U[i*5 + 4, col] = 1
            V[4*5 + k, col] = 1
            W[k*4 + i, col] = 1
            col += 1

    # --- Part 4: Tail A2 (4x1) * B22 (1x1) -> C22 (4x1) ---
    # A[i, 4] for i in 0..3
    # B[4, 4]
    # C[i, 4] for i in 0..3
    for i in range(4):
        if time.time() - start_time > deadline: break
        U[i*5 + 4, col] = 1
        V[4*5 + 4, col] = 1
        W[4*4 + i, col] = 1
        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
