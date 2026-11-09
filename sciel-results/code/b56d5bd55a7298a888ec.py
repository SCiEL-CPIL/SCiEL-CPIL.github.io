import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    
    # We split the problem into:
    # 1. 4x4x4 multiplication (C[:, 0:4] = A @ B[:, 0:4])
    # 2. 4x4x3 multiplication (C[:, 4:7] = A @ B[:, 4:7])
    
    # Part 1: Strassen 4x4x4 (Rank 49)
    # Part 2: Schoolbook 4x4x3 (Rank 48)
    
    total_rank = 49 + 48
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: Strassen 4x4x4 ---
    
    # Define the 7 M-terms for 4x4 Strassen.
    # Each M-term is defined by:
    # 1. P_blocks: list of (block_idx, coeff) for A. block_idx is 0..3 (00, 01, 10, 11)
    # 2. Q_blocks: list of (block_idx, coeff) for B.
    # 3. C_contrib: dict mapping C_block_idx (0..3) to sign.
    
    # Block indices:
    # 0: 00 (top-left)
    # 1: 01 (top-right)
    # 2: 10 (bottom-left)
    # 3: 11 (bottom-right)
    
    M_terms = [
        # M1: (A00+A11)(B00+B11)
        {
            'P': [(0, 1), (3, 1)],
            'Q': [(0, 1), (3, 1)],
            'C': {0: 1, 3: 1} # C00, C11
        },
        # M2: (A10+A11)B00
        {
            'P': [(2, 1), (3, 1)],
            'Q': [(0, 1)],
            'C': {1: 1, 3: -1} # C01, C11
        },
        # M3: A00(B01-B11)
        {
            'P': [(0, 1)],
            'Q': [(1, 1), (3, -1)],
            'C': {1: 1, 3: 1} # C01, C11
        },
        # M4: A11(B10-B00)
        {
            'P': [(3, 1)],
            'Q': [(2, 1), (0, -1)],
            'C': {0: -1, 2: 1, 3: -1} # C00, C10, C11
        },
        # M5: (A00+A01)B11
        {
            'P': [(0, 1), (1, 1)],
            'Q': [(3, 1)],
            'C': {2: 1, 3: 1} # C10, C11
        },
        # M6: (A10-A00)(B00+B01)
        {
            'P': [(2, 1), (0, -1)],
            'Q': [(0, 1), (1, 1)],
            'C': {0: 1} # C00
        },
        # M7: (A01-A11)(B10+B11)
        {
            'P': [(1, 1), (3, -1)],
            'Q': [(2, 1), (3, 1)],
            'C': {0: -1, 3: 1} # C00, C11
        }
    ]
    
    # Base 2x2x2 Strassen terms (S0..S6)
    # Each S-term is defined by:
    # 1. P_local: dict mapping local_idx (0..3) to coeff.
    # 2. Q_local: dict mapping local_idx (0..3) to coeff.
    # 3. R_contrib: dict mapping local_idx (0..3) to coeff.
    
    # Local indices: 0:00, 1:01, 2:10, 3:11
    
    S_terms = [
        # S0: (P00+P11)(Q00+Q11)
        {
            'P': {0: 1, 3: 1},
            'Q': {0: 1, 3: 1},
            'R': {0: 1, 3: 1} # R00, R11
        },
        # S1: (P10+P11)Q00
        {
            'P': {2: 1, 3: 1},
            'Q': {0: 1},
            'R': {1: 1, 3: -1} # R01, R11
        },
        # S2: P00(Q01-Q11)
        {
            'P': {0: 1},
            'Q': {1: 1, 3: -1},
            'R': {1: 1, 3: 1} # R01, R11
        },
        # S3: P11(Q10-Q00)
        {
            'P': {3: 1},
            'Q': {2: 1, 0: -1},
            'R': {0: -1, 2: 1, 3: -1} # R00, R10, R11
        },
        # S4: (P00+P01)Q11
        {
            'P': {0: 1, 1: 1},
            'Q': {3: 1},
            'R': {2: 1, 3: 1} # R10, R11
        },
        # S5: (P10-P00)(Q00+Q01)
        {
            'P': {2: 1, 0: -1},
            'Q': {0: 1, 1: 1},
            'R': {0: 1} # R00
        },
        # S6: (P01-P11)(Q10+Q11)
        {
            'P': {1: 1, 3: -1},
            'Q': {2: 1, 3: 1},
            'R': {0: -1, 3: 1} # R00, R11
        }
    ]
    
    rank_offset = 0
    for m_idx in range(7):
        m_def = M_terms[m_idx]
        for s_idx in range(7):
            s_def = S_terms[s_idx]
            
            # Combine P and Q definitions
            # For each local element in s_def['P'], we need to map to global A indices.
            # P_local element (li) corresponds to P[li//2, li%2].
            # In M_def, P is a sum of blocks.
            # Let's construct the global U, V, W for this specific scalar multiplication.
            
            u_coeff = {}
            v_coeff = {}
            w_coeff = {}
            
            # Process P part
            for p_local_idx, p_coeff in s_def['P'].items():
                # p_local_idx is 0..3.
                # This corresponds to P[pi, pj] where pi = p_local_idx // 2, pj = p_local_idx % 2.
                # In the M-term, P is formed from A blocks.
                # The element P[pi, pj] is the sum of A_block[pi, pj] for each block in m_def['P'].
                # A_block is 2x2.
                # So A_block[pi, pj] corresponds to global A index:
                # i_global = 2 * block_row + pi
                # j_global = 2 * block_col + pj
                
                for block_idx, block_coeff in m_def['P']:
                    # block_idx 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                    br = block_idx // 2
                    bc = block_idx % 2
                    
                    i_global = 2 * br + (p_local_idx // 2)
                    j_global = 2 * bc + (p_local_idx % 2)
                    
                    global_idx = i_global * 4 + j_global
                    u_coeff[global_idx] = u_coeff.get(global_idx, 0) + p_coeff * block_coeff
            
            # Process Q part
            for q_local_idx, q_coeff in s_def['Q'].items():
                # q_local_idx is 0..3.
                # Corresponds to Q[qi, qj] where qi = q_local_idx // 2, qj = q_local_idx % 2.
                # In M-term, Q is formed from B blocks.
                for block_idx, block_coeff in m_def['Q']:
                    br = block_idx // 2
                    bc = block_idx % 2
                    
                    i_global = 2 * br + (q_local_idx // 2)
                    j_global = 2 * bc + (q_local_idx % 2)
                    
                    # B index: j * p + k. Here j is the row of B (which is col of A), k is col of B.
                    # Wait, B is 4x4 in this part.
                    # B rows correspond to A cols (0..3).
                    # B cols correspond to C cols (0..3).
                    # So B index is j_global * 4 + k_global?
                    # In the term, Q is 2x2.
                    # Q[qi, qj] corresponds to B element at row qi+2*br, col qj+2*bc.
                    # So V index is (2*br + qi) * 4 + (2*bc + qj).
                    
                    v_row = 2 * br + (q_local_idx // 2)
                    v_col = 2 * bc + (q_local_idx % 2)
                    global_idx = v_row * 4 + v_col
                    v_coeff[global_idx] = v_coeff.get(global_idx, 0) + q_coeff * block_coeff
            
            # Process W part
            # R_contrib tells us which local R elements are affected.
            # R is the result of P Q, which is a 2x2 matrix.
            # R[ri, rj] corresponds to C block element.
            # In M-term, R contributes to specific C blocks with specific signs.
            for r_local_idx, r_coeff in s_def['R'].items():
                # r_local_idx 0..3.
                # ri = r_local_idx // 2, rj = r_local_idx % 2.
                # This R element is part of the 2x2 result M_m.
                # M_m contributes to C blocks.
                # Which C blocks? The ones in m_def['C'].
                # But wait, the M-term result M_m is a 2x2 matrix.
                # The local R[ri, rj] corresponds to the element at position (ri, rj) within the 2x2 block M_m.
                # The block M_m is added to C blocks.
                # Does M_m map to specific C blocks?
                # Yes, the M-term definition says M_m contributes to C blocks in m_def['C'].
                # But does the *entire* 2x2 matrix M_m go to those blocks?
                # Yes, Strassen works on blocks. M_1 is a 2x2 matrix that is added to C_00 and C_11.
                # So R[ri, rj] (which is a scalar) is added to C_block[ri, rj] for each C_block in m_def['C'].
                
                for c_block_idx, c_sign in m_def['C'].items():
                    cbr = c_block_idx // 2
                    cbc = c_block_idx % 2
                    
                    i_global = 2 * cbr + (r_local_idx // 2)
                    k_global = 2 * cbc + (r_local_idx % 2)
                    
                    # W index: k * n + i
                    global_idx = k_global * 4 + i_global
                    w_coeff[global_idx] = w_coeff.get(global_idx, 0) + r_coeff * c_sign
            
            # Add to U, V, W
            for idx, val in u_coeff.items():
                U[idx][rank_offset] += val
            for idx, val in v_coeff.items():
                V[idx][rank_offset] += val
            for idx, val in w_coeff.items():
                W[idx][rank_offset] += val
            
            rank_offset += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # --- Part 2: Schoolbook 4x4x3 ---
    # C[:, 4:7] = A @ B[:, 4:7]
    # A is 4x4. B_part is 4x3. C_part is 4x3.
    # For each i in 0..3, k in 4..6 (global k), j in 0..3:
    # C[i, k] += A[i, j] * B[j, k]
    # U index: i*4 + j
    # V index: j*7 + k
    # W index: k*4 + i
    
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                u_idx = i * 4 + j
                v_idx = j * 7 + k
                w_idx = k * 4 + i
                U[u_idx][rank_offset] = 1
                V[v_idx][rank_offset] = 1
                W[w_idx][rank_offset] = 1
                rank_offset += 1
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
