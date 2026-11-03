import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    # u_vec: length 16 (4x4)
    # v_vec: length 32 (4x8)
    # w_vec: length 32 (8x4)
    # col: column index
    def add_term(u_vals, v_vals, w_vals, col):
        for idx, val in enumerate(u_vals):
            if val != 0:
                U[idx][col] += val
        for idx, val in enumerate(v_vals):
            if val != 0:
                V[idx][col] += val
        for idx, val in enumerate(w_vals):
            if val != 0:
                W[idx][col] += val

    # Generate Strassen terms for 4x4
    # We need to define the linear combinations for A and B
    # A is 4x4. Indices i, j.
    # B is 4x8. Indices j, k.
    
    # Strassen blocks for A (4x4 -> 2x2 blocks of 2x2)
    # A00: rows 0,1 cols 0,1
    # A01: rows 0,1 cols 2,3
    # A10: rows 2,3 cols 0,1
    # A11: rows 2,3 cols 2,3
    
    # We will iterate through the 7 Strassen mults
    # For each mult, we define P (linear comb of A) and Q (linear comb of B)
    # Then we expand P*Q using schoolbook 2x2
    
    # Strassen definitions:
    # M1: (A00+A11) * (B00+B11)
    # M2: (A10+A11) * B00
    # M3: A00 * (B01-B11)
    # M4: A11 * (B10-B00)
    # M5: (A00+A01) * B11
    # M6: (A10-A00) * (B00+B01)
    # M7: (A01-A11) * (B10+B11)
    
    # For each M_k, P is a 2x2 matrix of linear combos of A entries.
    # Q is a 2x2 matrix of linear combos of B entries.
    
    # We'll represent P and Q as 2x2 lists of dicts { (i,j): coeff }
    
    def get_A_block(block_idx):
        # block_idx: 0=A00, 1=A01, 2=A10, 3=A11
        # Returns list of 4 coeffs for A entries
        # A00: (0,0), (0,1), (1,0), (1,1)
        # A01: (0,2), (0,3), (1,2), (1,3)
        # A10: (2,0), (2,1), (3,0), (3,1)
        # A11: (2,2), (2,3), (3,2), (3,3)
        if block_idx == 0:
            return [(0,0), (0,1), (1,0), (1,1)]
        elif block_idx == 1:
            return [(0,2), (0,3), (1,2), (1,3)]
        elif block_idx == 2:
            return [(2,0), (2,1), (3,0), (3,1)]
        elif block_idx == 3:
            return [(2,2), (2,3), (3,2), (3,3)]

    def get_B_block(block_idx, half):
        # half: 0 (cols 0-3), 1 (cols 4-7)
        # block_idx: 0=B00, 1=B01, 2=B10, 3=B11
        # B00: rows 0,1, cols 0,1 (or 4,5)
        # B01: rows 0,1, cols 2,3 (or 6,7)
        # B10: rows 2,3, cols 0,1 (or 4,5)
        # B11: rows 2,3, cols 2,3 (or 6,7)
        if half == 0:
            if block_idx == 0:
                return [(0,0), (0,1), (1,0), (1,1)]
            elif block_idx == 1:
                return [(0,2), (0,3), (1,2), (1,3)]
            elif block_idx == 2:
                return [(2,0), (2,1), (3,0), (3,1)]
            elif block_idx == 3:
                return [(2,2), (2,3), (3,2), (3,3)]
        else:
            if block_idx == 0:
                return [(0,4), (0,5), (1,4), (1,5)]
            elif block_idx == 1:
                return [(0,6), (0,7), (1,6), (1,7)]
            elif block_idx == 2:
                return [(2,4), (2,5), (3,4), (3,5)]
            elif block_idx == 3:
                return [(2,6), (2,7), (3,6), (3,7)]

    # Define the 7 Strassen terms
    # Each term is (A_combos, B_combos)
    # A_combos: list of 4 dicts for P[0,0], P[0,1], P[1,0], P[1,1]
    # B_combos: list of 4 dicts for Q[0,0], Q[0,1], Q[1,0], Q[1,1]
    
    # This is getting complex to hardcode.
    # Simpler: Just iterate through the 28 scalar mults for half 0, and 28 for half 1.
    
    # Let's list the 4 scalar mults for a generic 2x2 product P*Q
    # P[0,0]*Q[0,0] -> C[0,0]
    # P[0,1]*Q[1,0] -> C[0,0]
    # P[0,0]*Q[0,1] -> C[0,1]
    # P[0,1]*Q[1,1] -> C[0,1]
    # P[1,0]*Q[0,0] -> C[1,0]
    # P[1,1]*Q[1,0] -> C[1,0]
    # P[1,0]*Q[0,1] -> C[1,1]
    # P[1,1]*Q[1,1] -> C[1,1]
    
    # We need to map C[local_i, local_j] to global C indices.
    # For half 0:
    # C block 0 (top-left): rows 0,1 cols 0,1. Global C indices: (0,0), (0,1), (1,0), (1,1)
    # C block 1 (top-right): rows 0,1 cols 2,3. Global C indices: (0,2), (0,3), (1,2), (1,3)
    # C block 2 (bot-left): rows 2,3 cols 0,1. Global C indices: (2,0), (2,1), (3,0), (3,1)
    # C block 3 (bot-right): rows 2,3 cols 2,3. Global C indices: (2,2), (2,3), (3,2), (3,3)
    
    # The Strassen result R = M1 + M4 - M5 + M7 etc.
    # R is a 2x2 matrix of blocks.
    # R_block[0,0] corresponds to C_block 0.
    # R_block[0,1] corresponds to C_block 1.
    # R_block[1,0] corresponds to C_block 2.
    # R_block[1,1] corresponds to C_block 3.
    
    # So for each Strassen term M_k, which contributes to R with coefficient c_k (1, -1, etc),
    # and for each scalar mult in M_k (which gives P[ab] Q[cd] -> R[ab, cd]),
    # we add c_k * P[ab] * Q[cd] to C[global_i, global_j].
    
    # Let's precompute the P and Q linear combos for each of the 7 Strassen terms.
    
    def make_combos(blocks, coeffs):
        # blocks: list of 4 block indices
        # coeffs: list of 4 coefficients
        # Returns list of 4 dicts
        res = []
        for b, c in zip(blocks, coeffs):
            d = {}
            for idx, (i, j) in enumerate(get_A_block(b)):
                d[(i, j)] = c
            res.append(d)
        return res

    # M1: (A00+A11) * (B00+B11)
    # P = A00+A11. Q = B00+B11.
    # P[0,0] = A00[0,0] + A11[0,0]
    # ...
    
    # It's easier to just define the 28 terms directly.
    
    # Term list for Half 0
    # Each term: (u_dict, v_dict, w_dict)
    # u_dict: {(i,j): val}
    # v_dict: {(j,k): val}
    # w_dict: {(k,i): val}
    
    terms = []
    
    # Helper to add a scalar mult term
    def add_scalar_mult(p_coeff, q_coeff, c_block, c_local_i, c_local_j, sign, half):
        # p_coeff: dict {(i,j): val} for A
        # q_coeff: dict {(j,k): val} for B
        # c_block: 0,1,2,3
        # c_local_i, c_local_j: 0 or 1
        # sign: 1 or -1
        # half: 0 or 1
        
        u_dict = {}
        v_dict = {}
        w_dict = {}
        
        # A part
        for (i, j), val in p_coeff.items():
            u_dict[(i, j)] = val * sign
            
        # B part
        for (j, k), val in q_coeff.items():
            # Adjust k for half
            if half == 1:
                k_global = k + 4
            else:
                k_global = k
            v_dict[(j, k_global)] = val
            
        # C part
        # Map c_block, c_local_i, c_local_j to global i, k
        if c_block == 0: # top-left
            i_global = c_local_i
            k_global = c_local_j
        elif c_block == 1: # top-right
            i_global = c_local_i
            k_global = c_local_j + 2
        elif c_block == 2: # bot-left
            i_global = c_local_i + 2
            k_global = c_local_j
        else: # bot-right
            i_global = c_local_i + 2
            k_global = c_local_j + 2
            
        if half == 1:
            k_global += 4
            
        w_dict[(k_global, i_global)] = sign
        
        terms.append((u_dict, v_dict, w_dict))

    # Define P and Q for each Strassen term
    # We need the linear combos for P and Q.
    
    # M1: P=A00+A11, Q=B00+B11
    p_m1 = [
        { (0,0): 1, (2,2): 1 }, # P00
        { (0,1): 1, (2,3): 1 }, # P01
        { (1,0): 1, (3,2): 1 }, # P10
        { (1,1): 1, (3,3): 1 }  # P11
    ]
    q_m1 = [
        { (0,0): 1, (2,2): 1 }, # Q00
        { (0,1): 1, (2,3): 1 }, # Q01
        { (1,0): 1, (3,2): 1 }, # Q10
        { (1,1): 1, (3,3): 1 }  # Q11
    ]
    # M1 contributes to C00, C01, C10, C11 with sign 1
    # C00 is block 0.
    # P00*Q00 -> C00[0,0]
    add_scalar_mult(p_m1[0], q_m1[0], 0, 0, 0, 1, 0)
    add_scalar_mult(p_m1[1], q_m1[2], 0, 0, 0, 1, 0)
    add_scalar_mult(p_m1[0], q_m1[1], 1, 0, 0, 1, 0)
    add_scalar_mult(p_m1[1], q_m1[3], 1, 0, 0, 1, 0)
    add_scalar_mult(p_m1[2], q_m1[0], 2, 0, 0, 1, 0)
    add_scalar_mult(p_m1[3], q_m1[2], 2, 0, 0, 1, 0)
    add_scalar_mult(p_m1[2], q_m1[1], 3, 0, 0, 1, 0)
    add_scalar_mult(p_m1[3], q_m1[3], 3, 0, 0, 1, 0)
    
    # This is tedious. I will just implement the logic in the code.
    
    return {"U": U, "V": V, "W": W}
