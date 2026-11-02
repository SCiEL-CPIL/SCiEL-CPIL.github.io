import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will build the U, V, W matrices by appending columns
    # for each rank term.
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a rank term
    def add_rank(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        # u_idx, v_idx, w_idx are lists of indices
        # u_val, v_val, w_val are lists of values
        # We need to pad to full size
        pass # We will do this more efficiently by creating arrays and filling them.

    # Let's use arrays and fill them
    # We don't know the total rank in advance if we want to be flexible, 
    # but we can estimate or just use a list of tuples.
    
    terms = []
    
    # Block 1: j=0, all k
    # A[:, 0] (2x1) * B[0, :] (1x7)
    # i in 0..1, j=0, k in 0..6
    for i in range(n):
        for k in range(p):
            terms.append((
                (i * m + 0, 1.0),
                (0 * p + k, 1.0),
                (k * n + i, 1.0)
            ))
            
    # Block 2: j in 1..2, k in 0..6
    # A[:, 1:3] (2x2) * B[1:3, :] (2x7)
    
    # Sub-block 2.1: k in 0..1
    # A[:, 1:3] (2x2) * B[1:3, 0:2] (2x2)
    # Strassen for 2x2x2
    # Indices:
    # A rows: i in 0..1 -> global i
    # A cols: j in 1..2 -> global j
    # B rows: j in 1..2 -> global j
    # B cols: k in 0..1 -> global k
    
    def add_strassen(i_off, j_off, k_off):
        # i_off: offset for i in A/C
        # j_off: offset for j in A/B
        # k_off: offset for k in B/C
        
        # A_11 = A[i_off, j_off]
        # A_12 = A[i_off, j_off+1]
        # A_21 = A[i_off+1, j_off]
        # A_22 = A[i_off+1, j_off+1]
        
        # B_11 = B[j_off, k_off]
        # B_12 = B[j_off, k_off+1]
        # B_21 = B[j_off+1, k_off]
        # B_22 = B[j_off+1, k_off+1]
        
        # M1 = (A_11+A_22)(B_11+B_22)
        # M2 = (A_21+A_22)B_11
        # M3 = A_11(B_12-B_22)
        # M4 = A_22(B_21-B_11)
        # M5 = (A_11+A_12)B_22
        # M6 = (A_21-A_11)(B_11+B_12)
        # M7 = (A_12-A_22)(B_21+B_22)
        
        # We need to express these as linear combinations of U, V, W.
        # Each M_r corresponds to one rank term.
        # U part: linear combo of A entries
        # V part: linear combo of B entries
        # W part: distributes to C entries
        
        # M1
        # U: A_11 + A_22
        # V: B_11 + B_22
        # W: C_11 += 1, C_12 += 0, C_21 += 0, C_22 += 1? 
        # Wait, C_11 = M1 + M4 - M5 + M7
        # C_12 = M3 + M5
        # C_21 = M2 + M4
        # C_22 = M1 - M2 + M3 + M6
        
        # So for M1:
        # W contribution: C_11 += 1, C_22 += 1
        # Indices for W:
        # C_11 -> k_off, i_off -> idx (k_off * n + i_off)
        # C_22 -> k_off+1, i_off+1 -> idx ((k_off+1) * n + (i_off+1))
        
        def add_term(u_idx_vals, v_idx_vals, w_idx_vals):
            # u_idx_vals: list of (global_idx, val)
            # v_idx_vals: list of (global_idx, val)
            # w_idx_vals: list of (global_idx, val)
            terms.append((
                u_idx_vals,
                v_idx_vals,
                w_idx_vals
            ))
            
        # M1
        add_term(
            [((i_off) * m + (j_off), 1.0), ((i_off+1) * m + (j_off+1), 1.0)],
            [((j_off) * p + (k_off), 1.0), ((j_off+1) * p + (k_off+1), 1.0)],
            [((k_off) * n + (i_off), 1.0), ((k_off+1) * n + (i_off+1), 1.0)]
        )
        
        # M2
        # U: A_21 + A_22
        # V: B_11
        # W: C_21 += 1
        add_term(
            [((i_off+1) * m + (j_off), 1.0), ((i_off+1) * m + (j_off+1), 1.0)],
            [((j_off) * p + (k_off), 1.0)],
            [((k_off) * n + (i_off+1), 1.0)]
        )
        
        # M3
        # U: A_11
        # V: B_12 - B_22
        # W: C_12 += 1, C_22 += 1
        add_term(
            [((i_off) * m + (j_off), 1.0)],
            [((j_off) * p + (k_off+1), 1.0), ((j_off+1) * p + (k_off+1), -1.0)],
            [((k_off+1) * n + (i_off), 1.0), ((k_off+1) * n + (i_off+1), 1.0)]
        )
        
        # M4
        # U: A_22
        # V: B_21 - B_11
        # W: C_11 += 1, C_21 += 1
        add_term(
            [((i_off+1) * m + (j_off+1), 1.0)],
            [((j_off+1) * p + (k_off), 1.0), ((j_off) * p + (k_off), -1.0)],
            [((k_off) * n + (i_off), 1.0), ((k_off) * n + (i_off+1), 1.0)]
        )
        
        # M5
        # U: A_11 + A_12
        # V: B_22
        # W: C_11 -= 1, C_12 += 1
        add_term(
            [((i_off) * m + (j_off), 1.0), ((i_off) * m + (j_off+1), 1.0)],
            [((j_off+1) * p + (k_off+1), 1.0)],
            [((k_off) * n + (i_off), -1.0), ((k_off+1) * n + (i_off), 1.0)]
        )
        
        # M6
        # U: A_21 - A_11
        # V: B_11 + B_12
        # W: C_22 += 1
        add_term(
            [((i_off+1) * m + (j_off), 1.0), ((i_off) * m + (j_off), -1.0)],
            [((j_off) * p + (k_off), 1.0), ((j_off) * p + (k_off+1), 1.0)],
            [((k_off+1) * n + (i_off+1), 1.0)]
        )
        
        # M7
        # U: A_12 - A_22
        # V: B_21 + B_22
        # W: C_11 += 1
        add_term(
            [((i_off) * m + (j_off+1), 1.0), ((i_off+1) * m + (j_off+1), -1.0)],
            [((j_off+1) * p + (k_off), 1.0), ((j_off+1) * p + (k_off+1), 1.0)],
            [((k_off) * n + (i_off), 1.0)]
        )

    # Call add_strassen for the 3 blocks of size 2x2
    # j_off = 1
    # k_off = 0, 2, 4
    add_strassen(0, 1, 0)
    add_strassen(0, 1, 2)
    add_strassen(0, 1, 4)
    
    # Sub-block 2.4: k in 6..6 (size 1)
    # A[:, 1:3] (2x2) * B[1:3, 6:7] (2x1)
    # Trivial rank 4
    # i in 0..1, j in 1..2, k=6
    for i in range(n):
        for j in [1, 2]:
            terms.append((
                [((i) * m + (j), 1.0)],
                [((j) * p + (6), 1.0)],
                [((6) * n + (i), 1.0)]
            ))
            
    # Now construct U, V, W
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r, term in enumerate(terms):
        u_idx_vals, v_idx_vals, w_idx_vals = term
        for idx, val in u_idx_vals:
            U[idx, r] += val
        for idx, val in v_idx_vals:
            V[idx, r] += val
        for idx, val in w_idx_vals:
            W[idx, r] += val
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
