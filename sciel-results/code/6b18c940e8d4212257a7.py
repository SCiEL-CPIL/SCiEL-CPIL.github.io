import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Global indices for A blocks (row-major flattening of 4x4)
    # A00: rows 0,1 cols 0,1 -> 0,1,4,5
    # A01: rows 0,1 cols 2,3 -> 2,3,6,7
    # A10: rows 2,3 cols 0,1 -> 8,9,12,13
    # A11: rows 2,3 cols 2,3 -> 10,11,14,15
    A_blocks = {
        'A00': [0, 1, 4, 5],
        'A01': [2, 3, 6, 7],
        'A10': [8, 9, 12, 13],
        'A11': [10, 11, 14, 15]
    }
    
    # Global indices for B blocks (row-major flattening of 4x4)
    B_blocks = {
        'B00': [0, 1, 4, 5],
        'B01': [2, 3, 6, 7],
        'B10': [8, 9, 12, 13],
        'B11': [10, 11, 14, 15]
    }
    
    # Global indices for C blocks (column-major flattening of 4x4: k*n+i)
    # C00: rows 0,1 cols 0,1
    # (0,0)->0, (0,1)->4, (1,0)->1, (1,1)->5
    # C01: rows 0,1 cols 2,3
    # (0,2)->8, (0,3)->12, (1,2)->9, (1,3)->13
    # C10: rows 2,3 cols 0,1
    # (2,0)->2, (2,1)->6, (3,0)->3, (3,1)->7
    # C11: rows 2,3 cols 2,3
    # (2,2)->10, (2,3)->14, (3,2)->11, (3,3)->15
    C_blocks = {
        'C00': [0, 4, 1, 5],
        'C01': [8, 12, 9, 13],
        'C10': [2, 6, 3, 7],
        'C11': [10, 14, 11, 15]
    }
    
    # Outer Strassen terms
    # (X_ops, Y_ops, C_contribs)
    block_terms = [
        ([('A00', 1), ('A11', 1)], [('B00', 1), ('B11', 1)], [('C00', 1), ('C11', 1)]),
        ([('A10', 1), ('A11', 1)], [('B00', 1)], [('C10', 1), ('C11', -1)]),
        ([('A00', 1)], [('B01', 1), ('B11', -1)], [('C01', 1), ('C11', 1)]),
        ([('A11', 1)], [('B10', 1), ('B00', -1)], [('C00', 1), ('C10', 1)]),
        ([('A00', 1), ('A01', 1)], [('B11', 1)], [('C00', -1), ('C01', 1)]),
        ([('A10', 1), ('A00', -1)], [('B00', 1), ('B01', 1)], [('C11', 1)]),
        ([('A01', 1), ('A11', -1)], [('B10', 1), ('B11', 1)], [('C00', 1)]),
    ]
    
    # Inner Strassen factors
    # (x_local_coeffs, y_local_coeffs, z_local_coeffs)
    # Local pos: 0=00, 1=01, 2=10, 3=11
    scalar_factors = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}), # S1
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),      # S2
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),      # S3
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),      # S4
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),      # S5
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),      # S6
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),      # S7
    ]
    
    def expand_block_form(ops, block_map):
        res = {}
        for name, coeff in ops:
            indices = block_map[name]
            for i in indices:
                res[i] = res.get(i, 0) + coeff
        return res

    term_idx = 0
    for x_ops, y_ops, c_contribs in block_terms:
        x_local_ops = {0: [], 1: [], 2: [], 3: []}
        y_local_ops = {0: [], 1: [], 2: [], 3: []}
        
        for name, coeff in x_ops:
            if name == 'A00': x_local_ops[0].append(('A00', coeff))
            elif name == 'A01': x_local_ops[1].append(('A01', coeff))
            elif name == 'A10': x_local_ops[2].append(('A10', coeff))
            elif name == 'A11': x_local_ops[3].append(('A11', coeff))
            
        for name, coeff in y_ops:
            if name == 'B00': y_local_ops[0].append(('B00', coeff))
            elif name == 'B01': y_local_ops[1].append(('B01', coeff))
            elif name == 'B10': y_local_ops[2].append(('B10', coeff))
            elif name == 'B11': y_local_ops[3].append(('B11', coeff))
            
        for x_lc, y_lc, z_lc in scalar_factors:
            u_row = {}
            for lp, coeff in x_lc.items():
                if not x_local_ops[lp]: continue
                exp = expand_block_form(x_local_ops[lp], A_blocks)
                for gi, gc in exp.items():
                    u_row[gi] = u_row.get(gi, 0) + coeff * gc
                    
            v_row = {}
            for lp, coeff in y_lc.items():
                if not y_local_ops[lp]: continue
                exp = expand_block_form(y_local_ops[lp], B_blocks)
                for gi, gc in exp.items():
                    v_row[gi] = v_row.get(gi, 0) + coeff * gc
                    
            w_row = {}
            for z_pos, z_coeff in z_lc.items():
                for c_name, c_coeff in c_contribs:
                    tc = z_coeff * c_coeff
                    if tc == 0: continue
                    w_idx = C_blocks[c_name][z_pos]
                    w_row[w_idx] = w_row.get(w_idx, 0) + tc
            
            for idx, val in u_row.items():
                U[idx, term_idx] = val
            for idx, val in v_row.items():
                V[idx, term_idx] = val
            for idx, val in w_row.items():
                W[idx, term_idx] = val
                
            term_idx += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
