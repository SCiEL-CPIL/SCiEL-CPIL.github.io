import numpy as np

def _get_base_strassen():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U, V, W are lists of 7 vectors.
    U[t] is a 4-vector (2x2 flattened)
    V[t] is a 4-vector (2x2 flattened)
    W[t] is a 4-vector (2x2 flattened)
    """
    # Term 1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
    # A: 1,0,0,1 (idx 0,3)
    # B: 1,0,0,1 (idx 0,3)
    # C: 1,0,0,1 (idx 0,3)
    t1_u = [1, 0, 0, 1]
    t1_v = [1, 0, 0, 1]
    t1_w = [1, 0, 0, 1]
    
    # Term 2: (A10+A11)B00 -> C10+=1, C11-=1
    # A: 0,0,1,1 (idx 2,3)
    # B: 1,0,0,0 (idx 0)
    # C: 0,1,0,-1 (idx 1,3)
    t2_u = [0, 0, 1, 1]
    t2_v = [1, 0, 0, 0]
    t2_w = [0, 1, 0, -1]
    
    # Term 3: A00(B01-B11) -> C01+=1, C11+=1
    # A: 1,0,0,0 (idx 0)
    # B: 0,1,-1,0 (idx 1,2)
    # C: 0,0,1,1 (idx 2,3)
    t3_u = [1, 0, 0, 0]
    t3_v = [0, 1, -1, 0]
    t3_w = [0, 0, 1, 1]
    
    # Term 4: A11(B10-B00) -> C00+=1, C10+=1
    # A: 0,0,0,1 (idx 3)
    # B: -1,0,1,0 (idx 0,2)
    # C: 1,1,0,0 (idx 0,1)
    t4_u = [0, 0, 0, 1]
    t4_v = [-1, 0, 1, 0]
    t4_w = [1, 1, 0, 0]
    
    # Term 5: (A00+A01)B11 -> C00-=1, C01+=1
    # A: 1,1,0,0 (idx 0,1)
    # B: 0,0,0,1 (idx 3)
    # C: -1,0,1,0 (idx 0,2)
    t5_u = [1, 1, 0, 0]
    t5_v = [0, 0, 0, 1]
    t5_w = [-1, 0, 1, 0]
    
    # Term 6: (A10-A00)(B00+B01) -> C11+=1
    # A: -1,0,1,0 (idx 0,2)
    # B: 1,1,0,0 (idx 0,1)
    # C: 0,0,0,1 (idx 3)
    t6_u = [-1, 0, 1, 0]
    t6_v = [1, 1, 0, 0]
    t6_w = [0, 0, 0, 1]
    
    # Term 7: (A01-A11)(B10+B11) -> C00+=1
    # A: 0,1,-1,0 (idx 1,2)
    # B: 0,0,1,1 (idx 2,3)
    # C: 1,0,0,0 (idx 0)
    t7_u = [0, 1, -1, 0]
    t7_v = [0, 0, 1, 1]
    t7_w = [1, 0, 0, 0]
    
    Us = [t1_u, t2_u, t3_u, t4_u, t5_u, t6_u, t7_u]
    Vs = [t1_v, t2_v, t3_v, t4_v, t5_v, t6_v, t7_v]
    Ws = [t1_w, t2_w, t3_w, t4_w, t5_w, t6_w, t7_w]
    
    return Us, Vs, Ws

def construct():
    n, m, p = 4, 4, 8
    rank = 98
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    base_U, base_V, base_W = _get_base_strassen()
    
    # Outer Strassen terms
    # Each term c (0-6) defines:
    # A_coeffs: dict {(block_row, block_col): coeff}
    # B_coeffs: dict {(block_row, block_col): coeff}
    # C_coeffs: dict {(block_row, block_col): coeff} for the output C blocks
    
    # A blocks: 2x2 blocks of A (4x4)
    # B blocks: 2x4 blocks of B (4x8)
    # C blocks: 2x4 blocks of C (4x8)
    
    # Term 0: (A00+A11)(B00+B11)
    # A: 00, 11
    # B: 00, 11
    # C: 00 (coeff 1), 11 (coeff 1)
    # Note: C blocks are 2x4.
    # C00 is top-left 2x4. C11 is bottom-right 2x4.
    
    terms = [
        {
            'A': {(0,0): 1, (1,1): 1},
            'B': {(0,0): 1, (1,1): 1},
            'C': {(0,0): 1, (1,1): 1}
        },
        {
            'A': {(1,0): 1, (1,1): 1},
            'B': {(0,0): 1},
            'C': {(1,0): 1, (1,1): -1}
        },
        {
            'A': {(0,0): 1},
            'B': {(0,1): 1, (1,1): -1},
            'C': {(0,1): 1, (1,1): 1}
        },
        {
            'A': {(1,1): 1},
            'B': {(1,0): 1, (0,0): -1},
            'C': {(0,0): 1, (1,0): 1}
        },
        {
            'A': {(0,0): 1, (0,1): 1},
            'B': {(1,1): 1},
            'C': {(0,0): -1, (0,1): 1}
        },
        {
            'A': {(1,0): 1, (0,0): -1},
            'B': {(0,0): 1, (0,1): 1},
            'C': {(1,1): 1}
        },
        {
            'A': {(0,1): 1, (1,1): -1},
            'B': {(1,0): 1, (1,1): 1},
            'C': {(0,0): 1}
        }
    ]
    
    col = 0
    for c_idx, term in enumerate(terms):
        # For this outer term, we have 14 inner terms (7 for Left, 7 for Right)
        for side in ['L', 'R']:
            for t_idx in range(7):
                # Get base vectors
                u_base = base_U[t_idx]
                v_base = base_V[t_idx]
                w_base = base_W[t_idx]
                
                # Expand U
                # u_base is 4-vector for 2x2 A block.
                # Indices in u_base: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                # Global A indices: i in 0..3, j in 0..3
                # Block (br, bc) of A is at rows 2br..2br+1, cols 2bc..2bc+1
                for br, bc in term['A']:
                    coeff_A = term['A'][(br, bc)]
                    for r in range(2):
                        for cc in range(2):
                            # local index in base U
                            local_idx = r * 2 + cc
                            base_coeff = u_base[local_idx]
                            if base_coeff != 0:
                                global_i = 2 * br + r
                                global_j = 2 * bc + cc
                                U[global_i * m + global_j, col] += coeff_A * base_coeff
                
                # Expand V
                # v_base is 4-vector for 2x2 B block (Left or Right part).
                # B is 4x8.
                # Block (br, bc) of B is at rows 2br..2br+1, cols 4bc..4bc+3.
                # Side 'L' uses cols 0,1 of the block. Side 'R' uses cols 2,3.
                # local_idx in v_base: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                # These correspond to local cols 0,1 of the 2x2 sub-block.
                for br, bc in term['B']:
                    coeff_B = term['B'][(br, bc)]
                    for r in range(2):
                        for cc in range(2):
                            local_idx = r * 2 + cc
                            base_coeff = v_base[local_idx]
                            if base_coeff != 0:
                                global_i = 2 * br + r
                                # col offset in global B
                                if side == 'L':
                                    global_k = 4 * bc + cc
                                else:
                                    global_k = 4 * bc + 2 + cc
                                V[global_i * p + global_k, col] += coeff_B * base_coeff
                
                # Expand W
                # w_base is 4-vector for 2x2 C block.
                # C is 4x8.
                # Block (cr, cc) of C is at rows 2cr..2cr+1, cols 4cc..4cc+3.
                # Side 'L' contributes to cols 0,1 of the C block. Side 'R' to cols 2,3.
                # local_idx in w_base: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                # These correspond to local rows 0,1 and local cols 0,1 of the 2x2 result.
                # Wait, W index is k*n + i.
                # Global C indices: i (row) in 0..3, k (col) in 0..7.
                # Block (cr, cc) of C: rows 2cr..2cr+1, cols 4cc..4cc+3.
                for cr, cc in term['C']:
                    coeff_C = term['C'][(cr, cc)]
                    for r in range(2):
                        for cc_local in range(2):
                            local_idx = r * 2 + cc_local
                            base_coeff = w_base[local_idx]
                            if base_coeff != 0:
                                global_i = 2 * cr + r
                                if side == 'L':
                                    global_k = 4 * cc + cc_local
                                else:
                                    global_k = 4 * cc + 2 + cc_local
                                # W index is k*n + i
                                W[global_k * n + global_i, col] += coeff_C * base_coeff
                
                col += 1
    
    # Validate
    if not _validate(U, V, W, n, m, p):
        raise ValueError("Decomposition is invalid")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _validate(U, V, W, n, m, p):
    """Check that the decomposition exactly reproduces the matmul tensor."""
    rank = len(U[0]) if U else 0
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if not np.isclose(val, 1.0, atol=1e-10):
                    return False
    return True
