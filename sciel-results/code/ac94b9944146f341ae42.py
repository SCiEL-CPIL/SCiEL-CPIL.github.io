import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    # Strassen-based decomposition
    # Rank 49 for 4x4x4, doubled for 4x4x8 -> Rank 98
    
    # Inner Strassen coefficients for 2x2 matrices X, Y
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # m1 = (x00+x11)(y00+y11)
    # m2 = (x10+x11)y00
    # m3 = x00(y01-y11)
    # m4 = x11(y10-y00)
    # m5 = (x00+x10)y11
    # m6 = (x01-x11)(y10+y11)
    # m7 = (x10-x00)(y00+y01)
    
    # Reconstruction coefficients alpha[i][j][k] for Z_ij = sum_k alpha[i][j][k] m_k
    # Z00 = m1 - m4 + m5 - m7
    # Z01 = m3 + m5
    # Z10 = m2 + m4
    # Z11 = m1 - m2 + m3 + m6
    
    alpha = np.zeros((2, 2, 7))
    # k=0..6 corresponds to m1..m7
    alpha[0,0,0] = 1; alpha[0,0,3] = -1; alpha[0,0,4] = 1; alpha[0,0,6] = -1
    alpha[0,1,2] = 1; alpha[0,1,4] = 1
    alpha[1,0,1] = 1; alpha[1,0,3] = 1
    alpha[1,1,0] = 1; alpha[1,1,1] = -1; alpha[1,1,2] = 1; alpha[1,1,5] = 1
    
    # Inner Strassen L_X and L_Y coefficients
    # L_X[k] is a 2x2 matrix of coefficients for x_ij
    # L_Y[k] is a 2x2 matrix of coefficients for y_ij
    L_X = np.zeros((7, 2, 2))
    L_Y = np.zeros((7, 2, 2))
    
    # m1
    L_X[0, 0, 0] = 1; L_X[0, 1, 1] = 1
    L_Y[0, 0, 0] = 1; L_Y[0, 1, 1] = 1
    # m2
    L_X[1, 1, 0] = 1; L_X[1, 1, 1] = 1
    L_Y[1, 0, 0] = 1
    # m3
    L_X[2, 0, 0] = 1
    L_Y[2, 0, 1] = 1; L_Y[2, 1, 1] = -1
    # m4
    L_X[3, 1, 1] = 1
    L_Y[3, 1, 0] = 1; L_Y[3, 0, 0] = -1
    # m5
    L_X[4, 0, 0] = 1; L_X[4, 1, 0] = 1
    L_Y[4, 1, 1] = 1
    # m6
    L_X[5, 0, 1] = 1; L_X[5, 1, 1] = -1
    L_Y[5, 1, 0] = 1; L_Y[5, 1, 1] = 1
    # m7
    L_X[6, 1, 0] = 1; L_X[6, 0, 0] = -1
    L_Y[6, 0, 0] = 1; L_Y[6, 0, 1] = 1
    
    # Outer Strassen terms
    # Each term t has:
    # - M_A structure: 2x2 matrix of block indices (0:A00, 1:A01, 2:A10, 3:A11) or None
    # - M_B structure: 2x2 matrix of block indices
    # - Output blocks: list of (block_index, sign)
    
    # Blocks: 0:00, 1:01, 2:10, 3:11
    # M_A[t] is 2x2 array of block indices or None
    # If a block index is present, it means that block is added to the corresponding position in M_A.
    # Note: M_A is a 2x2 matrix of 2x2 blocks.
    # The "scalar" x_ij in inner Strassen corresponds to the block at (i, j) in M_A.
    
    outer_terms = []
    
    # P1 = (A00 + A11)(B00 + B11)
    # M_A: [[A00, 0], [0, A11]]
    # M_B: [[B00, 0], [0, B11]]
    # Output: C00 (+1), C11 (+1)
    outer_terms.append({
        'M_A': [[0, None], [None, 3]],
        'M_B': [[0, None], [None, 3]],
        'out': [(0, 1), (3, 1)]
    })
    
    # P2 = (A10 + A11)B00
    # M_A: [[0, 0], [A10, A11]] -> Wait, A10 is block 2, A11 is block 3.
    # M_A: [[None, None], [2, 3]]
    # M_B: [[B00, 0], [0, 0]] -> [[0, None], [None, None]]
    # Output: C10 (+1)
    outer_terms.append({
        'M_A': [[None, None], [2, 3]],
        'M_B': [[0, None], [None, None]],
        'out': [(2, 1)]
    })
    
    # P3 = A00(B01 - B11)
    # M_A: [[A00, 0], [0, 0]] -> [[0, None], [None, None]]
    # M_B: [[0, B01], [0, -B11]] -> [[None, 1], [None, 3]] with signs?
    # M_B entries are linear combos.
    # Let's store M_A and M_B as 2x2 arrays of (block_index, coeff) or None.
    
    # Let's redefine outer_terms with explicit coefficients.
    # M_A[i][j] = (block_idx, coeff)
    
    outer_terms = []
    
    # P1
    outer_terms.append({
        'M_A': [[(0, 1), None], [None, (3, 1)]],
        'M_B': [[(0, 1), None], [None, (3, 1)]],
        'out': [(0, 1), (3, 1)]
    })
    
    # P2
    outer_terms.append({
        'M_A': [[None, None], [(2, 1), (3, 1)]],
        'M_B': [[(0, 1), None], [None, None]],
        'out': [(2, 1)]
    })
    
    # P3
    outer_terms.append({
        'M_A': [[(0, 1), None], [None, None]],
        'M_B': [[None, (1, 1)], [None, (3, -1)]],
        'out': [(1, 1)]
    })
    
    # P4
    outer_terms.append({
        'M_A': [[None, None], [None, (3, 1)]],
        'M_B': [[(2, 1), None], [(0, -1), None]],
        'out': [(2, 1), (0, -1)] # C10 + P4, C00 - P4
    })
    
    # P5
    outer_terms.append({
        'M_A': [[(0, 1), None], [(2, 1), None]],
        'M_B': [[None, None], [None, (3, 1)]],
        'out': [(0, 1), (1, 1)] # C00 + P5, C01 + P5
    })
    
    # P6
    outer_terms.append({
        'M_A': [[None, (1, 1)], [None, (3, -1)]],
        'M_B': [[None, None], [(2, 1), (3, 1)]],
        'out': [(3, 1)] # C11 + P6
    })
    
    # P7
    outer_terms.append({
        'M_A': [[(0, -1), None], [(2, 1), None]],
        'M_B': [[(0, 1), (1, 1)], [None, None]],
        'out': [(0, -1)] # C00 - P7
    })
    
    U_list = []
    V_list = []
    W_list = []
    
    for t in range(7):
        term = outer_terms[t]
        for k in range(7):
            U_vec = np.zeros(16)
            V_vec = np.zeros(16)
            W_vec = np.zeros(16)
            
            # Calculate U_vec and V_vec
            # x_ij in L_X[k] corresponds to M_A[i][j]
            # M_A[i][j] is (block_idx, coeff)
            # Block idx maps to 4x4 indices:
            # 0: 0,1,2,3
            # 1: 4,5,6,7
            # 2: 8,9,10,11
            # 3: 12,13,14,15
            
            block_map = {
                0: [0, 1, 4, 5],
                1: [2, 3, 6, 7],
                2: [8, 9, 12, 13],
                3: [10, 11, 14, 15]
            }
            
            # Iterate over 2x2 structure of M_A
            for i in range(2):
                for j in range(2):
                    coeff_x = L_X[k, i, j]
                    if coeff_x != 0:
                        ma_entry = term['M_A'][i][j]
                        if ma_entry is not None:
                            block_idx, coeff_a = ma_entry
                            # Add to U_vec
                            indices = block_map[block_idx]
                            for idx in indices:
                                U_vec[idx] += coeff_x * coeff_a
            
            # Iterate over 2x2 structure of M_B
            for i in range(2):
                for j in range(2):
                    coeff_y = L_Y[k, i, j]
                    if coeff_y != 0:
                        mb_entry = term['M_B'][i][j]
                        if mb_entry is not None:
                            block_idx, coeff_b = mb_entry
                            # Add to V_vec
                            indices = block_map[block_idx]
                            for idx in indices:
                                V_vec[idx] += coeff_y * coeff_b
            
            # Calculate W_vec
            # For each output block in term['out']
            for out_block_idx, sign in term['out']:
                # out_block_idx: 0:C00, 1:C01, 2:C10, 3:C11
                # Map to 4x4 indices
                w_block_map = {
                    0: [0, 1, 4, 5],
                    1: [2, 3, 6, 7],
                    2: [8, 9, 12, 13],
                    3: [10, 11, 14, 15]
                }
                indices = w_block_map[out_block_idx]
                # The inner Strassen result Z is a 2x2 matrix
                # Z_ij is added to the block at position (i, j)
                for i in range(2):
                    for j in range(2):
                        coeff_w = alpha[i, j, k]
                        if coeff_w != 0:
                            global_idx = indices[i * 2 + j]
                            W_vec[global_idx] += sign * coeff_w
            
            U_list.append(U_vec)
            V_list.append(V_vec)
            W_list.append(W_vec)
            
            # Second half of B
            V_vec_2 = np.zeros(32)
            V_vec_2[16:32] = V_vec
            W_vec_2 = np.zeros(32)
            # W is same, but output is shifted?
            # No, W is the output C.
            # C1 is cols 0-3, C2 is cols 4-7.
            # So W for the second half should be shifted?
            # W index is k*n + i.
            # k is col index.
            # For C1, k in 0..3.
            # For C2, k in 4..7.
            # So W_vec_2 should have non-zeros at indices corresponding to k=4..7.
            
            # Original W_vec has non-zeros at indices corresponding to k=0..3.
            # Index = k*4 + i.
            # For k=0..3, indices are 0..15.
            # For k=4..7, indices are 16..31.
            
            W_vec_2 = np.zeros(32)
            # Shift W_vec to the second half
            # The structure of W is the same, just shifted in k.
            # Index in W1: k*4 + i.
            # Index in W2: (k+4)*4 + i = k*4 + i + 16.
            W_vec_2[16:32] = W_vec
            
            U_list.append(U_vec)
            V_list.append(V_vec_2)
            W_list.append(W_vec_2)
            
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
