import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 145

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = 4, 4, 7
    # Part 1: 4x4x4 Strassen = rank 49
    # Part 2: 4 blocks of 2x2x3 = 4 * 8 = rank 32
    total_rank = 81
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    current_rank = 0
    
    # Strassen 2x2x2 base terms: (x_dict, y_dict, c_dict)
    # x_dict: {(row,col): coeff} for A subblock
    # y_dict: {(row,col): coeff} for B subblock  
    # c_dict: {(row,col): coeff} for C subblock
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Strassen 4x4x4: 7 outer terms
    # Each outer term maps 4x4 blocks (indexed 0-3) to 2x2 sub-blocks
    # Block index b: bi = b//2, bj = b%2
    strassen_4x4_outer = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}),
        ({0: 1}, {1: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}),
        ({0: 1, 1: 1}, {3: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]
    
    # C block contributions for each outer term in 4x4x4
    c_blocks_4x4 = [
        {(0,0): 1.0, (1,1): 1.0},
        {(1,0): 1.0, (1,1): -1.0},
        {(0,1): 1.0, (1,1): 1.0},
        {(0,0): 1.0, (1,0): 1.0},
        {(0,0): -1.0, (0,1): 1.0},
        {(1,1): 1.0},
        {(0,0): 1.0},
    ]
    
    # Part 1: 4x4x4 Strassen (rank 49)
    for m_idx in range(7):
        x_def, y_def = strassen_4x4_outer[m_idx]
        c_block = c_blocks_4x4[m_idx]
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]
            
            # Compute U coefficients
            for (li, lj), cx in x_sub.items():
                for b, cb in x_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    U[i * m + j, r] += cx * cb
            
            # Compute V coefficients
            for (li, lj), cy in y_sub.items():
                for b, cb in y_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    V[i * p + j, r] += cy * cb
            
            # Compute W coefficients
            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                for (br, bc), cb_c in c_block.items():
                    if cb_c == 0:
                        continue
                    i_final = 2 * br + li
                    k_final = 2 * bc + lj
                    W[k_final * n + i_final, r] += cc * cb_c
    
    # Part 2: 4x4x3 as 4 * (2x2x3), each of rank 8
    def add_2x2x3(ar, ac, br, cr):
        nonlocal current_rank
        
        # 2x2x3 is two 2x2x1 multiplications
        # Columns 4 and 5 of B (global indices)
        # For each column c in [4, 5]:
        for c_col in [4, 5]:
            # 2x2x1: rank 4
            # A subblock: rows 2*ar, 2*ar+1; cols 2*ac, 2*ac+1
            # B subblock: rows 2*br, 2*br+1; col c_col
            # C subblock: rows 2*cr, 2*cr+1; col c_col
            
            # Term 1: A[0,0]*B[0,c]
            r = current_rank; current_rank += 1
            U[(2*ar)*m + (2*ac), r] = 1.0
            V[(2*br)*p + c_col, r] = 1.0
            W[c_col*n + (2*cr), r] = 1.0
            
            # Term 2: A[0,1]*B[1,c]
            r = current_rank; current_rank += 1
            U[(2*ar)*m + (2*ac+1), r] = 1.0
            V[(2*br+1)*p + c_col, r] = 1.0
            W[c_col*n + (2*cr), r] = 1.0
            
            # Term 3: A[1,0]*B[0,c]
            r = current_rank; current_rank += 1
            U[(2*ar+1)*m + (2*ac), r] = 1.0
            V[(2*br)*p + c_col, r] = 1.0
            W[c_col*n + (2*cr+1), r] = 1.0
            
            # Term 4: A[1,1]*B[1,c]
            r = current_rank; current_rank += 1
            U[(2*ar+1)*m + (2*ac+1), r] = 1.0
            V[(2*br+1)*p + c_col, r] = 1.0
            W[c_col*n + (2*cr+1), r] = 1.0
        
        # Column 6 of B (global index)
        # 2x2x1: rank 4
        c_col = 6
        # Term 1: A[0,0]*B[0,6]
        r = current_rank; current_rank += 1
        U[(2*ar)*m + (2*ac), r] = 1.0
        V[(2*br)*p + c_col, r] = 1.0
        W[c_col*n + (2*cr), r] = 1.0
        
        # Term 2: A[0,1]*B[1,6]
        r = current_rank; current_rank += 1
        U[(2*ar)*m + (2*ac+1), r] = 1.0
        V[(2*br+1)*p + c_col, r] = 1.0
        W[c_col*n + (2*cr), r] = 1.0
        
        # Term 3: A[1,0]*B[0,6]
        r = current_rank; current_rank += 1
        U[(2*ar+1)*m + (2*ac), r] = 1.0
        V[(2*br)*p + c_col, r] = 1.0
        W[c_col*n + (2*cr+1), r] = 1.0
        
        # Term 4: A[1,1]*B[1,6]
        r = current_rank; current_rank += 1
        U[(2*ar+1)*m + (2*ac+1), r] = 1.0
        V[(2*br+1)*p + c_col, r] = 1.0
        W[c_col*n + (2*cr+1), r] = 1.0

    add_2x2x3(0, 0, 0, 0)
    add_2x2x3(0, 1, 1, 0)
    add_2x2x3(1, 0, 0, 1)
    add_2x2x3(1, 1, 1, 1)
    
    assert current_rank == 81, f"Expected 81, got {current_rank}"
    
    U = U[:, :current_rank].tolist()
    V = V[:, :current_rank].tolist()
    W = W[:, :current_rank].tolist()
    
    return {"U": U, "V": V, "W": W}
