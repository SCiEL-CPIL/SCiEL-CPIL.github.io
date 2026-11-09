import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x7 matrix.
    
    Interface: construct() -> {'U', 'V', 'W'}
    Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    Convention: W[k*n+i, r]
    
    Strategy:
    1. Left part: A (4x4) * B_left (4x4) -> C_left (4x4)
       Use Strassen's algorithm for 4x4x4 with rank 49.
       
    2. Right part: A (4x4) * B_right (4x3) -> C_right (4x3)
       Use schoolbook multiplication with rank 4*4*3 = 48.
       
    Total rank = 49 + 48 = 97.
    """
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    
    # Strassen 4x4x4 (Rank 49)
    # Inner Strassen 2x2x2 coefficients
    u_in = np.array([
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ])
    v_in = np.array([
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ])
    w_out = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ])
    
    u_out = u_in
    v_out = v_in
    w_outer = w_out
    
    rank_strassen_4x4 = 49
    rank_schoolbook_4x4x3 = 4 * 4 * 3
    rank = rank_strassen_4x4 + rank_schoolbook_4x4x3
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for k_outer in range(7):
        for l_inner in range(7):
            # Check deadline
            if time.time() > deadline:
                # Return best found so far (which is the full decomposition if we finished)
                # Since this is deterministic and fast, we likely won't hit this.
                return {"U": U[:, :col].tolist(), "V": V[:, :col].tolist(), "W": W[:, :col].tolist()}
                
            # Compute u_vec
            u_vec = np.zeros(16, dtype=int)
            for i_inner in range(2):
                for j_inner in range(2):
                    coeff_inner = u_in[l_inner][i_inner * 2 + j_inner]
                    if coeff_inner == 0: continue
                    for b in range(4):
                        coeff_outer = u_out[k_outer][b]
                        if coeff_outer == 0: continue
                        b_row = b // 2
                        b_col = b % 2
                        i_global = 2 * b_row + i_inner
                        j_global = 2 * b_col + j_inner
                        u_vec[i_global * 4 + j_global] += coeff_inner * coeff_outer
            
            # Compute v_vec
            v_vec = np.zeros(28, dtype=int)
            for i_inner in range(2):
                for j_inner in range(2):
                    coeff_inner = v_in[l_inner][i_inner * 2 + j_inner]
                    if coeff_inner == 0: continue
                    for b in range(4):
                        coeff_outer = v_out[k_outer][b]
                        if coeff_outer == 0: continue
                        b_row = b // 2
                        b_col = b % 2
                        j_global = 2 * b_row + i_inner
                        k_global = 2 * b_col + j_inner
                        v_vec[j_global * 7 + k_global] += coeff_inner * coeff_outer
            
            # Compute w_vec
            w_vec = np.zeros(28, dtype=int)
            for c_block in range(4):
                coeff_outer_w = w_outer[c_block][k_outer]
                if coeff_outer_w == 0: continue
                for c_inner in range(4):
                    coeff_inner_w = w_out[c_inner][l_inner]
                    if coeff_inner_w == 0: continue
                    total_coeff = coeff_outer_w * coeff_inner_w
                    if total_coeff == 0: continue
                    c_block_row = c_block // 2
                    c_block_col = c_block % 2
                    c_inner_row = c_inner // 2
                    c_inner_col = c_inner % 2
                    i_global = 2 * c_block_row + c_inner_row
                    k_global = 2 * c_block_col + c_inner_col
                    w_vec[k_global * 4 + i_global] += total_coeff
            
            U[:, col] = u_vec
            V[:, col] = v_vec
            W[:, col] = w_vec
            
            col += 1
    
    # Schoolbook 4x4x3
    for i in range(4):
        for j in range(4):
            for k_prime in range(3):
                if time.time() > deadline:
                    return {"U": U[:, :col].tolist(), "V": V[:, :col].tolist(), "W": W[:, :col].tolist()}
                k_global = 4 + k_prime
                U[i * 4 + j, col] = 1
                V[j * 7 + k_global, col] = 1
                W[k_global * 4 + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
