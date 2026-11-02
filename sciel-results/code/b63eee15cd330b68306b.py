import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin before 150s limit
    n, m, p = 2, 4, 8
    
    # The problem is equivalent to two independent 2x2 by 2x8 multiplications.
    # Rank of 2x2x2x8 is 28. Total rank 56.
    # We construct the 56 rank-1 terms.
    
    U_list = []
    V_list = []
    W_list = []
    
    # Strassen terms for 2x2x2
    # x: 2x2 coeffs, y: 2x2 coeffs, z: 2x2 coeffs
    # We iterate over 8 blocks: 2 A-blocks (cols 0-1, 2-3) and 4 B-blocks (cols 0-1, 2-3, 4-5, 6-7 of the 2x8 part? No)
    # A is 2x4. Split into A1 (cols 0,1) and A2 (cols 2,3).
    # B is 4x8. Split into B_top (rows 0,1) and B_bot (rows 2,3).
    # B_top is 2x8. Split into 4 blocks of 2x2: B00, B01, B02, B03.
    # B_bot is 2x8. Split into 4 blocks of 2x2: B10, B11, B12, B13.
    # Terms for A1 * B00, A1 * B01, A1 * B02, A1 * B03
    # Terms for A2 * B10, A2 * B11, A2 * B12, A2 * B13
    
    strassen_terms = [
        {'x': {(0,0):1, (1,1):1}, 'y': {(0,0):1, (1,1):1}, 'z': {(0,0):1, (1,1):1}},
        {'x': {(1,0):1, (1,1):1}, 'y': {(0,0):1}, 'z': {(1,0):1, (1,1):-1}},
        {'x': {(0,0):1}, 'y': {(0,1):1, (1,1):-1}, 'z': {(0,1):1, (1,1):1}},
        {'x': {(1,1):1}, 'y': {(1,0):1, (0,0):-1}, 'z': {(0,0):1, (1,0):1}},
        {'x': {(0,0):1, (0,1):1}, 'y': {(1,1):1}, 'z': {(0,0):-1, (0,1):1}},
        {'x': {(1,0):1, (0,0):-1}, 'y': {(0,0):1, (0,1):1}, 'z': {(1,1):1}},
        {'x': {(0,1):1, (1,1):-1}, 'y': {(1,0):1, (1,1):1}, 'z': {(0,0):1}},
    ]
    
    for a_block in range(2): # 0 for A1, 1 for A2
        if time.time() > deadline:
            break
        for b_block in range(4): # 0..3 for B_top blocks, 4..7 for B_bot blocks
            if time.time() > deadline:
                break
            
            # Determine which A and B block this term belongs to
            # a_block 0 -> A cols 0,1. a_block 1 -> A cols 2,3.
            # b_block 0..3 -> B rows 0,1 (top). b_block 4..7 -> B rows 2,3 (bot).
            # Wait, the previous code logic was:
            # a in 0..1, b in 0..3.
            # u_idx = i_prime * m + (2*a + j_prime)
            # v_idx = (2*a + j_prime) * p + (2*b + k_prime)
            # w_idx = (2*b + k_prime) * n + i_prime
            
            # Let's trace:
            # A is 2x4. Indices i (0,1), j (0,1,2,3).
            # B is 4x8. Indices j (0,1,2,3), k (0..7).
            # C is 2x8. Indices i (0,1), k (0..7).
            
            # The term 'x' maps to A. (i_prime, j_prime) in 0..1.
            # u_idx = i_prime * 4 + (2*a + j_prime).
            # If a=0, j_prime in 0,1 -> cols 0,1.
            # If a=1, j_prime in 0,1 -> cols 2,3.
            # This covers all 4 cols of A.
            
            # The term 'y' maps to B. (j_prime, k_prime) in 0..1.
            # v_idx = (2*a + j_prime) * 8 + (2*b + k_prime).
            # If a=0, j_prime in 0,1 -> rows 0,1.
            # If a=1, j_prime in 0,1 -> rows 2,3.
            # If b=0, k_prime in 0,1 -> cols 0,1.
            # If b=1, k_prime in 0,1 -> cols 2,3.
            # If b=2, k_prime in 0,1 -> cols 4,5.
            # If b=3, k_prime in 0,1 -> cols 6,7.
            # This covers all 4 rows and 8 cols of B.
            # Note: The row index of B is linked to 'a'.
            # So for a=0, we use B rows 0,1.
            # For a=1, we use B rows 2,3.
            # This matches the block decomposition.
            
            # The term 'z' maps to C. (i_prime, k_prime) in 0..1.
            # w_idx = (2*b + k_prime) * 2 + i_prime.
            # If b=0, k_prime in 0,1 -> cols 0,1.
            # ...
            # This covers all 8 cols of C.
            
            for t in range(7):
                term = strassen_terms[t]
                u_vec = np.zeros(n*m)
                v_vec = np.zeros(m*p)
                w_vec = np.zeros(p*n)
                
                for (i_prime, j_prime), coeff in term['x'].items():
                    u_idx = i_prime * m + (2*a_block + j_prime)
                    u_vec[u_idx] += coeff
                
                for (j_prime, k_prime), coeff in term['y'].items():
                    v_idx = (2*a_block + j_prime) * p + (2*b_block + k_prime)
                    v_vec[v_idx] += coeff
                
                for (i_prime, k_prime), coeff in term['z'].items():
                    w_idx = (2*b_block + k_prime) * n + i_prime
                    w_vec[w_idx] += coeff
                
                U_list.append(u_vec)
                V_list.append(v_vec)
                W_list.append(w_vec)
    
    if not U_list:
        # Fallback if time ran out immediately (unlikely)
        return {"U": [], "V": [], "W": []}

    U_arr = np.array(U_list).T
    V_arr = np.array(V_list).T
    W_arr = np.array(W_list).T
    
    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
