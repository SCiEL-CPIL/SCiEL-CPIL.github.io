import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin before 150s limit
    
    n, m, p = 6, 6, 6
    
    # Decompose 6x6 matrix multiplication using:
    # 1. Strassen's algorithm at the block level (4 blocks of size 3x3)
    # 2. Schoolbook multiplication (rank 9) for each 3x3 block multiplication
    # Total rank = 7 (Strassen) * 9 (schoolbook for 3x3) = 63
    
    total_rank = 7 * 9
    
    U = np.zeros((n * m, total_rank), dtype=np.int8)
    V = np.zeros((m * p, total_rank), dtype=np.int8)
    W = np.zeros((p * n, total_rank), dtype=np.int8)
    
    # Strassen's 7 products with block coefficients
    strassen_products = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),      # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),                   # M2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),                  # M3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),                  # M4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),                   # M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),        # M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),        # M7
    ]
    
    # C output: (C_block_row, C_block_col, {M_index: coeff})
    c_output = [
        (0, 0, {0: 1, 3: 1, 4: -1, 6: 1}),   # C00 = M1 + M4 - M5 + M7
        (0, 1, {2: 1, 4: 1}),                 # C01 = M3 + M5
        (1, 0, {1: 1, 3: 1}),                 # C10 = M2 + M4
        (1, 1, {0: 1, 1: -1, 2: 1, 5: 1}),   # C11 = M1 - M2 + M3 + M6
    ]
    
    col = 0
    for m_idx, (a_coeffs, b_coeffs) in enumerate(strassen_products):
        for i_local in range(3):
            for k_local in range(3):
                for j_local in range(3):
                    # U: A[i_local, k_local] from each A block
                    for (ar, ac), a_coeff in a_coeffs.items():
                        row_a = ar * 3 + i_local
                        col_a = ac * 3 + k_local
                        U[row_a * 6 + col_a, col] += a_coeff
                    
                    # V: B[k_local, j_local] from each B block
                    for (br, bc), b_coeff in b_coeffs.items():
                        row_b = br * 3 + k_local
                        col_b = bc * 3 + j_local
                        V[row_b * 6 + col_b, col] += b_coeff
                    
                    # W: C[i_local, j_local] in each C block
                    for (cr, cc), c_coeff_dict in c_output:
                        if m_idx in c_coeff_dict:
                            c_coeff = c_coeff_dict[m_idx]
                            row_c = cr * 3 + i_local
                            col_c = cc * 3 + j_local
                            W[col_c * 6 + row_c, col] += c_coeff
                    
                    col += 1
    
    assert col == total_rank
    
    # Verify: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] should be 1 for valid (i,j,k) triples
    U_arr = U.astype(np.int64)
    V_arr = V.astype(np.int64)
    W_arr = W.astype(np.int64)
    
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 10:
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    a_idx = i * 6 + k
                    b_idx = k * 6 + j
                    c_idx = j * 6 + i
                    val = int(np.sum(U_arr[a_idx, :] * V_arr[b_idx, :] * W_arr[c_idx, :]))
                    if val != 1:
                        raise ValueError(f"Verification failed at i={i}, j={j}, k={k}: got {val}")
                    if time.time() - start_time > DEADLINE - 5:
                        break
                if time.time() - start_time > DEADLINE - 5:
                    break
            if time.time() - start_time > DEADLINE - 5:
                break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
