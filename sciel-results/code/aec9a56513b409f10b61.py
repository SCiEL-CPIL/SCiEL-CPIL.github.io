import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    block_size = 2
    blocks_per_dim = 3
    
    rank = (blocks_per_dim ** 3) * 7
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Strassen 7-term patterns for 2x2 matrix multiplication
    # Each pattern: (U_coeffs, V_coeffs, W_coeffs)
    # Coeffs are dicts mapping (row, col) -> value
    strassen_patterns = [
        # M1 = (A11 + A22)(B11 + B22) -> C11 += M1, C22 += M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A21 + A22)B11 -> C21 += M2, C22 -= M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A11(B21 - B22) -> C12 += M3, C22 += M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = A22(B11 - B12) -> C11 += M4, C21 += M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A11 + A12)B22 -> C11 -= M5, C12 += M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (A21 - A11)(B11 + B12) -> C21 += M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A12 - A22)(B21 + B22) -> C12 += M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    # Fill U: rows are (I_block_row, J_block_col) -> global row = I*2 + i, col = J*2 + j
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c, col_idx] += val
                    
                    # Fill V: rows are (J_block_row, K_block_col) -> global row = J*2 + i, col = K*2 + j
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c, col_idx] += val
                    
                    # Fill W: rows are (K_block_row, I_block_col) -> global row = K*2 + i, col = I*2 + j
                    # W[k*n+i] where k is from B's row (K block), i is from C's col (I block)
                    for (pi, pj), val in w_coeffs.items():
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i, col_idx] += val
                    
                    col_idx += 1
                
                # Check deadline
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Convert to lists for the expected interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
