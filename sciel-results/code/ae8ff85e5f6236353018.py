import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Block structure: 6x6 = 2x2 blocks of 3x3
    block_size = 3
    num_blocks = 2
    
    # Strassen's 7 multiplications for 2x2 matrix multiplication:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mi is a 3x3 matrix product, which has 27 scalar products
    # Total rank = 7 * 27 = 189
    
    total_rank = 189
    
    U = np.zeros((36, total_rank), dtype=np.int32)
    V = np.zeros((36, total_rank), dtype=np.int32)
    W = np.zeros((36, total_rank), dtype=np.int32)
    
    # Index mapping:
    # A is 6x6: A[i,j] where i,j in 0..5
    # Block form: A[ib*3 + ir, jb*3 + jr] where ib,jb in 0..1, ir,jr in 0..2
    # 
    # U has 36 rows: row = i*6 + j (input A indices)
    # V has 36 rows: row = j*6 + k (input B indices)
    # W has 36 rows: row = k*6 + i (output C indices, trace convention W[k*n+i, r])
    
    # Strassen steps: (X_block_coeffs, Y_block_coeffs, C_block_coeffs)
    # X_block_coeffs: dict from (ib, jb) -> coeff for A[ib, jb]
    # Y_block_coeffs: dict from (jb, kb) -> coeff for B[jb, kb]
    # C_block_coeffs: dict from (ib, kb) -> coeff for C[ib, kb]
    
    strassen_steps = [
        # M1 = (A00 + A11) * (B00 + B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A10 + A11) * B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A00 * (B01 - B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = A11 * (B10 - B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A00 + A01) * B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (A10 - A00) * (B00 + B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A01 - A11) * (B10 + B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col = 0
    
    for step in strassen_steps:
        X_coeffs, Y_coeffs, C_coeffs = step
        
        # For each of the 27 scalar products in the 3x3 matrix multiplication
        for r in range(block_size):  # row within block
            for s in range(block_size):  # sum index within block
                for c in range(block_size):  # col within block
                    
                    # U part: X[r, s] = sum_{ib,jb} X_coeffs[(ib,jb)] * A[ib*3+r, jb*3+s]
                    for (ib, jb), x_coeff in X_coeffs.items():
                        u_row = ib * block_size + r
                        u_col = jb * block_size + s
                        u_idx = u_row * 6 + u_col
                        U[u_idx, col] += x_coeff
                    
                    # V part: Y[s, c] = sum_{jb,kb} Y_coeffs[(jb,kb)] * B[jb*3+s, kb*3+c]
                    for (jb, kb), y_coeff in Y_coeffs.items():
                        v_row = jb * block_size + s
                        v_col = kb * block_size + c
                        v_idx = v_row * 6 + v_col
                        V[v_idx, col] += y_coeff
                    
                    # W part: M[r, c] contributes to C blocks
                    # C[ib, kb, r, c] += C_coeffs[(ib,kb)] * M[r,c]
                    # W row index (trace convention): (kb*3+c) * 6 + (ib*3+r)
                    for (ib, kb), c_coeff in C_coeffs.items():
                        w_k = kb * block_size + c
                        w_i = ib * block_size + r
                        w_idx = w_k * 6 + w_i
                        W[w_idx, col] += c_coeff
                    
                    col += 1
        
        # Check time
        if time.time() > deadline:
            break
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
