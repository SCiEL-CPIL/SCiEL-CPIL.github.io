import numpy as np
import time

def construct():
    n, m, p = 4, 4, 4
    t_start = time.time()
    LIMIT = 140.0
    
    # Strassen 2x2 decomposition matrices
    # U2[flat_idx, rank_idx] where flat_idx = i*2 + j
    # V2[flat_idx, rank_idx] where flat_idx = j*2 + k
    # W2[flat_idx, rank_idx] where flat_idx = k*2 + i
    
    # Standard Strassen 7-term decomposition
    # Terms:
    # 0: (A00+A11)(B00+B11)
    # 1: (A10+A11)B00
    # 2: A00(B01-B11)
    # 3: A11(B10-B00)
    # 4: (A00+A01)B11
    # 5: (A10-A00)(B00+B01)
    # 6: (A01-A11)(B10+B11)
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # i=0, j=0 (A00)
        [0, 0, 0, 0, 1, 0, 1],   # i=0, j=1 (A01)
        [0, 1, 0, 0, 0, 1, 0],   # i=1, j=0 (A10)
        [1, 1, 0, 1, 0, 0, -1]   # i=1, j=1 (A11)
    ], dtype=np.int64)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # j=0, k=0 (B00)
        [0, 0, 1, 0, 0, 1, 0],   # j=0, k=1 (B01)
        [0, 0, 0, 1, 0, 0, 1],   # j=1, k=0 (B10)
        [1, 0, -1, 0, 1, 0, 1]   # j=1, k=1 (B11)
    ], dtype=np.int64)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # k=0, i=0 (C00)
        [0, 1, 0, 1, 0, 0, 0],   # k=0, i=1 (C10)
        [0, 0, 1, 0, 1, 0, 0],   # k=1, i=0 (C01)
        [1, -1, 1, 0, 0, 1, 0]   # k=1, i=1 (C11)
    ], dtype=np.int64)
    
    # Verify 2x2 Strassen validity
    def check_2x2():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    val = np.dot(U2[i*2+j], V2[j*2+k] * W2[k*2+i])
                    if val != 1:
                        return False
        return True
        
    if not check_2x2():
        # Fallback to schoolbook if Strassen is somehow broken (shouldn't happen)
        rank_fallback = 64
        U_fallback = np.zeros((16, rank_fallback), dtype=np.int64)
        V_fallback = np.zeros((16, rank_fallback), dtype=np.int64)
        W_fallback = np.zeros((16, rank_fallback), dtype=np.int64)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_fallback[i*4+j, col] = 1
                    V_fallback[j*4+k, col] = 1
                    W_fallback[k*4+i, col] = 1
                    col += 1
        return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}

    # Construct 4x4 decomposition using Kronecker structure manually
    # Rank is 7 * 7 = 49
    rank_total = 49
    U4 = np.zeros((16, rank_total), dtype=np.int64)
    V4 = np.zeros((16, rank_total), dtype=np.int64)
    W4 = np.zeros((16, rank_total), dtype=np.int64)
    
    # Index mapping:
    # U4 row index: i * 4 + j, where i = 2*i_outer + i_inner, j = 2*j_outer + j_inner
    # So row = (2*i_o + i_i)*4 + (2*j_o + j_j) = 8*i_o + 2*i_i + 4*j_o + j_j
    # This is equivalent to the row index in np.kron(U2, U2) if we permute correctly.
    # Let's just iterate and assign.
    
    for c_outer in range(7):
        for c_inner in range(7):
            c_idx = c_outer * 7 + c_inner
            
            # U component: U2[i_o, j_o, c_outer] * U2[i_i, j_i, c_inner]
            # We need to fill U4[row, c_idx] for all i, j
            for i in range(2):
                for j in range(2):
                    # Outer indices
                    i_o, j_o = i, j
                    val_outer = U2[i_o * 2 + j_o, c_outer]
                    if val_outer == 0:
                        continue
                    
                    for i_i in range(2):
                        for j_i in range(2):
                            # Inner indices
                            val_inner = U2[i_i * 2 + j_i, c_inner]
                            if val_inner == 0:
                                continue
                            
                            val = val_outer * val_inner
                            if val == 0:
                                continue
                                
                            # Calculate full indices
                            full_i = i_o * 2 + i_i
                            full_j = j_o * 2 + j_i
                            row_u = full_i * 4 + full_j
                            
                            U4[row_u, c_idx] = val
                            
                            # V component: V2[j_o, k_o, c_outer] * V2[j_i, k_i, c_inner]
                            # Row in V4 is full_j * 4 + full_k
                            # We fill this in the same c_idx loop
                            
                            # W component: W2[k_o, i_o, c_outer] * W2[k_i, i_i, c_inner]
                            # Row in W4 is full_k * 4 + full_i

            # Fill V4
            for j in range(2):
                for k in range(2):
                    j_o, k_o = j, k
                    val_outer = V2[j_o * 2 + k_o, c_outer]
                    if val_outer == 0:
                        continue
                    for j_i in range(2):
                        for k_i in range(2):
                            val_inner = V2[j_i * 2 + k_i, c_inner]
                            if val_inner == 0:
                                continue
                            val = val_outer * val_inner
                            if val == 0:
                                continue
                            
                            full_j = j_o * 2 + j_i
                            full_k = k_o * 2 + k_i
                            row_v = full_j * 4 + full_k
                            
                            V4[row_v, c_idx] = val
                            
            # Fill W4
            for k in range(2):
                for i in range(2):
                    k_o, i_o = k, i
                    val_outer = W2[k_o * 2 + i_o, c_outer]
                    if val_outer == 0:
                        continue
                    for k_i in range(2):
                        for i_i in range(2):
                            val_inner = W2[k_i * 2 + i_i, c_inner]
                            if val_inner == 0:
                                continue
                            val = val_outer * val_inner
                            if val == 0:
                                continue
                            
                            full_k = k_o * 2 + k_i
                            full_i = i_o * 2 + i_i
                            row_w = full_k * 4 + full_i
                            
                            W4[row_w, c_idx] = val

    # Verify 4x4 Decomposition
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] should be 1
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0
                for c in range(rank_total):
                    u_val = U4[i*4+j, c]
                    v_val = V4[j*4+k, c]
                    w_val = W4[k*4+i, c]
                    if u_val != 0 and v_val != 0 and w_val != 0:
                        s += u_val * v_val * w_val
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback
        rank_fallback = 64
        U_fallback = np.zeros((16, rank_fallback), dtype=np.int64)
        V_fallback = np.zeros((16, rank_fallback), dtype=np.int64)
        W_fallback = np.zeros((16, rank_fallback), dtype=np.int64)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_fallback[i*4+j, col] = 1
                    V_fallback[j*4+k, col] = 1
                    W_fallback[k*4+i, col] = 1
                    col += 1
        return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}

    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
