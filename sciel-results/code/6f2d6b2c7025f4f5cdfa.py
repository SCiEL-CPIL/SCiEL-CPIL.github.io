DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS
    # p=14, so 7 blocks of 2 columns
    num_blocks = p // 2
    rank = 7 * num_blocks
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A part (U), B part (V), and C part (W)
    # We define the 7 terms for a generic 2x2 multiplication.
    # Term t has U_coeffs, V_coeffs, W_coeffs
    # U index: 2*i + j
    # V index: 14*j + k (local k is 0 or 1 in block, global is 2*b + local_k)
    # W index: 2*k + i (local k is 0 or 1 in block, global is 2*b + local_k)
    
    # Let's define the 7 Strassen terms explicitly
    # M1: (A00 + A11) * (B00 + B11)
    # M2: (A10 + A11) * B00
    # M3: A00 * (B01 - B11)
    # M4: A11 * (B10 - B00)
    # M5: (A00 + A01) * B11
    # M6: (A10 - A00) * (B00 + B01)
    # M7: (A01 - A11) * (B10 + B11)
    
    # U vectors (size 4): indices 00, 01, 10, 11 -> 0, 1, 2, 3
    # V vectors (size 2 per block): indices 00, 01, 10, 11 -> but V is global.
    # For a specific block b, the V indices are:
    # B00 -> 14*0 + (2*b + 0) = 14 + 2*b
    # B01 -> 14*0 + (2*b + 1) = 14 + 2*b + 1
    # B10 -> 14*1 + (2*b + 0) = 14 + 14 + 2*b = 28 + 2*b
    # B11 -> 14*1 + (2*b + 1) = 28 + 2*b + 1
    
    # W vectors (size 2 per block):
    # C00 -> 2*(2*b) + 0 = 4*b
    # C01 -> 2*(2*b+1) + 0 = 4*b + 2
    # C10 -> 2*(2*b) + 1 = 4*b + 1
    # C11 -> 2*(2*b+1) + 1 = 4*b + 3
    
    # Define the 7 terms as (U_list, V_list, W_list)
    # U_list is dict {idx: val}
    # V_list is dict {idx: val}
    # W_list is dict {idx: val}
    
    terms = []
    
    # M1
    u1 = {0: 1, 3: 1}
    v1 = {0: 1, 3: 1} # Local indices 0,1,2,3 for B00,B01,B10,B11
    w1 = {0: 1, 3: 1}
    terms.append((u1, v1, w1))
    
    # M2
    u2 = {2: 1, 3: 1}
    v2 = {0: 1}
    # C10 = M2 + ... -> coeff 1
    # C11 = M1 - M2 + ... -> coeff -1
    w2 = {2: 1, 3: -1}
    terms.append((u2, v2, w2))
    
    # M3
    u3 = {0: 1}
    v3 = {1: 1, 3: -1}
    # C01 = M3 + ... -> coeff 1
    # C11 = ... + M3 + ... -> coeff 1
    w3 = {1: 1, 3: 1}
    terms.append((u3, v3, w3))
    
    # M4
    u4 = {3: 1}
    v4 = {2: 1, 0: -1}
    # C00 = M1 + M4 ... -> coeff 1
    # C10 = M2 + M4 ... -> coeff 1
    w4 = {0: 1, 2: 1}
    terms.append((u4, v4, w4))
    
    # M5
    u5 = {0: 1, 1: 1}
    v5 = {3: 1}
    # C00 = M1 ... - M5 ... -> coeff -1
    # C01 = M3 + M5 ... -> coeff 1
    w5 = {0: -1, 1: 1}
    terms.append((u5, v5, w5))
    
    # M6
    u6 = {2: 1, 0: -1}
    v6 = {0: 1, 1: 1}
    # C11 = ... + M6 -> coeff 1
    w6 = {3: 1}
    terms.append((u6, v6, w6))
    
    # M7
    u7 = {1: 1, 3: -1}
    v7 = {2: 1, 3: 1}
    # C00 = ... + M7 -> coeff 1
    w7 = {0: 1}
    terms.append((u7, v7, w7))
    
    col = 0
    for b in range(num_blocks):
        # Global offsets for V and W
        # V local index l corresponds to:
        # 0 -> B00 -> 14*0 + 2*b
        # 1 -> B01 -> 14*0 + 2*b + 1
        # 2 -> B10 -> 14*1 + 2*b
        # 3 -> B11 -> 14*1 + 2*b + 1
        
        # W local index l corresponds to:
        # 0 -> C00 -> 2*(2*b) + 0
        # 1 -> C01 -> 2*(2*b+1) + 0
        # 2 -> C10 -> 2*(2*b) + 1
        # 3 -> C11 -> 2*(2*b+1) + 1
        
        v_offsets = [2*b, 2*b+1, 14+2*b, 14+2*b+1]
        w_offsets = [4*b, 4*b+2, 4*b+1, 4*b+3]
        
        for u_dict, v_dict, w_dict in terms:
            # Fill U
            for idx, val in u_dict.items():
                U[idx][col] += val
            # Fill V
            for l_idx, val in v_dict.items():
                global_v_idx = v_offsets[l_idx]
                V[global_v_idx][col] += val
            # Fill W
            for l_idx, val in w_dict.items():
                global_w_idx = w_offsets[l_idx]
                W[global_w_idx][col] += val
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
