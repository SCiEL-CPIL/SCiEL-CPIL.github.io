import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # p is 16, so there are 8 blocks of size 2
    num_blocks = p // 2
    block_size = 2
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Define Strassen decomposition for 2x2x2
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> Note: W index is k*n + i.
    # For 2x2x2, n=2.
    # k=0, i=0 -> 0
    # k=1, i=0 -> 2
    # k=0, i=1 -> 1
    # k=1, i=1 -> 3
    # So W local indices:
    # C_00 -> 0
    # C_10 -> 2
    # C_01 -> 1
    # C_11 -> 3
    
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B01
    # M3 = A00B10
    # M4 = A11B11
    # M5 = A00(B11-B01)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M5
    # C01 = M2 + M4
    # C10 = M3 + M6
    # C11 = M1 - M2 + M3
    
    # We store U, V, W contributions for each term
    # Term: (U_indices, U_vals, V_indices, V_vals, W_indices, W_vals)
    
    terms = []
    
    # Term 1: M1
    u_idx = [0, 3]
    u_val = [1, 1]
    v_idx = [0, 3]
    v_val = [1, 1]
    w_idx = [0, 3] # C00, C11
    w_val = [1, 1]
    terms.append((u_idx, u_val, v_idx, v_val, w_idx, w_val))
    
    # Term 2: M2
    u_idx = [2, 3]
    u_val = [1, 1]
    v_idx = [1]
    v_val = [1]
    w_idx = [1, 3] # C01, C11
    w_val = [1, -1]
    terms.append((u_idx, u_val, v_idx, v_val, w_idx, w_val))
    
    # Term 3: M3
    u_idx = [0]
    u_val = [1]
    v_idx = [2]
    v_val = [1]
    w_idx = [2, 3] # C10, C11
    w_val = [1, 1]
    terms.append((u_idx, u_val, v_idx, v_val, w_idx, w_val))
    
    # Term 4: M4
    u_idx = [3]
    u_val = [1]
    v_idx = [3]
    v_val = [1]
    w_idx = [0, 1] # C00, C01
    w_val = [-1, 1]
    terms.append((u_idx, u_val, v_idx, v_val, w_idx, w_val))
    
    # Term 5: M5
    u_idx = [0]
    u_val = [1]
    v_idx = [3, 1]
    v_val = [1, -1]
    w_idx = [0] # C00
    w_val = [1]
    terms.append((u_idx, u_val, v_idx, v_val, w_idx, w_val))
    
    # Term 6: M6
    u_idx = [2, 0]
    u_val = [1, -1]
    v_idx = [0, 1]
    v_val = [1, 1]
    w_idx = [2] # C10
    w_val = [1]
    terms.append((u_idx, u_val, v_idx, v_val, w_idx, w_val))
    
    # Term 7: M7
    u_idx = [1, 3]
    u_val = [1, -1]
    v_idx = [2, 3]
    v_val = [1, 1]
    w_idx = [3] # C11
    w_val = [1]
    terms.append((u_idx, u_val, v_idx, v_val, w_idx, w_val))
    
    # Map local indices to global indices for each block
    for l in range(num_blocks):
        base_col = l * rank_per_block
        for t_idx, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms):
            col = base_col + t_idx
            
            # U mapping: local (i,j) -> global 2i+j
            # Local indices in terms are already 2i+j
            for idx, val in zip(u_idx, u_val):
                U[idx][col] = val
                
            # V mapping: local (j, s) -> global 16j + (2l+s)
            # Local V index is 2j+s
            for idx, val in zip(v_idx, v_val):
                j = idx // 2
                s = idx % 2
                global_v_idx = 16 * j + (2 * l + s)
                V[global_v_idx][col] = val
                
            # W mapping: local (s, i) -> global 2(2l+s)+i
            # Local W index is 2s+i (since n=2, index = k*n + i = s*2 + i)
            for idx, val in zip(w_idx, w_val):
                s = idx // 2
                i = idx % 2
                global_w_idx = 2 * (2 * l + s) + i
                W[global_w_idx][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
