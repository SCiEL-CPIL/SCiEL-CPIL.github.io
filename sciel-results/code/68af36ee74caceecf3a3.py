import time
import sys

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Base Strassen tensors for 2x2x2
    # u, v, w are length 4 vectors (flattened 2x2)
    base_tensors = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]), # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, -1]), # M3
        ([0, 0, 0, 1], [1, 0, -1, 0], [-1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, -1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 0, 0, 0]), # M7
    ]
    
    # Macro Strassen steps for 4x4x4
    # A_macro, B_macro, C_macro are length 4 vectors (coeffs for blocks 00, 01, 10, 11)
    macro_steps = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]), # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, -1]), # M3
        ([0, 0, 0, 1], [1, 0, -1, 0], [-1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, -1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 0, 0, 0]), # M7
    ]
    
    U_list = []
    V_list = []
    W_list = []
    
    # Block offsets for 4x4 matrix (16 elements)
    # Block 00: 0-3, 4-7? No, row-major.
    # A is 4x4.
    # Block 00 (rows 0-1, cols 0-1): indices 0,1,4,5
    # Block 01 (rows 0-1, cols 2-3): indices 2,3,6,7
    # Block 10 (rows 2-3, cols 0-1): indices 8,9,12,13
    # Block 11 (rows 2-3, cols 2-3): indices 10,11,14,15
    
    def get_block_indices(block_id):
        # block_id: 0=00, 1=01, 2=10, 3=11
        r = block_id // 2
        c = block_id % 2
        start_r = r * 2
        start_c = c * 2
        indices = []
        for i in range(2):
            for j in range(2):
                indices.append((start_r + i) * 4 + (start_c + j))
        return indices
    
    # B is 4x8. Split into two 4x4 halves.
    # Half 0: cols 0-3. Same indices as A.
    # Half 1: cols 4-7. Indices offset by 16.
    
    # C is 4x8. Split into two 4x4 halves.
    # Half 0: cols 0-3.
    # Half 1: cols 4-7.
    
    for half in range(2):
        offset_b = half * 16
        offset_c = half * 16
        
        for macro_idx in range(7):
            a_macro, b_macro, c_macro = macro_steps[macro_idx]
            
            for base_idx in range(7):
                u_base, v_base, w_base = base_tensors[base_idx]
                
                U = [0.0] * 16
                V = [0.0] * 32
                W = [0.0] * 32
                
                # Fill U
                for b_idx in range(4):
                    if a_macro[b_idx] != 0:
                        indices = get_block_indices(b_idx)
                        for k in range(4):
                            U[indices[k]] = u_base[k] * a_macro[b_idx]
                
                # Fill V
                for b_idx in range(4):
                    if b_macro[b_idx] != 0:
                        indices = get_block_indices(b_idx)
                        for k in range(4):
                            V[indices[k] + offset_b] = v_base[k] * b_macro[b_idx]
                
                # Fill W
                for b_idx in range(4):
                    if c_macro[b_idx] != 0:
                        indices = get_block_indices(b_idx)
                        for k in range(4):
                            W[indices[k] + offset_c] = w_base[k] * c_macro[b_idx]
                
                U_list.append(U)
                V_list.append(V)
                W_list.append(W)
    
    rank = len(U_list)
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
