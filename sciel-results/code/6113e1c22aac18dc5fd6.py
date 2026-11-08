import numpy as np
import time

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 Strassen decomposition
    # We will build U, V, W of shape (16, 49)
    
    # Base 2x2 Strassen coefficients
    # Inputs A: indices 0,1,2,3 (A00, A01, A10, A11)
    # Inputs B: indices 0,1,2,3 (B00, B01, B10, B11)
    # Outputs C: indices 0,1,2,3 (C00, C01, C10, C11)
    
    # U_base[4, 7], V_base[4, 7], W_base[4, 7]
    # Each column is a rank-1 term
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    U_base = np.zeros((4, 7))
    V_base = np.zeros((4, 7))
    W_base = np.zeros((4, 7))
    
    # Term 1
    U_base[0, 0] = 1; U_base[3, 0] = 1
    V_base[0, 0] = 1; V_base[3, 0] = 1
    # Term 2
    U_base[2, 1] = 1; U_base[3, 1] = 1
    V_base[0, 1] = 1
    # Term 3
    U_base[0, 2] = 1
    V_base[1, 2] = 1; V_base[3, 2] = -1
    # Term 4
    U_base[3, 3] = 1
    V_base[2, 3] = 1; V_base[0, 3] = -1
    # Term 5
    U_base[0, 4] = 1; U_base[1, 4] = 1
    V_base[3, 4] = 1
    # Term 6
    U_base[2, 5] = 1; U_base[0, 5] = -1
    V_base[0, 5] = 1; V_base[1, 5] = 1
    # Term 7
    U_base[1, 6] = 1; U_base[3, 6] = -1
    V_base[2, 6] = 1; V_base[3, 6] = 1
    
    # Output weights W_base
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    W_base[0, 0] = 1 # M1 to C00
    W_base[0, 3] = 1 # M4 to C00
    W_base[0, 4] = -1 # M5 to C00
    W_base[0, 6] = 1 # M7 to C00
    
    W_base[1, 2] = 1 # M3 to C01
    W_base[1, 4] = 1 # M5 to C01
    
    W_base[2, 1] = 1 # M2 to C10
    W_base[2, 3] = 1 # M4 to C10
    
    W_base[3, 0] = 1 # M1 to C11
    W_base[3, 1] = -1 # M2 to C11
    W_base[3, 2] = 1 # M3 to C11
    W_base[3, 5] = 1 # M6 to C11
    
    # Now build 4x4
    # Global U: 16x49
    # Global V: 16x49
    # Global W: 16x49
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Mapping of 2x2 blocks to global indices
    # A blocks:
    # A11: 0,1,4,5
    # A12: 2,3,6,7
    # A21: 8,9,12,13
    # A22: 10,11,14,15
    # (Row-major: i*4+j)
    
    # Top-level Strassen linear combinations
    # A'1 = A11 + A22
    # A'2 = A21 + A22
    # A'3 = A11
    # A'4 = A22
    # A'5 = A11 + A12
    # A'6 = A21 - A11
    # A'7 = A12 - A22
    
    # B'1 = B11 + B22
    # B'2 = B11
    # B'3 = B12 - B22
    # B'4 = B21 - B11
    # B'5 = B22
    # B'6 = B11 + B12
    # B'7 = B21 + B22
    
    # We define the coefficients for A'K and B'K
    # Each is a 4x4 matrix (mapping global 16 indices to 4 local indices)
    # Or simply, for each K, we define which global indices contribute and with what weight.
    
    # Let's use a list of (block_indices, coefficients) for each A'K
    # Block indices: 0=A11, 1=A12, 2=A21, 3=A22
    # Each block is 4 global indices.
    
    A_blocks = [
        [0, 1, 4, 5],   # A11
        [2, 3, 6, 7],   # A12
        [8, 9, 12, 13], # A21
        [10, 11, 14, 15] # A22
    ]
    B_blocks = A_blocks # Same structure for B
    
    # A'K definitions: list of (block_id, coeff)
    A_defs = [
        [(0, 1), (3, 1)],      # A'1
        [(2, 1), (3, 1)],      # A'2
        [(0, 1)],              # A'3
        [(3, 1)],              # A'4
        [(0, 1), (1, 1)],      # A'5
        [(2, 1), (0, -1)],     # A'6
        [(1, 1), (3, -1)]      # A'7
    ]
    
    B_defs = [
        [(0, 1), (3, 1)],      # B'1
        [(0, 1)],              # B'2
        [(1, 1), (3, -1)],     # B'3
        [(2, 1), (0, -1)],     # B'4
        [(3, 1)],              # B'5
        [(0, 1), (1, 1)],      # B'6
        [(2, 1), (3, 1)]       # B'7
    ]
    
    # Top-level output weights
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # M_K is a 2x2 matrix.
    # C11 is block 0, C12 is block 1, C21 is block 2, C22 is block 3.
    # So for each M_K, we add its 2x2 result to the corresponding C blocks with specific weights.
    
    # C_block_weights[K][C_block_id]
    C_weights = [
        {0: 1, 1: 0, 2: 0, 3: 0}, # M1 -> C11
        {0: 0, 1: 0, 2: 1, 3: 0}, # M2 -> C21
        {0: 0, 1: 1, 2: 0, 3: 0}, # M3 -> C12
        {0: 1, 1: 0, 2: 1, 3: 0}, # M4 -> C11, C21
        {0: -1, 1: 1, 2: 0, 3: 0},# M5 -> C11(-1), C12(1)
        {0: 0, 1: 0, 2: 0, 3: 1}, # M6 -> C22
        {0: 1, 1: 0, 2: 0, 3: 1}  # M7 -> C11, C22
    ]
    
    col = 0
    for K in range(7):
        # For each of the 7 base Strassen terms
        for k_prime in range(7):
            # Construct U column
            # U_global[global_idx, col] = sum over local_idx of U_base[local_idx, k_prime] * coeff_A
            # local_idx 0..3 correspond to the 4 entries of the 2x2 matrix A'_K
            
            # First, get the linear form from U_base
            u_local = U_base[:, k_prime]
            v_local = V_base[:, k_prime]
            w_local = W_base[:, k_prime]
            
            # Map u_local to global U
            # A'_K is formed by A_defs[K]
            # A_defs[K] is a list of (block_id, coeff)
            for block_id, coeff in A_defs[K]:
                global_indices = A_blocks[block_id]
                for local_pos, g_idx in enumerate(global_indices):
                    # u_local[local_pos] is the weight for the local entry
                    U[g_idx, col] += u_local[local_pos] * coeff
            
            # Map v_local to global V
            for block_id, coeff in B_defs[K]:
                global_indices = B_blocks[block_id]
                for local_pos, g_idx in enumerate(global_indices):
                    V[g_idx, col] += v_local[local_pos] * coeff
            
            # Map w_local to global W
            # w_local[local_pos] is the weight for the output entry of M_K
            # M_K contributes to C blocks with weights from C_weights[K]
            # local_pos 0..3 correspond to C00, C01, C10, C11 of the 2x2 result M_K
            # These map to specific positions in the global C blocks.
            
            # C blocks:
            # C11 (block 0): indices 0,1,4,5
            # C12 (block 1): indices 2,3,6,7
            # C21 (block 2): indices 8,9,12,13
            # C22 (block 3): indices 10,11,14,15
            
            # local_pos 0 (C00 of M_K) -> top-left of the C block
            # local_pos 1 (C01 of M_K) -> top-right of the C block
            # local_pos 2 (C10 of M_K) -> bottom-left of the C block
            # local_pos 3 (C11 of M_K) -> bottom-right of the C block
            
            # For each C block that M_K contributes to:
            for c_block_id, weight in C_weights[K].items():
                if weight == 0:
                    continue
                global_indices = A_blocks[c_block_id] # Same index layout for C
                # Map local_pos to offset in global_indices
                # 0->0, 1->1, 2->2, 3->3 relative to block start
                # But global_indices are already the 4 indices of the block.
                # local_pos 0 -> global_indices[0]
                # local_pos 1 -> global_indices[1]
                # local_pos 2 -> global_indices[2]
                # local_pos 3 -> global_indices[3]
                
                for local_pos in range(4):
                    if w_local[local_pos] != 0:
                        g_idx = global_indices[local_pos]
                        W[g_idx, col] += w_local[local_pos] * weight
            
            col += 1
    
    # Convert to lists for output
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
