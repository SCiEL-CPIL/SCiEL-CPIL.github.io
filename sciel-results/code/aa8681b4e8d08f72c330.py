"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _get_strassen_2x2_UVW():
    """
    Returns U, V, W for 2x2 matrix multiplication using Strassen's algorithm.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    Indices:
    A: 0,1,2,3 (row-major: a00, a01, a10, a11)
    B: 0,1,2,3 (row-major: b00, b01, b10, b11)
    C: 0,1,2,3 (row-major: c00, c01, c10, c11)
    """
    rank = 7
    U = np.zeros((4, rank), dtype=int)
    V = np.zeros((4, rank), dtype=int)
    W = np.zeros((4, rank), dtype=int)
    
    # M1 = (a00 + a11)(b00 + b11)
    # U: a00(0), a11(3)
    # V: b00(0), b11(3)
    # W: c00(0)+1, c11(3)+1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (a10 + a11)b00
    # U: a10(2), a11(3)
    # V: b00(0)
    # W: c10(2)+1, c11(3)-1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a00(b01 - b11)
    # U: a00(0)
    # V: b01(1), b11(3)
    # W: c01(1)+1, c11(3)+1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = a11(b10 - b00)
    # U: a11(3)
    # V: b10(2), b00(0)
    # W: c00(0)+1, c10(2)+1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a00 + a01)b11
    # U: a00(0), a01(1)
    # V: b11(3)
    # W: c00(0)-1, c01(1)+1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (a10 - a00)(b00 + b01)
    # U: a10(2), a00(0)
    # V: b00(0), b01(1)
    # W: c11(3)+1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (a01 - a11)(b10 + b11)
    # U: a01(1), a11(3)
    # V: b10(2), b11(3)
    # W: c00(0)+1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _get_strassen_4x4_UVW():
    """
    Returns U, V, W for 4x4 matrix multiplication using Strassen's algorithm.
    U shape: (16, 49)
    V shape: (16, 49)
    W shape: (16, 49)
    Indices are row-major 0..15.
    """
    U2, V2, W2 = _get_strassen_2x2_UVW()
    
    # Block indices for 4x4 matrix (row-major)
    # A0 (0,0): 0,1,4,5
    # A1 (0,1): 2,3,6,7
    # A2 (1,0): 8,9,12,13
    # A3 (1,1): 10,11,14,15
    block_indices = [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15]
    ]
    
    # Strassen coefficients for 4x4 (using 2x2 blocks)
    # M1 = (A0+A3)(B0+B3)
    # M2 = (A2+A3)B0
    # M3 = A0(B1-B3)
    # M4 = A3(B2-B0)
    # M5 = (A0+A1)B3
    # M6 = (A2-A0)(B0+B1)
    # M7 = (A1-A3)(B2+B3)
    
    # Each Mk is a 2x2 matrix product.
    # We use the 2x2 Strassen decomposition for each Mk.
    
    rank_4x4 = 49
    U = np.zeros((16, rank_4x4), dtype=int)
    V = np.zeros((16, rank_4x4), dtype=int)
    W = np.zeros((16, rank_4x4), dtype=int)
    
    # Define the linear combinations for A and B blocks for each of the 7 products
    # Each entry is a list of (block_index, coefficient)
    A_combos = [
        [(0, 1), (3, 1)], # M1
        [(2, 1), (3, 1)], # M2
        [(0, 1)],         # M3
        [(3, 1)],         # M4
        [(0, 1), (1, 1)], # M5
        [(2, 1), (0, -1)],# M6
        [(1, 1), (3, -1)] # M7
    ]
    B_combos = [
        [(0, 1), (3, 1)], # M1
        [(0, 1)],         # M2
        [(1, 1), (3, -1)],# M3
        [(2, 1), (0, -1)],# M4
        [(3, 1)],         # M5
        [(0, 1), (1, 1)], # M6
        [(2, 1), (3, 1)]  # M7
    ]
    
    # Output combinations
    # C0 = M1 + M4 - M5 + M7
    # C1 = M3 + M5
    # C2 = M2 + M4
    # C3 = M1 - M2 + M3 + M6
    # Each Ck is a 2x2 matrix.
    # We need to sum the contributions from each Mk to each Ck block.
    
    # Initialize W contributions
    # For each Mk (k=0..6), and for each 2x2 Strassen term (t=0..6),
    # we add the W2 contribution to the appropriate C block.
    
    # C block indices:
    # C0 (0,0): 0,1,4,5
    # C1 (0,1): 2,3,6,7
    # C2 (1,0): 8,9,12,13
    # C3 (1,1): 10,11,14,15
    
    # Mapping from Mk to C blocks and coefficients
    # C0 gets: M1(+1), M4(+1), M5(-1), M7(+1)
    # C1 gets: M3(+1), M5(+1)
    # C2 gets: M2(+1), M4(+1)
    # C3 gets: M1(+1), M2(-1), M3(+1), M6(+1)
    
    c_contributions = [
        [(0, 1), (3, 1), (4, -1), (6, 1)], # C0 from M0, M3, M4, M6
        [(2, 1), (4, 1)],                  # C1 from M2, M4
        [(1, 1), (3, 1)],                  # C2 from M1, M3
        [(0, 1), (1, -1), (2, 1), (5, 1)]  # C3 from M0, M1, M2, M5
    ]
    
    for k in range(7):
        # Build U and V for this Mk
        # U_Mk is 4x7 (local 2x2 indices)
        # We need to map to global 4x4 indices.
        
        # For each term t in 0..6 of the 2x2 Strassen:
        for t in range(7):
            # U2[:, t] has non-zeros at local indices.
            # V2[:, t] has non-zeros at local indices.
            # W2[:, t] has non-zeros at local indices.
            
            # Add to U
            for local_i in range(4):
                coeff_u = U2[local_i, t]
                if coeff_u != 0:
                    # Apply A_combos[k]
                    for block_idx, coeff_a in A_combos[k]:
                        # The local index local_i corresponds to a position in the 2x2 block.
                        # We need to add coeff_u * coeff_a to U[global_idx, rank_offset]
                        # Wait, U2 is for the "A" part of the 2x2 product.
                        # The "A" part of Mk is a linear combination of blocks.
                        # So for each block in the combo, we add the contribution.
                        
                        # The local index local_i maps to the same position in each block.
                        # E.g., if local_i=0 (top-left of 2x2), it maps to index 0 of block A0, index 2 of block A1, etc.
                        # Actually, block_indices[block_idx] gives the 4 global indices for that block.
                        # local_i=0 -> global index block_indices[block_idx][0]
                        # local_i=1 -> global index block_indices[block_idx][1]
                        # local_i=2 -> global index block_indices[block_idx][2]
                        # local_i=3 -> global index block_indices[block_idx][3]
                        
                        global_idx = block_indices[block_idx][local_i]
                        U[global_idx, k*7 + t] += coeff_u * coeff_a
                        
            # Add to V
            for local_j in range(4):
                coeff_v = V2[local_j, t]
                if coeff_v != 0:
                    for block_idx, coeff_b in B_combos[k]:
                        global_idx = block_indices[block_idx][local_j]
                        V[global_idx, k*7 + t] += coeff_v * coeff_b
                        
            # Add to W
            # W2[:, t] contributes to the 2x2 result of Mk.
            # This result is then added to the appropriate C blocks.
            for local_k in range(4):
                coeff_w = W2[local_k, t]
                if coeff_w != 0:
                    # Find which C blocks this Mk contributes to.
                    # We iterate over c_contributions to find if Mk is in the list.
                    # Actually, it's easier to pre-calculate.
                    pass
                    
    # The W construction is slightly more complex because one Mk contributes to multiple C blocks.
    # Let's redo W construction.
    W = np.zeros((16, rank_4x4), dtype=int)
    for k in range(7):
        # Mk contributes to some C blocks.
        # Find the C blocks that involve Mk.
        for c_block_idx in range(4):
            # Check if Mk is in c_contributions[c_block_idx]
            coeff_m = 0
            for m_idx, coeff in c_contributions[c_block_idx]:
                if m_idx == k:
                    coeff_m = coeff
                    break
            if coeff_m == 0:
                continue
            
            # Mk contributes to C_block_idx with coefficient coeff_m.
            # The 2x2 Strassen W2[:, t] gives the local 2x2 result.
            for t in range(7):
                for local_k in range(4):
                    coeff_w = W2[local_k, t]
                    if coeff_w != 0:
                        global_idx = block_indices[c_block_idx][local_k]
                        W[global_idx, k*7 + t] += coeff_w * coeff_m
                        
    return U, V, W

def _get_schoolbook_4x1_UVW():
    """
    Returns U, V, W for 4x4 * 4x1 multiplication (schoolbook).
    U shape: (16, 16)
    V shape: (4, 16)  # Only 1 column of B
    W shape: (4, 16)  # Only 1 column of C
    Note: The global V and W will have size 36, but here we return local parts.
    Actually, to make it consistent, I will return U, V, W with shapes matching the local dimensions.
    U: 16x16
    V: 4x16 (indices 0..3 for the single column of B)
    W: 4x16 (indices 0..3 for the single column of C)
    """
    rank = 16
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((4, rank), dtype=int)
    W = np.zeros((4, rank), dtype=int)
    
    for i in range(4):
        for j in range(4):
            idx = i*4 + j
            U[idx, idx] = 1
            V[j, idx] = 1
            W[i, idx] = 1
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=9
    
    # We split B into B1 (cols 0-3), B2a (cols 4-7), B2b (col 8).
    # C is split into C1 (cols 0-3), C2a (cols 4-7), C2b (col 8).
    
    # Part 1: A * B1 -> C1 (4x4 * 4x4)
    U1, V1, W1 = _get_strassen_4x4_UVW()
    # U1: 16x49, V1: 16x49, W1: 16x49
    
    # Part 2: A * B2a -> C2a (4x4 * 4x4)
    U2, V2, W2 = _get_strassen_4x4_UVW()
    
    # Part 3: A * B2b -> C2b (4x4 * 4x1)
    U3, V3, W3 = _get_schoolbook_4x1_UVW()
    # U3: 16x16, V3: 4x16, W3: 4x16
    
    total_rank = 49 + 49 + 16
    U_total = np.zeros((16, total_rank), dtype=int)
    V_total = np.zeros((36, total_rank), dtype=int)
    W_total = np.zeros((36, total_rank), dtype=int)
    
    # Assemble Part 1
    # U1 maps to U_total[:, 0:49]
    U_total[:, 0:49] = U1
    
    # V1 maps to V_total.
    # V1 indices are 0..15 (j*4+k for j=0..3, k=0..3).
    # Global V indices are j*9+k.
    for j in range(4):
        for k in range(4):
            local_idx = j*4 + k
            global_idx = j*9 + k
            V_total[global_idx, 0:49] = V1[local_idx, :]
            
    # W1 maps to W_total.
    # W1 indices are 0..15 (k*4+i for k=0..3, i=0..3).
    # Global W indices are k*4+i (since n=4).
    # Wait, the convention is W[k*n+i]. Here n=4.
    # So local index k*4+i is the same as global index k*4+i.
    W_total[0:16, 0:49] = W1
    
    # Assemble Part 2
    # U2 maps to U_total[:, 49:98]
    U_total[:, 49:98] = U2
    
    # V2 maps to V_total.
    # V2 indices are 0..15 (j*4+k for j=0..3, k=0..3).
    # These correspond to cols 4-7 of B.
    # Global V indices are j*9 + (k+4).
    for j in range(4):
        for k in range(4):
            local_idx = j*4 + k
            global_idx = j*9 + (k+4)
            V_total[global_idx, 49:98] = V2[local_idx, :]
            
    # W2 maps to W_total.
    # W2 indices are 0..15 (k*4+i for k=0..3, i=0..3).
    # These correspond to cols 4-7 of C.
    # Global W indices are (k+4)*4 + i.
    for k in range(4):
        for i in range(4):
            local_idx = k*4 + i
            global_idx = (k+4)*4 + i
            W_total[global_idx, 49:98] = W2[local_idx, :]
            
    # Assemble Part 3
    # U3 maps to U_total[:, 98:114]
    U_total[:, 98:114] = U3
    
    # V3 maps to V_total.
    # V3 indices are 0..3 (j*1+k for j=0..3, k=0).
    # These correspond to col 8 of B.
    # Global V indices are j*9 + 8.
    for j in range(4):
        local_idx = j
        global_idx = j*9 + 8
        V_total[global_idx, 98:114] = V3[local_idx, :]
        
    # W3 maps to W_total.
    # W3 indices are 0..3 (k*1+i for k=0, i=0..3).
    # These correspond to col 8 of C.
    # Global W indices are 8*4 + i = 32 + i.
    for i in range(4):
        local_idx = i
        global_idx = 32 + i
        W_total[global_idx, 98:114] = W3[local_idx, :]
        
    # Convert to lists of lists for the return format
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
