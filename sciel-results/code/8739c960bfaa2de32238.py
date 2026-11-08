import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_2x2_decomposition():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen (rank 7).
    U: 4x7, V: 4x7, W: 4x7
    Indices:
    U[i*2+j, r]
    V[j*2+k, r]
    W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # A indices: 00->0, 01->1, 10->2, 11->3
    # B indices: 00->0, 01->1, 10->2, 11->3
    # C indices (k*2+i): 00->0, 10->1 (wait, k=0,i=1->1), 01->2 (k=1,i=0->2), 11->3 (k=1,i=1->3)
    # Let's verify C index mapping:
    # C[0,0] -> k=0, i=0 -> 0
    # C[0,1] -> k=1, i=0 -> 2
    # C[1,0] -> k=0, i=1 -> 1
    # C[1,1] -> k=1, i=1 -> 3
    
    # M1 = (A00 + A11)(B00 + B11)
    U[:, 0][[0, 3]] = 1
    V[:, 0][[0, 3]] = 1
    W[:, 0][[0, 3]] = 1  # C00 and C11
    
    # M2 = (A10 + A11)B00
    U[:, 1][[2, 3]] = 1
    V[:, 1][[0]] = 1
    W[:, 1][[1]] = 1     # C10
    
    # M3 = A00(B01 - B11)
    U[:, 2][[0]] = 1
    V[:, 2][[1]] = 1
    V[:, 2][[3]] = -1
    W[:, 2][[2]] = 1     # C01
    W[:, 2][[3]] = -1    # C11
    
    # M4 = A11(B10 - B00)
    U[:, 3][[3]] = 1
    V[:, 3][[2]] = 1
    V[:, 3][[0]] = -1
    W[:, 3][[0]] = -1    # C00
    W[:, 3][[1]] = 1     # C10
    
    # M5 = (A00 + A01)B11
    U[:, 4][[0, 1]] = 1
    V[:, 4][[3]] = 1
    W[:, 4][[2]] = 1     # C01
    W[:, 4][[3]] = 1     # C11
    
    # M6 = (A10 - A00)(B00 + B01)
    U[:, 5][[2]] = 1
    U[:, 5][[0]] = -1
    V[:, 5][[0, 1]] = 1
    W[:, 5][[0]] = -1    # C00
    W[:, 5][[1]] = 1     # C10
    
    # M7 = (A01 - A11)(B10 + B11)
    U[:, 6][[1]] = 1
    U[:, 6][[3]] = -1
    V[:, 6][[2, 3]] = 1
    W[:, 6][[0]] = 1     # C00
    W[:, 6][[2]] = 1     # C01
    
    return U, V, W

def _build_4x4_from_strassen():
    """
    Builds U, V, W for 4x4x4 matrix multiplication using recursive Strassen (rank 49).
    Uses the 2x2 Strassen decomposition for each of the 7 block multiplications.
    """
    U_2, V_2, W_2 = _strassen_2x2_decomposition()
    
    # We need to construct the 4x4 matrices.
    # A is 4x4, split into 4 blocks of 2x2.
    # B is 4x4, split into 4 blocks of 2x2.
    # C is 4x4, split into 4 blocks of 2x2.
    
    # The 7 terms of Strassen for 4x4 are linear combinations of products of 2x2 blocks.
    # Let A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # Let B = [[B00, B01], [B10, B11]] where Bij are 2x2.
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Each Mj is a 2x2 matrix product.
    # We replace each product with the 7-term decomposition.
    # This creates 7*7 = 49 terms.
    
    U_4 = np.zeros((16, 49))
    V_4 = np.zeros((16, 49))
    W_4 = np.zeros((16, 49))
    
    # Helper to add a term
    def add_term(u_coeff, v_coeff, w_coeff, a_idx, b_idx, c_idx):
        # a_idx, b_idx, c_idx are indices of the 2x2 block (0..3)
        # We need to expand the 2x2 block indices into 4x4 indices.
        # Block 0 (00): rows 0-1, cols 0-1. Global indices: 0,1,4,5
        # Block 1 (01): rows 0-1, cols 2-3. Global indices: 2,3,6,7
        # Block 2 (10): rows 2-3, cols 0-1. Global indices: 8,9,12,13
        # Block 3 (11): rows 2-3, cols 2-3. Global indices: 10,11,14,15
        
        # For U (A):
        # Local index in 2x2 is l (0..3).
        # Global index = block_row * 4 + block_col * 2 + local_col? No.
        # A is 4x4. Row i, Col j.
        # Block (br, bc) contains A[br*2:r+1, bc*2:c+1].
        # Global row = br*2 + local_row.
        # Global col = bc*2 + local_col.
        # Flat index = Global row * 4 + Global col.
        
        # u_coeff is a 4x1 vector (from U_2 column).
        # For each local index l in 0..3:
        for l in range(4):
            coeff = u_coeff[l]
            if coeff == 0: continue
            lr = l // 2
            lc = l % 2
            gr = a_idx * 2 + lr  # a_idx is block index 0..3. br=a_idx//2, bc=a_idx%2
            bc_a = a_idx % 2
            br_a = a_idx // 2
            gr = br_a * 2 + lr
            gc = bc_a * 2 + lc
            global_idx = gr * 4 + gc
            U_4[global_idx, target_col] += coeff
            
        # For V (B):
        for l in range(4):
            coeff = v_coeff[l]
            if coeff == 0: continue
            lr = l // 2
            lc = l % 2
            br_b = b_idx // 2
            bc_b = b_idx % 2
            gr = br_b * 2 + lr
            gc = bc_b * 2 + lc
            global_idx = gr * 4 + gc
            V_4[global_idx, target_col] += coeff
            
        # For W (C):
        # W index is k*4 + i (since n=4, p=4).
        # C is 4x4. Row i, Col k.
        # Block (br, bc) corresponds to C[br*2:br*2+2, bc*2:bc*2+2].
        # So local row = i, local col = k.
        for l in range(4):
            coeff = w_coeff[l]
            if coeff == 0: continue
            # l in W_2 corresponds to k*2+i for 2x2.
            # So local_k = l // 2, local_i = l % 2.
            lk = l // 2
            li = l % 2
            br_c = c_idx // 2
            bc_c = c_idx % 2
            global_i = br_c * 2 + li
            global_k = bc_c * 2 + lk
            global_idx = global_k * 4 + global_i
            W_4[global_idx, target_col] += coeff

    # Define the 7 Strassen terms for 4x4 in terms of block indices
    # Term 0: M1 = (A00 + A11)(B00 + B11)
    # A parts: A00 (idx 0), A11 (idx 3)
    # B parts: B00 (idx 0), B11 (idx 3)
    # C parts: C00 (idx 0), C11 (idx 3) -- wait, Strassen reconstruction for 2x2.
    # The W coefficients come from the 2x2 Strassen reconstruction.
    # But here, the "result" is a 2x2 matrix M1.
    # When we assemble the final C, we use the 2x2 Strassen formula.
    # So the W coefficients for the 4x4 case are determined by how M1 contributes to C blocks.
    
    # Let's re-evaluate.
    # The decomposition is for the tensor $T$.
    # $T = \sum_{r=1}^{49} u_r \otimes v_r \otimes w_r$.
    # The $w_r$ components are fixed by the Strassen reconstruction formula applied to the blocks.
    
    # For M1 = (A00+A11)(B00+B11):
    # This term contributes to C blocks as per the 2x2 Strassen formula:
    # C00 += 1 * M1
    # C01 += 0 * M1
    # C10 += 0 * M1
    # C11 += 1 * M1
    
    # So for the "C" part of this term, we have contributions to C00 and C11.
    # The coefficients are 1.
    
    # We iterate over the 7 "outer" Strassen terms (M1..M7).
    # For each outer term, we have a specific U_block, V_block, W_block mapping.
    # But actually, U and V are just linear combinations of A and B blocks.
    # W is a linear combination of C blocks.
    
    # Let's define the structure for each of the 7 outer terms.
    # Outer Term 0 (M1):
    # U involves A00, A11.
    # V involves B00, B11.
    # W involves C00, C11.
    
    # We can construct the 4x4 matrices by "inflating" the 2x2 Strassen coefficients.
    
    outer_terms = [
        # (u_blocks, v_blocks, w_blocks)
        # u_blocks: list of (block_idx, coeff)
        # v_blocks: list of (block_idx, coeff)
        # w_blocks: list of (block_idx, coeff)
        {
            'u': [(0, 1), (3, 1)],
            'v': [(0, 1), (3, 1)],
            'w': [(0, 1), (3, 1)] # C00, C11
        },
        {
            'u': [(2, 1), (3, 1)],
            'v': [(0, 1)],
            'w': [(1, 1)] # C10
        },
        {
            'u': [(0, 1)],
            'v': [(1, 1), (3, -1)],
            'w': [(2, 1), (3, -1)] # C01, C11
        },
        {
            'u': [(3, 1)],
            'v': [(2, 1), (0, -1)],
            'w': [(0, -1), (1, 1)] # C00, C10
        },
        {
            'u': [(0, 1), (1, 1)],
            'v': [(3, 1)],
            'w': [(2, 1), (3, 1)] # C01, C11
        },
        {
            'u': [(2, 1), (0, -1)],
            'v': [(0, 1), (1, 1)],
            'w': [(0, -1), (1, 1)] # C00, C10
        },
        {
            'u': [(1, 1), (3, -1)],
            'v': [(2, 1), (3, 1)],
            'w': [(0, 1), (2, 1)] # C00, C01
        }
    ]
    
    for outer_idx, term in enumerate(outer_terms):
        for inner_idx in range(7):
            col = outer_idx * 7 + inner_idx
            
            # Get the 2x2 Strassen vectors for the inner multiplication
            u_vec = U_2[:, inner_idx]
            v_vec = V_2[:, inner_idx]
            w_vec = W_2[:, inner_idx]
            
            # Add contributions for U
            for a_block_idx, a_coeff in term['u']:
                # a_coeff is the coefficient from the outer Strassen formula
                # We multiply it with the inner U vector components
                for l in range(4):
                    val = a_coeff * u_vec[l]
                    if val == 0: continue
                    lr = l // 2
                    lc = l % 2
                    br = a_block_idx // 2
                    bc = a_block_idx % 2
                    gr = br * 2 + lr
                    gc = bc * 2 + lc
                    U_4[gr * 4 + gc, col] += val
                    
            # Add contributions for V
            for b_block_idx, b_coeff in term['v']:
                for l in range(4):
                    val = b_coeff * v_vec[l]
                    if val == 0: continue
                    lr = l // 2
                    lc = l % 2
                    br = b_block_idx // 2
                    bc = b_block_idx % 2
                    gr = br * 2 + lr
                    gc = bc * 2 + lc
                    V_4[gr * 4 + gc, col] += val
                    
            # Add contributions for W
            for c_block_idx, c_coeff in term['w']:
                for l in range(4):
                    val = c_coeff * w_vec[l]
                    if val == 0: continue
                    lk = l // 2
                    li = l % 2
                    br = c_block_idx // 2
                    bc = c_block_idx % 2
                    gi = br * 2 + li
                    gk = bc * 2 + lk
                    W_4[gk * 4 + gi, col] += val

    return U_4, V_4, W_4

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=8
    
    U_4, V_4, W_4 = _build_4x4_from_strassen()
    
    # We need to split B (4x8) into B_L (4x4) and B_R (4x4).
    # And C (4x8) into C_L (4x4) and C_R (4x4).
    
    # Total rank is 49 + 49 = 98.
    rank = 98
    
    U_final = np.zeros((16, rank))
    V_final = np.zeros((32, rank))
    W_final = np.zeros((32, rank))
    
    # Left half (B_L, C_L)
    # U is same as U_4
    U_final[:, 0:49] = U_4
    
    # V is V_4 for B_L (cols 0-3 of B)
    V_final[0:16, 0:49] = V_4
    
    # W is W_4 for C_L (cols 0-3 of C, which are indices 0-15 in W_final?
    # W index is k*n+i.
    # For C_L (4x4), k is 0..3. W index is k*4+i. Range 0..15.
    W_final[0:16, 0:49] = W_4
    
    # Right half (B_R, C_R)
    # U is same as U_4
    U_final[:, 49:98] = U_4
    
    # V is V_4 for B_R (cols 4-7 of B).
    # B indices: Row j, Col k.
    # For B_R, k is 4..7.
    # Flat index in V is j*8 + k? No, V is (m*p, rank) = (4*8, 98) = (32, 98).
    # Index is j*8 + k.
    # V_4 has index j*4 + k' where k' is 0..3.
    # We need to map k' to k = k' + 4.
    # So V_final[j*8 + (k'+4), 49:98] = V_4[j*4+k', 0:49]
    for j in range(4):
        for k_prime in range(4):
            V_final[j*8 + k_prime + 4, 49:98] = V_4[j*4 + k_prime, 0:49]
    
    # W is W_4 for C_R (cols 4-7 of C).
    # C indices: Row i, Col k.
    # For C_R, k is 4..7.
    # W index is k*4 + i.
    # For C_R, k is 4..7.
    # W_4 has index k'*4 + i where k' is 0..3.
    # We need to map k' to k = k' + 4.
    # So W_final[(k'+4)*4 + i, 49:98] = W_4[k'*4 + i, 0:49]
    for k_prime in range(4):
        for i in range(4):
            W_final[(k_prime + 4) * 4 + i, 49:98] = W_4[k_prime * 4 + i, 0:49]
            
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
# EVOLVE-BLOCK-END
