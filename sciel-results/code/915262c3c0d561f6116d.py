"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-33 exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    
    Strategy: Block the output into 3 independent 2x2 blocks (since p=6, n=2).
    Each block corresponds to multiplying a 2x3 matrix A by a 3x2 matrix B_block.
    We use a rank-11 decomposition for the 2x3x2 tensor.
    Total rank = 3 blocks * 11 = 33.
    
    The decomposition is:
    1. A rank-7 Strassen-like decomposition for the top-left 2x2 sub-blocks (columns 0,1 of A and rows 0,1 of B).
    2. Four rank-1 terms for the remaining cross-terms involving the 3rd column of A and 3rd row of B.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # We partition the p=6 dimension into 3 blocks of size 2.
    # For each block b (0..2), the local B matrix is B[:, 2*b : 2*b+2] which is 3x2.
    # The local C block is C[:, 2*b : 2*b+2] which is 2x2.
    # The A matrix is the same 2x3 matrix for all blocks.
    
    num_blocks = 3
    block_size = 2
    rank_per_block = 11
    total_rank = num_blocks * rank_per_block  # 33
    
    # Initialize factor matrices
    # U: (n*m, rank) = (6, 33)
    # V: (m*p, rank) = (18, 33)
    # W: (p*n, rank) = (12, 33)
    
    # We will build these as lists of lists for clarity, then return them.
    # The evaluator expects specific shapes and integer/half-integer entries.
    
    # Define the 11 rank-1 terms for a single 2x3x2 block.
    # Local indices:
    # A (2x3): a[i,j] with i in {0,1}, j in {0,1,2}. Flat index = i*3 + j.
    # B (3x2): b[j,k] with j in {0,1,2}, k in {0,1}. Flat index = j*2 + k.
    # C (2x2): c[i,k] with i in {0,1}, k in {0,1}. Flat index = i*2 + k.
    # Note: The tensor T_{i,j,k} = sum_u U_{i,j,u} V_{j,k,u} W_{i,k,u} (schematically, though W index is (k,i) in output)
    # Actually, the decomposition is T = sum_r u_r ⊗ v_r ⊗ w_r.
    # U_r has entries for (i,j). V_r has entries for (j,k). W_r has entries for (k,i).
    
    # Term definitions: (dict_u, dict_v, dict_w)
    # Keys are flat indices, values are coefficients.
    
    terms = []
    
    # Strassen terms for the 2x2 sub-problem (using A cols 0,1 and B rows 0,1)
    # A_sub = [[A00, A01], [A10, A11]] -> indices: 0, 1, 3, 4
    # B_sub = [[B00, B01], [B10, B11]] -> indices: 0, 1, 2, 3
    
    # M1 = (A00+A11)(B00+B11)
    terms.append(({0: 1, 4: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}))
    
    # M2 = (A10+A11)B00
    terms.append(({3: 1, 4: 1}, {0: 1}, {2: 1, 3: -1}))
    
    # M3 = A00(B01-B11)
    terms.append(({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}))
    
    # M4 = A11(B10-B00)
    terms.append(({4: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}))
    
    # M5 = (A00+A01)B11
    terms.append(({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}))
    
    # M6 = (A10-A00)(B00+B01)
    terms.append(({3: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}))
    
    # M7 = (A01-A11)(B10+B11)
    terms.append(({1: 1, 4: -1}, {2: 1, 3: 1}, {0: 1}))
    
    # Cross terms for the 3rd column of A (j=2) and 3rd row of B (j=2)
    # A_col2 = [A02, A12] -> indices: 2, 5
    # B_row2 = [B20, B21] -> indices: 4, 5
    
    # K1 = A02 * B20 -> C00
    terms.append(({2: 1}, {4: 1}, {0: 1}))
    
    # K2 = A02 * B21 -> C01
    terms.append(({2: 1}, {5: 1}, {1: 1}))
    
    # K3 = A12 * B20 -> C10
    terms.append(({5: 1}, {4: 1}, {2: 1}))
    
    # K4 = A12 * B21 -> C11
    terms.append(({5: 1}, {5: 1}, {3: 1}))
    
    # Now assemble the global matrices.
    # U_global: shape (6, 33)
    # V_global: shape (18, 33)
    # W_global: shape (12, 33)
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    for b in range(num_blocks):
        base_col = b * rank_per_block
        base_k = b * block_size  # Start of k-index for this block
        
        for t_idx in range(rank_per_block):
            col = base_col + t_idx
            u_dict, v_dict, w_dict = terms[t_idx]
            
            # Fill U
            # U index: i * m + j
            for l_idx, val in u_dict.items():
                U[l_idx, col] = val
            
            # Fill V
            # V index: j * p + k
            # Local v index: j * block_size + k_local
            for l_idx, val in v_dict.items():
                j_local = l_idx // block_size
                k_local = l_idx % block_size
                g_idx = j_local * p + (base_k + k_local)
                V[g_idx, col] = val
            
            # Fill W
            # W index: k * n + i (trace convention W[k*n+i, r])
            # Local w index: i * block_size + k_local
            # Wait, the local W index in the term definition was i*2+k.
            # Let's re-verify the local W index definition.
            # In the terms above, e.g., M1 w: {0: 1, 3: 1}.
            # If local index is i*2+k:
            # 0 -> i=0, k=0
            # 3 -> i=1, k=1
            # This matches C00 and C11 contributions.
            # So local w index = i * block_size + k_local.
            for l_idx, val in w_dict.items():
                i_local = l_idx // block_size
                k_local = l_idx % block_size
                g_idx = (base_k + k_local) * n + i_local
                W[g_idx, col] = val
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
