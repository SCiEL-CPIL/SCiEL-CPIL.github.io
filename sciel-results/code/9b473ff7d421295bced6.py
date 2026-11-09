import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin
    
    n, m, p = 4, 4, 6
    
    # Part 1: Strassen 4x4 for B[:, 0:4] -> C[:, 0:4]
    # Rank 49
    U_L, V_L, W_L = _strassen_4x4()
    
    # Part 2: Schoolbook 4x4x2 for B[:, 4:6] -> C[:, 4:6]
    # Rank 32
    U_R, V_R, W_R = _schoolbook_4x4x2()
    
    # Assemble
    rank_total = 49 + 32
    U = np.zeros((16, rank_total))
    V = np.zeros((24, rank_total))
    W = np.zeros((24, rank_total))
    
    # Left part (cols 0-48)
    U[:, 0:49] = U_L
    V[0:16, 0:49] = V_L
    W[0:16, 0:49] = W_L
    
    # Right part (cols 49-80)
    U[:, 49:81] = U_R
    V[16:24, 49:81] = V_R
    W[16:24, 49:81] = W_R
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _strassen_4x4():
    """
    Returns U, V, W for 4x4x4 matrix multiplication using Strassen (rank 49).
    U shape: (16, 49)
    V shape: (16, 49)
    W shape: (16, 49)
    """
    # Get 2x2 Strassen factors
    U2, V2, W2 = _strassen_2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Strassen formulas for 4x4 using 2x2 blocks
    # Blocks A00, A01, A10, A11 correspond to indices:
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    # Same for B and C.
    
    # Define block indices for A, B, C
    # idx = [ [0,1,4,5], [2,3,6,7], [8,9,12,13], [10,11,14,15] ]
    # Block 0: 0,1,4,5
    # Block 1: 2,3,6,7
    # Block 2: 8,9,12,13
    # Block 3: 10,11,14,15
    
    def get_block_indices(block_id):
        # block_id 0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right
        if block_id == 0: return [0, 1, 4, 5]
        if block_id == 1: return [2, 3, 6, 7]
        if block_id == 2: return [8, 9, 12, 13]
        if block_id == 3: return [10, 11, 14, 15]

    # Strassen 7 steps
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Mi is a 2x2x2 matmul.
    # We have 7 such matmuls. Each uses 7 scalar multiplications (from _strassen_2x2).
    # Total 49 scalar multiplications.
    
    # The output C is composed of:
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 - M5
    
    # Let's populate U and V for each of the 7 super-steps.
    # For step s (0..6), the columns in U,V,W are s*7 to s*7+6.
    
    # Define the linear combinations for A and B for each step
    # Step 1: A00+A11, B00+B11
    # A blocks involved: 0, 3. Coeffs: 1, 1.
    # B blocks involved: 0, 3. Coeffs: 1, 1.
    # Step 2: A10+A11, B00
    # A blocks: 2, 3. Coeffs: 1, 1.
    # B blocks: 0. Coeffs: 1.
    # Step 3: A00, B01-B11
    # A blocks: 0. Coeffs: 1.
    # B blocks: 1, 3. Coeffs: 1, -1.
    # Step 4: A11, B10-B00
    # A blocks: 3. Coeffs: 1.
    # B blocks: 2, 0. Coeffs: 1, -1.
    # Step 5: A00+A01, B11
    # A blocks: 0, 1. Coeffs: 1, 1.
    # B blocks: 3. Coeffs: 1.
    # Step 6: A10-A00, B00+B01
    # A blocks: 2, 0. Coeffs: 1, -1.
    # B blocks: 0, 1. Coeffs: 1, 1.
    # Step 7: A01-A11, B10+B11
    # A blocks: 1, 3. Coeffs: 1, -1.
    # B blocks: 2, 3. Coeffs: 1, 1.
    
    steps = [
        # (A_blocks, A_coeffs, B_blocks, B_coeffs)
        ([0, 3], [1, 1], [0, 3], [1, 1]),
        ([2, 3], [1, 1], [0], [1]),
        ([0], [1], [1, 3], [1, -1]),
        ([3], [1], [2, 0], [1, -1]),
        ([0, 1], [1, 1], [3], [1]),
        ([2, 0], [1, -1], [0, 1], [1, 1]),
        ([1, 3], [1, -1], [2, 3], [1, 1])
    ]
    
    for s in range(7):
        a_blocks, a_coeffs, b_blocks, b_coeffs = steps[s]
        col_offset = s * 7
        
        # Populate U for this step
        # U[:, col_offset:col_offset+7] should represent the linear combination of A
        # U2 is (4,7). It maps A indices (0..3) to 7 factors.
        # We need to map A block indices to A element indices.
        
        for i in range(4):
            # U2[i, k] is coefficient for A element i in factor k
            for k in range(7):
                coeff = 0
                for b_idx, coeff_val in zip(a_blocks, a_coeffs):
                    # Map block index to element index
                    # Block 0: 0,1,4,5
                    # Block 1: 2,3,6,7
                    # Block 2: 8,9,12,13
                    # Block 3: 10,11,14,15
                    elem_indices = get_block_indices(b_idx)
                    # i is 0..3 within the block
                    elem_idx = elem_indices[i]
                    coeff += coeff_val * U2[i, k]
                U[elem_idx, col_offset + k] = coeff
                
        # Populate V for this step
        for i in range(4):
            for k in range(7):
                coeff = 0
                for b_idx, coeff_val in zip(b_blocks, b_coeffs):
                    elem_indices = get_block_indices(b_idx)
                    elem_idx = elem_indices[i]
                    coeff += coeff_val * V2[i, k]
                V[elem_idx, col_offset + k] = coeff
                
    # Populate W
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 - M5
    
    # M1 is step 0, M2 is step 1, ..., M7 is step 6.
    # W2 is (4,7). It maps C elements (0..3) to 7 factors.
    
    # C Block 0 (C00) corresponds to C elements 0,1,4,5
    # C Block 1 (C01) corresponds to C elements 2,3,6,7
    # C Block 2 (C10) corresponds to C elements 8,9,12,13
    # C Block 3 (C11) corresponds to C elements 10,11,14,15
    
    # For each C block, we sum contributions from M steps.
    
    # C00: M1(0) - M4(3) + M6(5) + M7(6)
    # C01: M3(2) + M5(4)
    # C10: M2(1) + M4(3)
    # C11: M1(0) - M3(2) - M5(4)
    
    c_block_steps = {
        0: [(0, 1), (3, -1), (5, 1), (6, 1)], # C00
        1: [(2, 1), (4, 1)],                   # C01
        2: [(1, 1), (3, 1)],                   # C10
        3: [(0, 1), (2, -1), (4, -1)]          # C11
    }
    
    for c_block_id in range(4):
        elem_indices = get_block_indices(c_block_id)
        for i in range(4):
            elem_idx = elem_indices[i]
            for k in range(7):
                coeff = 0
                for step_idx, coeff_val in c_block_steps[c_block_id]:
                    coeff += coeff_val * W2[i, k]
                # Wait, W2[i,k] is the coefficient for C element i in factor k of the 2x2 decomposition.
                # We are summing over the 7 super-steps.
                # The factor index in the 49-dim space is step_idx * 7 + k.
                # So we add to W[elem_idx, step_idx*7+k]
                
                # Actually, it's easier to iterate over the 49 columns.
                pass
                
    # Let's redo W population more directly.
    W = np.zeros((16, 49))
    for c_block_id in range(4):
        elem_indices = get_block_indices(c_block_id)
        for i in range(4):
            elem_idx = elem_indices[i]
            for step_idx, coeff_val in c_block_steps[c_block_id]:
                # Add coeff_val * W2[i, :] to W[elem_idx, step_idx*7 : step_idx*7+7]
                W[elem_idx, step_idx*7 : step_idx*7+7] += coeff_val * W2[i, :]
                
    return U, V, W

def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen (rank 7).
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    """
    # Indices: 0:A00, 1:A01, 2:A10, 3:A11
    # B: 0:B00, 1:B01, 2:B10, 3:B11
    # C: 0:C00, 1:C01, 2:C10, 3:C11
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A00: M1, M3, M5, M6
        [0, 0, 0, 0, 1, 0, 1],  # A01: M5, M7
        [0, 1, 0, 0, 0, 1, 0],  # A10: M2, M6
        [1, 1, 0, 1, 0, 0, -1]  # A11: M1, M2, M4, M7
    ], dtype=float)
    
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B00: M1, M2, M4, M6
        [0, 0, 1, 0, 0, 1, 0],  # B01: M3, M6
        [0, 0, 0, 1, 0, 0, 1],  # B10: M4, M7
        [1, 0, -1, 0, 1, 0, 1]  # B11: M1, M3, M5, M7
    ], dtype=float)
    
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 - M5
    
    W = np.array([
        [1, 0, 0, -1, 0, 1, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, 0, -1, -1, -1, 0, 0] # C11
    ], dtype=float)
    
    return U, V, W

def _schoolbook_4x4x2():
    """
    Returns U, V, W for 4x4x2 matrix multiplication (schoolbook, rank 32).
    U shape: (16, 32)
    V shape: (8, 32)
    W shape: (8, 32)
    """
    n, m, p = 4, 4, 2
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

# EVOLVE-BLOCK-END
