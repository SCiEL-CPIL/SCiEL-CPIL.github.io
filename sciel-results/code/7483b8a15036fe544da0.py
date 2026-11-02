import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def _strassen_2x2x2_factors():
    # Rank 7 decomposition for 2x2x2
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, -1]
    # M2
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 0, 1, -1]
    # M3
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 1, 0, -1]
    W[:, 2] = [0, 1, 0, 1]
    # M4
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [0, 0, 1, -1]
    W[:, 3] = [1, 0, 1, 0]
    # M5
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [-1, 1, 0, 0]
    # M6
    U[:, 5] = [0, 0, 1, -1]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [0, 0, 0, 1]
    # M7
    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    W[:, 6] = [1, 0, 0, 0]
    
    return U, V, W

def _schoolbook_2x2x1_factors():
    # Rank 4 decomposition for 2x2x1
    # A: 2x2, B: 2x1, C: 2x1
    # C[0] = A[0,0]B[0] + A[0,1]B[1]
    # C[1] = A[1,0]B[0] + A[1,1]B[1]
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    
    # Term 0: A00 B0
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    
    # Term 1: A01 B1
    U[1, 1] = 1
    V[1, 1] = 1
    W[0, 1] = 1
    
    # Term 2: A10 B0
    U[2, 2] = 1
    V[0, 2] = 1
    W[1, 2] = 1
    
    # Term 3: A11 B1
    U[3, 3] = 1
    V[1, 3] = 1
    W[1, 3] = 1
    
    return U, V, W

def _build_block_2x2x3_factors():
    # Rank 11 decomposition for 2x2x3
    # U: 4x11, V: 6x11, W: 6x11
    # A: 2x2 (idx 0,1,2,3)
    # B: 2x3 (idx 0,1,2, 3,4,5)
    # C: 2x3 (idx 0,1,2, 3,4,5)
    
    U_s, V_s, W_s = _strassen_2x2x2_factors()
    U_b, V_b, W_b = _schoolbook_2x2x1_factors()
    
    U = np.zeros((4, 11))
    V = np.zeros((6, 11))
    W = np.zeros((6, 11))
    
    # Fill Strassen part (cols 0-6)
    U[:, 0:7] = U_s
    
    # Map V_s (4x7) to V (6x7)
    # V_s rows: 0->B[0,0], 1->B[0,1], 2->B[1,0], 3->B[1,1]
    # Global B idx: 0, 1, 3, 4
    V[0, 0:7] = V_s[0, :]
    V[1, 0:7] = V_s[1, :]
    V[3, 0:7] = V_s[2, :]
    V[4, 0:7] = V_s[3, :]
    
    # Map W_s (4x7) to W (6x7)
    # W_s rows: 0->C[0,0], 1->C[0,1], 2->C[1,0], 3->C[1,1]
    # Global C idx: 0, 1, 3, 4
    W[0, 0:7] = W_s[0, :]
    W[1, 0:7] = W_s[1, :]
    W[3, 0:7] = W_s[2, :]
    W[4, 0:7] = W_s[3, :]
    
    # Fill Schoolbook part (cols 7-10)
    U[:, 7:11] = U_b
    
    # Map V_b (2x4) to V (6x4)
    # V_b rows: 0->B[0,2], 1->B[1,2]
    # Global B idx: 2, 5
    V[2, 7:11] = V_b[0, :]
    V[5, 7:11] = V_b[1, :]
    
    # Map W_b (2x4) to W (6x4)
    # W_b rows: 0->C[0,2], 1->C[1,2]
    # Global C idx: 2, 5
    W[2, 7:11] = W_b[0, :]
    W[5, 7:11] = W_b[1, :]
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Get block factors
    U_blk, V_blk, W_blk = _build_block_2x2x3_factors()
    
    # Global dimensions
    n, m, p = 4, 4, 6
    rank_total = 7 * 11
    U = np.zeros((n*m, rank_total))
    V = np.zeros((m*p, rank_total))
    W = np.zeros((p*n, rank_total))
    
    # Define block indices
    # A blocks (2x2 each):
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    # B blocks (2x3 each):
    # B00: 0,1,2, 4,5,6  (Rows 0,1; Cols 0,1,2) -> B[0,0]=0, B[0,1]=1, B[0,2]=2, B[1,0]=4, B[1,1]=5, B[1,2]=6
    # B01: 3,4,5, 7,8,9  (Rows 0,1; Cols 3,4,5) -> B[0,3]=3, B[0,4]=4, B[0,5]=5, B[1,3]=7, B[1,4]=8, B[1,5]=9
    # B10: 12,13,14, 16,17,18 (Rows 2,3; Cols 0,1,2) -> B[2,0]=12, B[2,1]=13, B[2,2]=14, B[3,0]=16, B[3,1]=17, B[3,2]=18
    # B11: 15,16,17, 19,20,21 (Rows 2,3; Cols 3,4,5) -> B[2,3]=15, B[2,4]=16, B[2,5]=17, B[3,3]=19, B[3,4]=20, B[3,5]=21
    
    # Wait, B is 4x6.
    # B[i,j] index = i*6 + j.
    # B00: i in {0,1}, j in {0,1,2}. Indices: 0,1,2, 6,7,8.
    # B01: i in {0,1}, j in {3,4,5}. Indices: 3,4,5, 9,10,11.
    # B10: i in {2,3}, j in {0,1,2}. Indices: 12,13,14, 18,19,20.
    # B11: i in {2,3}, j in {3,4,5}. Indices: 15,16,17, 21,22,23.
    
    # C blocks (2x3 each):
    # C is 4x6? No, C is 4x6?
    # Task: 4x4 * 4x6 -> 4x6.
    # C[i,k] index = k*4 + i (trace convention W[k*n+i]).
    # C00: i in {0,1}, k in {0,1,2}.
    #   k=0: i=0->0, i=1->1.
    #   k=1: i=0->4, i=1->5.
    #   k=2: i=0->8, i=1->9.
    #   Indices: 0,1,4,5,8,9.
    # C01: i in {0,1}, k in {3,4,5}.
    #   k=3: i=0->12, i=1->13.
    #   k=4: i=0->16, i=1->17.
    #   k=5: i=0->20, i=1->21.
    #   Indices: 12,13,16,17,20,21.
    # C10: i in {2,3}, k in {0,1,2}.
    #   k=0: i=2->2, i=3->3.
    #   k=1: i=2->6, i=3->7.
    #   k=2: i=2->10, i=3->11.
    #   Indices: 2,3,6,7,10,11.
    # C11: i in {2,3}, k in {3,4,5}.
    #   k=3: i=2->14, i=3->15.
    #   k=4: i=2->18, i=3->19.
    #   k=5: i=2->22, i=3->23.
    #   Indices: 14,15,18,19,22,23.

    # Define the 7 Strassen steps
    # Each step defines A' (2x2) and B' (2x3) as linear combos of blocks.
    # We represent A' entries as lists of (global_A_idx, coeff).
    # We represent B' entries as lists of (global_B_idx, coeff).
    
    steps = []
    
    # Helper to get A block indices
    def a_block_indices(block):
        if block == 0: return [0, 1, 4, 5]
        if block == 1: return [2, 3, 6, 7]
        if block == 2: return [8, 9, 12, 13]
        if block == 3: return [10, 11, 14, 15]
        
    # Helper to get B block indices
    def b_block_indices(block):
        if block == 0: return [0, 1, 2, 6, 7, 8]
        if block == 1: return [3, 4, 5, 9, 10, 11]
        if block == 2: return [12, 13, 14, 18, 19, 20]
        if block == 3: return [15, 16, 17, 21, 22, 23]
        
    # Helper to get C block indices
    def c_block_indices(block):
        if block == 0: return [0, 1, 4, 5, 8, 9]
        if block == 1: return [12, 13, 16, 17, 20, 21]
        if block == 2: return [2, 3, 6, 7, 10, 11]
        if block == 3: return [14, 15, 18, 19, 22, 23]

    # M1: (A00+A11)(B00+B11)
    # A' = A00+A11, B' = B00+B11
    # A' entries:
    # A'00 = A00[0]+A11[0], etc.
    # We define A_coeffs as 2x2 list of lists of (idx, coeff)
    # But it's easier to define the linear combination of the whole block.
    # Since U_blk is linear in A entries, we can just sum the U vectors for each contributing A entry.
    
    # Let's define the steps as (A_blocks, B_blocks, C_blocks, signs)
    # A_blocks: dict {block_idx: coeff}
    # B_blocks: dict {block_idx: coeff}
    # C_blocks: dict {block_idx: coeff}
    
    # M1
    steps.append({
        'A': {0: 1, 3: 1},
        'B': {0: 1, 3: 1},
        'C': {0: 1, 3: -1} # C00 += M1, C11 -= M1
    })
    
    # M2
    steps.append({
        'A': {2: 1, 3: 1},
        'B': {0: 1},
        'C': {2: 1, 3: -1} # C10 += M2, C11 -= M2
    })
    
    # M3
    steps.append({
        'A': {0: 1},
        'B': {1: 1, 3: -1},
        'C': {1: 1, 3: 1} # C01 += M3, C11 += M3
    })
    
    # M4
    steps.append({
        'A': {3: 1},
        'B': {2: 1, 0: -1},
        'C': {0: 1, 2: 1} # C00 += M4, C10 += M4
    })
    
    # M5
    steps.append({
        'A': {0: 1, 1: 1},
        'B': {3: 1},
        'C': {0: -1, 1: 1} # C00 -= M5, C01 += M5
    })
    
    # M6
    steps.append({
        'A': {2: 1, 0: -1},
        'B': {0: 1, 1: 1},
        'C': {3: 1} # C11 += M6
    })
    
    # M7
    steps.append({
        'A': {1: 1, 3: -1},
        'B': {2: 1, 3: 1},
        'C': {0: 1} # C00 += M7
    })
    
    col_offset = 0
    for step in steps:
        # For each term t in 0..10 of the block decomposition
        for t in range(11):
            # Construct U vector for this step and term
            # U_blk[:, t] has 4 entries corresponding to A'[0,0], A'[0,1], A'[1,0], A'[1,1]
            # We need to expand these into global A indices.
            # A'[r,c] is sum of coeff * A_block[block][r,c]
            # So U_global[i, col] = sum_{r,c} U_blk[r*2+c, t] * sum_{block} coeff_A[block] * (A_block[block][r,c] is at global index idx)
            
            # This is equivalent to:
            # For each block in step['A']:
            #   For each local index l in 0..3:
            #     global_idx = a_block_indices(block)[l]
            #     U_global[global_idx, col_offset] += U_blk[l, t] * coeff_A[block]
            
            for block, coeff in step['A'].items():
                idxs = a_block_indices(block)
                for l in range(4):
                    U[idxs[l], col_offset] += U_blk[l, t] * coeff
            
            # Similarly for V
            for block, coeff in step['B'].items():
                idxs = b_block_indices(block)
                for l in range(6):
                    V[idxs[l], col_offset] += V_blk[l, t] * coeff
            
            # Similarly for W
            for block, coeff in step['C'].items():
                idxs = c_block_indices(block)
                for l in range(6):
                    W[idxs[l], col_offset] += W_blk[l, t] * coeff
            
            col_offset += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
