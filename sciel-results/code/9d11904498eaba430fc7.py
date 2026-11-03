import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 130  # 20s margin before 150s kill
    
    # --- Part 1: Build schoolbook decomposition for the 4x4x3 slice ---
    # This handles columns k=4,5,6 of the B matrix
    # Rank contribution: n * m * 3 = 4 * 4 * 3 = 48
    rank_3 = n * m * 3
    U_3 = np.zeros((n * m, rank_3))
    V_3 = np.zeros((m * 3, rank_3))
    W_3 = np.zeros((3 * n, rank_3))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k_offset in range(3):
                k = k_offset  # k=0,1,2 corresponds to B columns 4,5,6
                U_3[i * m + j, col] = 1
                V_3[j * 3 + k_offset, col] = 1
                W_3[k_offset * n + i, col] = 1
                col += 1
    
    # --- Part 2: Strassen-based decomposition for the 4x4x4 slice ---
    # Using recursive Strassen: 4x4x4 = 7 * (2x2x2 blocks)
    # Each 2x2x2 block has rank 7 (Strassen)
    # Total rank for 4x4x4: 7 * 7 = 49... wait, let me reconsider.
    
    # Actually, Strassen for 4x4: we split into 2x2 blocks.
    # 4x4 * 4x4 = 7 mults of 2x2 * 2x2
    # Each 2x2*2x2 has rank 7
    # So total rank = 7 * 7 = 49
    
    # Let's build the 4x4x4 Strassen decomposition
    # First, build a 2x2x2 Strassen decomposition
    rank_222 = 7
    U_222 = np.zeros((4, rank_222))
    V_222 = np.zeros((4, rank_222))
    W_222 = np.zeros((4, rank_222))
    
    # Strassen formulas for 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B11 - B21)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # A indices: A11=0, A12=1, A21=2, A22=3
    # B indices: B11=0, B12=1, B21=2, B22=3
    # C indices: C11=0, C12=1, C21=2, C22=3
    
    U_222[0, 0] = 1  # A11
    U_222[3, 0] = 1  # A22
    V_222[0, 0] = 1  # B11
    V_222[3, 0] = 1  # B22
    W_222[0, 0] = 1  # C11
    W_222[3, 0] = 1  # C22
    
    U_222[2, 1] = 1  # A21
    U_222[3, 1] = 1  # A22
    V_222[0, 1] = 1  # B11
    W_222[2, 1] = 1  # C21
    W_222[3, 1] = 1  # C22
    
    U_222[0, 2] = 1  # A11
    V_222[2, 2] = 1  # B21
    W_222[2, 2] = 1  # C21
    V_222[3, 2] = -1  # -B22
    W_222[3, 2] = -1  # -C22
    
    U_222[3, 3] = 1  # A22
    V_222[0, 3] = 1  # B11
    W_222[0, 3] = 1  # C11
    V_222[2, 3] = -1  # -B21
    W_222[2, 3] = -1  # -C21
    
    U_222[0, 4] = 1  # A11
    U_222[1, 4] = 1  # A12
    V_222[3, 4] = 1  # B22
    W_222[1, 4] = 1  # C12
    W_222[3, 4] = 1  # C22
    
    U_222[2, 5] = 1  # A21
    U_222[0, 5] = -1  # -A11
    V_222[0, 5] = 1  # B11
    V_222[1, 5] = 1  # B12
    W_222[0, 5] = 1  # C11
    W_222[1, 5] = 1  # C12
    
    U_222[1, 6] = 1  # A12
    U_222[3, 6] = -1  # -A22
    V_222[2, 6] = 1  # B21
    V_222[3, 6] = 1  # B22
    W_222[1, 6] = 1  # C12
    W_222[2, 6] = 1  # C21
    
    # Now build 4x4x4 from 2x2x2 using Kronecker-like structure
    # 4x4 matrix split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = A*B
    
    # Using Strassen at the 2x2-block level:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B11 - B21)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 - M2
    
    # Each Mi is a 2x2 product, which we decompose with rank 7
    # So total rank = 7 * 7 = 49
    
    rank_444 = 49
    U_444 = np.zeros((n * m, rank_444))
    V_444 = np.zeros((m * 4, rank_444))
    W_444 = np.zeros((4 * n, rank_444))
    
    # Block indices for 4x4 matrix:
    # A blocks: A11 = rows 0-1, cols 0-1; A12 = rows 0-1, cols 2-3
    #           A21 = rows 2-3, cols 0-1; A22 = rows 2-3, cols 2-3
    # B blocks: similar
    # C blocks: similar
    
    # Define block offsets
    # For A (4x4): row_block * 2 + local_row, col_block * 2 + local_col
    # For B (4x4): same
    # For C (4x4): same
    
    # We need to build U_444, V_444, W_444 such that
    # For each of the 7 Strassen multipliers, we use the 2x2x2 decomposition
    
    # Let's index the 4x4 elements as: index = row * 4 + col
    # A: 16 elements, indices 0-15
    # B: 16 elements, indices 0-15
    # C: 16 elements, indices 0-15
    
    # Block structure:
    # A11: rows 0,1 cols 0,1 -> indices 0,1,4,5
    # A12: rows 0,1 cols 2,3 -> indices 2,3,6,7
    # A21: rows 2,3 cols 0,1 -> indices 8,9,12,13
    # A22: rows 2,3 cols 2,3 -> indices 10,11,14,15
    
    # For each Strassen multiplier s (0..6), we have 7 sub-multipliers (from 2x2x2)
    # Total: 7 * 7 = 49 terms
    
    # Let me define the Strassen multipliers and their block components
    # For each multiplier, I need to know which blocks of A and B are involved
    
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B11 - B21)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # For each multiplier, the A side is a sum of blocks, B side is a sum of blocks
    # Each block is 2x2, so we need to decompose each 2x2*2x2 product
    
    # Let's create a helper: given a list of (block_index, coefficient) for A side
    # and B side, and the W contribution, build the rank-7 decomposition
    
    # Block index to element indices:
    # Block 0 (11): rows 0,1 cols 0,1 -> [0,1,4,5]
    # Block 1 (12): rows 0,1 cols 2,3 -> [2,3,6,7]
    # Block 2 (21): rows 2,3 cols 0,1 -> [8,9,12,13]
    # Block 3 (22): rows 2,3 cols 2,3 -> [10,11,14,15]
    
    block_elems = [
        [0, 1, 4, 5],   # block 11
        [2, 3, 6, 7],   # block 12
        [8, 9, 12, 13], # block 21
        [10, 11, 14, 15] # block 22
    ]
    
    # For each Strassen multiplier, define:
    # a_blocks: list of (block_idx, coeff)
    # b_blocks: list of (block_idx, coeff)
    # w_blocks: list of (block_idx, coeff) for C
    
    strassen_defs = [
        # M1 = (A11 + A22)(B11 + B22) -> C11 += M1, C22 += M1
        {'a': [(0, 1), (3, 1)], 'b': [(0, 1), (3, 1)], 'w': [(0, 1), (3, 1)]},
        # M2 = (A21 + A22)B11 -> C21 += M2, C22 -= M2
        {'a': [(2, 1), (3, 1)], 'b': [(0, 1)], 'w': [(2, 1), (3, -1)]},
        # M3 = A11(B21 - B22) -> C12 += M3, C22 -= M3
        {'a': [(0, 1)], 'b': [(2, 1), (3, -1)], 'w': [(1, 1), (3, -1)]},
        # M4 = A22(B11 - B21) -> C11 += M4, C21 += M4
        {'a': [(3, 1)], 'b': [(0, 1), (2, -1)], 'w': [(0, 1), (2, 1)]},
        # M5 = (A11 + A12)B22 -> C12 += M5, C11 -= M5
        {'a': [(0, 1), (1, 1)], 'b': [(3, 1)], 'w': [(1, 1), (0, -1)]},
        # M6 = (A21 - A11)(B11 + B12) -> C22 += M6
        {'a': [(2, 1), (0, -1)], 'b': [(0, 1), (1, 1)], 'w': [(3, 1)]},
        # M7 = (A12 - A22)(B21 + B22) -> C11 -= M7, C21 += M7
        {'a': [(1, 1), (3, -1)], 'b': [(2, 1), (3, 1)], 'w': [(0, -1), (2, 1)]},
    ]
    
    # Now build the full 4x4x4 decomposition
    col = 0
    for s_def in strassen_defs:
        a_blocks = s_def['a']
        b_blocks = s_def['b']
        w_blocks = s_def['w']
        
        # For each of the 7 sub-terms in the 2x2x2 decomposition:
        for sub in range(rank_222):
            # Determine which elements of A, B, C are involved in this sub-term
            # U_222[sub_row, sub] gives coefficient for A element sub_row
            # V_222[sub_row, sub] gives coefficient for B element sub_row
            # W_222[sub_row, sub] gives coefficient for C element sub_row
            
            # For A side: for each (block_idx, coeff) in a_blocks,
            # and for each local_row where U_222[local_row, sub] != 0,
            # add coeff * U_222[local_row, sub] to U_444[element_index, col]
            
            for (a_block, a_coeff) in a_blocks:
                for local_row in range(4):
                    u_val = U_222[local_row, sub]
                    if u_val != 0:
                        elem_idx = block_elems[a_block][local_row]
                        U_444[elem_idx, col] += a_coeff * u_val
            
            # For B side
            for (b_block, b_coeff) in b_blocks:
                for local_row in range(4):
                    v_val = V_222[local_row, sub]
                    if v_val != 0:
                        elem_idx = block_elems[b_block][local_row]
                        V_444[elem_idx, col] += b_coeff * v_val
            
            # For C side
            for (w_block, w_coeff) in w_blocks:
                for local_row in range(4):
                    w_val = W_222[local_row, sub]
                    if w_val != 0:
                        elem_idx = block_elems[w_block][local_row]
                        W_444[elem_idx, col] += w_coeff * w_val
            
            col += 1
    
    # Verify rank_444
    assert col == rank_444, f"Expected {rank_444}, got {col}"
    
    # --- Combine Part 1 and Part 2 ---
    # Total rank = rank_3 + rank_444 = 48 + 49 = 97
    total_rank = rank_3 + rank_444
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Copy Part 1 (4x4x3 slice, k=0,1,2 corresponding to B columns 4,5,6)
    U[:, :rank_3] = U_3
    # V_3 is (m*3, rank_3), needs to go into rows 4*m to 7*m of V (since B has 4*7=28 cols, first 16 are for 4x4x4, next 12 are for 4x4x3)
    # Wait, let me reconsider the indexing.
    
    # A is 4x4 (n=4, m=4), so A has 16 elements, indices 0-15
    # B is 4x7 (m=4, p=7), so B has 28 elements, indices 0-27
    #   - B[0:4, 0:4] corresponds to elements 0-15 (for 4x4x4)
    #   - B[0:4, 4:7] corresponds to elements 16-27 (for 4x4x3)
    # C is 4x7 (p=7, n=4), so C has 28 elements, indices 0-27
    #   - C[0:4, 0:4] corresponds to elements 0-15 (for 4x4x4)
    #   - C[4:7, 0:4] corresponds to elements 16-27 (for 4x4x3)
    
    # Wait, the trace convention is W[k*n+i, r] where k is column index of C, i is row index.
    # So C element (i, k) has index k*n + i = k*4 + i
    # For C[0:4, 0:4]: indices 0*4+0=0, 0*4+1=1, 0*4+2=2, 0*4+3=3, 1*4+0=4, ..., 3*4+3=15
    # For C[0:4, 4:7]: indices 4*4+0=16, 4*4+1=17, ..., 6*4+3=27
    
    # Similarly for B: B element (j, k) has index j*p + k = j*7 + k
    # B[0:4, 0:4]: indices 0*7+0=0, 0*7+1=1, ..., 3*7+3=23? No, that's not right.
    
    # Wait, I need to be more careful. Let me re-read the problem.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by [k*n + i, r] where i is row of A/C, k is column of B/C.
    
    # U is (n*m, rank), indexed by [i*m + j, r] where i is row of A, j is column of A
    # V is (m*p, rank), indexed by [j*p + k, r] where j is column of A/row of B, k is column of B
    # W is (p*n, rank), indexed by [k*n + i, r] where k is column of B, i is row of A/C
    
    # So:
    # A[i,j] -> U[i*m+j, r] = U[i*4+j, r]
    # B[j,k] -> V[j*p+k, r] = V[j*7+k, r]
    # C[i,k] -> W[k*n+i, r] = W[k*4+i, r]
    
    # For the 4x4x4 part (k=0,1,2,3):
    # A[i,j] for i,j in 0..3 -> U[i*4+j, r]
    # B[j,k] for j in 0..3, k in 0..3 -> V[j*7+k, r]
    # C[i,k] for i in 0..3, k in 0..3 -> W[k*4+i, r]
    
    # For the 4x4x3 part (k=4,5,6):
    # A[i,j] for i,j in 0..3 -> U[i*4+j, r]
    # B[j,k] for j in 0..3, k in 4..6 -> V[j*7+k, r]
    # C[i,k] for i in 0..3, k in 4..6 -> W[k*4+i, r]
    
    # So my earlier U_444 and V_444, W_444 need to be adjusted.
    # U_444 is (16, 49), which is correct for U.
    # V_444 should be (m*4=16, 49) but V is (m*p=28, 97).
    # So V_444 goes into rows 0..15 of V (for k=0,1,2,3, j=0,1,2,3: j*7+k)
    #   j=0: k=0,1,2,3 -> rows 0,1,2,3
    #   j=1: k=0,1,2,3 -> rows 7,8,9,10
    #   j=2: k=0,1,2,3 -> rows 14,15,16,17
    #   j=3: k=0,1,2,3 -> rows 21,22,23,24
    
    # Hmm, this is getting complicated. Let me rebuild more carefully.
    
    # Let me restart the combination step.
    
    # First, let me verify the 4x4x4 decomposition is correct by checking
    # that it produces the right tensor.
    
    # Actually, let me just directly build the full U, V, W matrices.
    
    # Part 2: 4x4x4 Strassen decomposition
    # U_444: (16, 49) - already correct
    # V_444: needs to be (28, 49) but only rows corresponding to k=0,1,2,3 are non-zero
    # W_444: (16, 49) - already correct
    
    # Let me rebuild V_444 properly
    V_444_full = np.zeros((m * p, rank_444))
    # Copy V_444 (which is (16, 49) indexed by j*4+k for j,k in 0..3)
    # into V_444_full (which is (28, 49) indexed by j*7+k for j in 0..3, k in 0..3)
    for j in range(4):
        for k in range(4):
            V_444_full[j * p + k, :] = V_444[j * 4 + k, :]
    
    # Part 1: 4x4x3 schoolbook
    # U_3: (16, 48) - already correct
    # V_3: (12, 48) indexed by j*3+k_offset for j in 0..3, k_offset in 0..2
    #   needs to go into V at rows j*7+(4+k_offset) for j in 0..3, k_offset in 0..2
    V_3_full = np.zeros((m * p, rank_3))
    for j in range(4):
        for k_offset in range(3):
            V_3_full[j * p + (4 + k_offset), :] = V_3[j * 3 + k_offset, :]
    
    # W_3: (12, 48) indexed by k_offset*n+i for k_offset in 0..2, i in 0..3
    #   needs to go into W at rows (4+k_offset)*n+i for k_offset in 0..2, i in 0..3
    W_3_full = np.zeros((p * n, rank_3))
    for k_offset in range(3):
        for i in range(4):
            W_3_full[(4 + k_offset) * n + i, :] = W_3[k_offset * n + i, :]
    
    # Now combine
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Part 1 (4x4x3) goes in columns 0..47
    U[:, :rank_3] = U_3
    V[:, :rank_3] = V_3_full
    W[:, :rank_3] = W_3_full
    
    # Part 2 (4x4x4) goes in columns 48..96
    U[:, rank_3:] = U_444
    V[:, rank_3:] = V_444_full
    W[:, rank_3:] = W_444
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = total_rank
    
    # Verify the decomposition is correct
    if time.time() < deadline:
        # Check that sum_r U[:,r] * V[:,r] * W[:,r]^T gives the multiplication tensor
        # For each (i,j,k), the tensor element should be 1 if C[i,k] = A[i,j]*B[j,k]
        # i.e., T[i,j,k] = delta (it's 1 for the multiplication tensor)
        
        # Actually, the tensor T has T[i,j,k] = 1 for all valid (i,j,k) in the multiplication
        # The decomposition should satisfy: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Let's spot check a few entries
        max_err = 0
        for i in range(min(n, 2)):
            for j in range(min(m, 2)):
                for k in range(min(p, 2)):
                    expected = 1.0
                    actual = 0.0
                    for r in range(total_rank):
                        actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    max_err = max(max_err, abs(actual - expected))
        
        if max_err > 1e-6:
            # If verification fails, fall back to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
            
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
    
    return best_result
# EVOLVE-BLOCK-END
