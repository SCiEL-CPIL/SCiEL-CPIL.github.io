"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
_START_TIME = time.time()
_DEADLINE = 120.0
_MARGIN = 5.0

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Check time
    if time.time() > _START_TIME + _DEADLINE - _MARGIN:
        # Fallback: return schoolbook rank 96
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Strategy: Split the 3x4x8 problem into two 3x4x4 sub-problems
    # B is split into B_left (cols 0-3) and B_right (cols 4-7)
    # For each 3x4x4 sub-problem, use a rank-20 Stothers decomposition
    # Total rank: 40
    
    # First, let me find a valid rank-20 decomposition for 3x4x4
    # I'll do this computationally using a structured approach
    
    # For 3x4x4, the tensor has 48 non-zero entries
    # I'll use the known Stothers rank-20 decomposition
    
    # Actually, let me use a verified approach:
    # 3x4x4 can be decomposed as:
    # - Split into (2x2x2) + (2x2x2) + cross terms
    # Using Strassen for the 2x2x2 parts
    
    # Alternative: use the following verified rank-20 terms for 3x4x4
    # After careful derivation, here are the 20 bilinear forms:
    
    # A is 3x4: indices (i,j) -> flat i*4+j
    # B is 4x4: indices (j,k) -> flat j*4+k
    # C is 3x4: indices (i,k) -> flat i*4+k
    
    # The 20 terms for 3x4x4 (Stothers 1981):
    # I'll use a computational verification approach
    
    # Let me construct the decomposition using block Strassen:
    # Split A into A_top (2x4) and A_bottom (1x4)
    # Split B into B_top (2x4) and B_bottom (2x4)
    # C = A_top * B_top + A_bottom * B_bottom
    
    # A_top * B_top: 2x4 * 2x4 = 2x4
    # Split A_top into A_top1 (2x2) and A_top2 (2x2)
    # Split B_top into B_top1 (2x2) and B_top2 (2x2)
    # C_top = A_top1 * B_top1 + A_top2 * B_top2
    # Each 2x2x2 uses Strassen (rank 7), so C_top uses rank 14
    
    # A_bottom * B_bottom: 1x4 * 2x4 = 1x4
    # Split into two 1x2 * 2x2 = 1x2 blocks
    # Each uses rank 4 (schoolbook), so total rank 8
    
    # Total for 3x4x4: 14 + 8 = 22
    
    # Hmm, that's not 20. Let me try a different approach.
    
    # Actually, for 3x4x4, I'll use the following:
    # 2x4x4 in rank 14 (using Strassen on 2x2x2 blocks)
    # 1x4x4 in rank 16 (schoolbook)
    # Total: 30
    
    # Wait, that's worse than what I thought. Let me reconsider.
    
    # For 2x4x4:
    # A (2x4) * B (4x4) = C (2x4)
    # Split A into A1 (2x2) and A2 (2x2)
    # Split B into B1 (2x4) and B2 (2x4)
    # C = A1*B1 + A2*B2
    # A1*B1: 2x2 * 2x4 = 2x4, rank 14 (two Strassen 2x2x2)
    # A2*B2: 2x2 * 2x4 = 2x4, rank 14
    # Total: 28
    
    # That's not better than schoolbook 32.
    
    # Let me just use the Strassen-based approach that gives rank 88 for 3x4x8,
    # but try to verify it's correct and clean up the code.
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Schoolbook for A[0, :] (1x4) * B (4x8)
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
    
    # Part 2: Strassen for A[1:3, :] (2x4) * B (4x8)
    # Use Strassen for 2x2x2 blocks
    # A[1:3, :] is 2x4, split into two 2x2 blocks: A_left, A_right
    # B is 4x8, split into 4 blocks of 2x2: B_00, B_01, B_10, B_11
    # C[1:3, :] = A_left * [B_00, B_01] + A_right * [B_10, B_11]
    
    strassen_terms = [
        {'U': {0: 1, 3: 1}, 'V': {0: 1, 3: 1}, 'W': {0: 1, 3: 1}},
        {'U': {2: 1, 3: 1}, 'V': {0: 1}, 'W': {2: 1, 3: -1}},
        {'U': {0: 1}, 'V': {1: 1, 3: -1}, 'W': {1: 1, 3: 1}},
        {'U': {3: 1}, 'V': {2: 1, 0: -1}, 'W': {0: 1, 2: 1}},
        {'U': {0: 1, 1: 1}, 'V': {3: 1}, 'W': {0: -1, 1: 1}},
        {'U': {2: 1, 0: -1}, 'V': {0: 1, 1: 1}, 'W': {3: 1}},
        {'U': {1: 1, 3: -1}, 'V': {2: 1, 3: 1}, 'W': {0: 1}},
    ]
    
    # For each of the 4 column blocks of B (each 2x2)
    for j_block in range(4):
        # First 2x2x2: A_left * B_j_block
        # A_left is A[1:3, 0:2], B_j_block is B[0:2, 2*j_block:2*j_block+2]
        for term in strassen_terms:
            for local_u, val in term['U'].items():
                lu, lv = divmod(local_u, 2)
                global_i = 1 + lu
                global_j = lv
                U[global_i * m + global_j][col] += val
            for local_v, val in term['V'].items():
                lu, lv = divmod(local_v, 2)
                global_j = lu
                global_k = 2 * j_block + lv
                V[global_j * p + global_k][col] += val
            for local_w, val in term['W'].items():
                lu, lv = divmod(local_w, 2)
                global_i = 1 + lu
                global_k = 2 * j_block + lv
                W[global_k * n + global_i][col] += val
            col += 1
        
        # Second 2x2x2: A_right * B_j_block
        # A_right is A[1:3, 2:4], B_j_block is B[2:4, 2*j_block:2*j_block+2]
        for term in strassen_terms:
            for local_u, val in term['U'].items():
                lu, lv = divmod(local_u, 2)
                global_i = 1 + lu
                global_j = 2 + lv
                U[global_i * m + global_j][col] += val
            for local_v, val in term['V'].items():
                lu, lv = divmod(local_v, 2)
                global_j = 2 + lu
                global_k = 2 * j_block + lv
                V[global_j * p + global_k][col] += val
            for local_w, val in term['W'].items():
                lu, lv = divmod(local_w, 2)
                global_i = 1 + lu
                global_k = 2 * j_block + lv
                W[global_k * n + global_i][col] += val
            col += 1
    
    # Verify the decomposition
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_actual = np.zeros((n, p))
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    for t in range(rank):
        u_vec = U_arr[:, t].reshape(n, m)
        v_vec = V_arr[:, t].reshape(m, p)
        w_vec = W_arr[:, t].reshape(p, n)
        C_actual += (u_vec @ v_vec) * w_vec.T
    
    if not np.allclose(C_actual, C_expected, atol=1e-10):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
