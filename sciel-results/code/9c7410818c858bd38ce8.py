import time
import random
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = 4, 4, 7
    
    # Strassen 2x2 tensor decomposition (7 tensors)
    # Standard Strassen: for A, B in M_2, C = A*B
    # The 7 intermediate products are:
    # M0 = (A00+A11)*(B00+B11)
    # M1 = (A10+A11)*B00
    # M2 = A00*(B01-B11)
    # M3 = A11*(B10-B00)
    # M4 = (A00+A01)*B11
    # M5 = (A10-A00)*(B00+B01)
    # M6 = (A01-A11)*(B10+B11)
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    strassen_A = [
        [1, 1, 0, 0, 1, -1, 0],   # A00
        [0, 1, -1, 0, -1, 1, 0],   # A01
        [1, 1, 0, 1, 0, 1, -1],   # A10
        [1, 0, 0, 1, 1, 0, -1]    # A11
    ]
    strassen_B = [
        [1, 0, 1, -1, 0, 1, 0],   # B00
        [0, 0, 1, 0, 1, 1, 1],    # B01
        [1, 1, 0, 1, 0, 0, 1],    # B10
        [1, 0, -1, -1, 1, 0, 1]   # B11
    ]
    
    # Schoolbook 2x2x2 decomposition (4 tensors)
    school2_A = [[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]
    school2_B = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    
    # Build the full decomposition using block structure
    # We partition the 4x4x7 matrix multiplication into blocks.
    # A is 4x4, B is 4x7, C is 4x7.
    # We can partition A into 2x2 blocks of 2x2, B into 2x(4x7) blocks.
    
    # Strategy: use Strassen for the 4x4 part (2x2 blocks of 2x2) 
    # and schoolbook for the remaining columns.
    
    # Actually, let's think more carefully.
    # A (4x4) * B (4x7) = C (4x7)
    # Partition A = [[A00, A01],[A10, A11]] where each is 2x2
    # Partition B = [[B0],[B1]] where B0 is 2x7, B1 is 2x7
    # Then C0 = A00*B0 + A01*B1 (2x7)
    #       C1 = A10*B0 + A11*B1 (2x7)
    
    # We can use Strassen-style for combining the 2x2 with 2x7.
    # For fixed 2x2 * 2x7:
    # (A00+A11)*(B0+B1), (A10+A11)*B0, A00*(B1-B0), A11*(B0-B1)
    # That's 4 multiplications of 2x2 * 2x7... wait no.
    
    # Let me use a cleaner approach:
    # Partition A into four 2x2 blocks, and B into two 2x7 blocks (top and bottom).
    # C_top = A_top * B  (2x4 * 4x7 = 2x7)
    # C_bot = A_bot * B  (2x4 * 4x7 = 2x7)
    
    # For each 2x4 * 4x7 = 2x7:
    # Partition the 2x4 into two 2x2 blocks: [A0, A1]
    # Partition 4x7 into two 2x7 blocks: [B0; B1]
    # Result = A0*B0 + A1*B1 (2x7)
    # We can do this with schoolbook: 4 multiplications of 2x2*2x7... 
    # No, A0*B0 is 2x2 * 2x7, which has rank 2*2*7 = 28 by schoolbook.
    # Two such: 56 per 2x4*4x7. Total 112. Same as schoolbook.
    
    # Better: use Strassen on the 2x2 * 2x2 part.
    # For 2x2 * 2x2, Strassen uses 7 instead of 4.
    # So for 2x2 * 2x7 = 2x7:
    # = [2x2] * [2x7] where 2x7 = [col1, col2, ..., col7]
    # Each column is 2x1. Hmm, this gets complex.
    
    # Alternative clean approach:
    # Partition all three matrices into 2x2 blocks where possible.
    # A: 2x2 blocks of 2x2 (4 blocks)
    # B: partition rows into 2+2, cols into 4+3. So B = [B00, B01; B10, B11] 
    #   where B00, B01, B10 are 2x2 and B01_right, B11_right are 2x3.
    #   Actually: B is 4x7. Rows 0-3, Cols 0-6.
    #   B = [[B00(2x2), B01(2x3)],
    #        [B10(2x2), B11(2x3)]]
    # C: 4x7 = [[C00(2x2), C01(2x3)],
    #             [C10(2x2), C11(2x3)]]
    
    # C00 = A00*B00 + A01*B10 (2x2)
    # C01 = A00*B01 + A01*B11 (2x2 * 2x3 = 2x3)
    # C10 = A10*B00 + A11*B10 (2x2)
    # C11 = A10*B01 + A11*B11 (2x2 * 2x3 = 2x3)
    
    # For C00 and C10 (2x2 * 2x2): use Strassen-like
    # C00 + C10 = (A00+A10)*B00 ... no that's not right.
    
    # Let me just use a direct block-Stra­sen approach.
    # For the 4x4x4 subproblem (first 4 columns of B), use 2x2 block Strassen.
    # For the 4x4x3 subproblem (last 3 columns of B), use schoolbook.
    
    # 4x4x4 with Strassen: rank = 7 * 4 = 28? No.
    # 4x4x4: partition into 2x2 blocks. There are 4 block-multiplications of 2x2x2.
    # Each 2x2x2 can be done with Strassen rank 7.
    # Total: 7 * 7 = 49? Let me count more carefully.
    
    # 4x4 * 4x4 = 4x4
    # A = [[A00,A01],[A10,A11]], B = [[B00,B01],[B10,B11]], each 2x2
    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11
    
    # Using Strassen:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # Each Mi is 2x2 * 2x2, which by Strassen has rank 7.
    # Total rank = 7*7 = 49 for 4x4x4.
    
    # For 4x4x3 (last 3 cols): schoolbook = 4*4*3 = 48.
    # Total: 49 + 48 = 97.
    
    # Now I need to construct the U, V, W matrices for rank 97.
    
    # Let me build this systematically.
    # U has shape (16, 97), V has shape (28, 97), W has shape (28, 97).
    
    # Part 1: Strassen for 4x4x4 (columns 0-3 of B)
    # Part 2: Schoolbook for 4x4x3 (columns 4-6 of B)
    
    # Part 1: Strassen 4x4x4
    # The 7 Strassen multiplications, each 2x2x2 with rank 7.
    # For each of the 7 multiplications Mi = Xi * Yi where Xi, Yi are 2x2:
    #   We use the 7 elementary Strassen tensors.
    #   Total: 7 * 7 = 49 rank terms.
    
    # Strassen 2x2x2 tensors (each is (4,4,4) with one non-zero entry pattern)
    # sA[i,r], sB[j,r], sC[k,r] for r=0..6
    # Using standard Strassen:
    sA = [
        [1, 1, 0, 0, 1, 0, 0],  # A00
        [0, 0, 1, 0, 0, 1, 1],  # A01  -- wait, let me be careful
        [1, 1, 0, 1, 0, 1, 0],  # A10
        [1, 0, 0, 1, 1, 0, 1]   # A11
    ]
    
    # Actually, let me redefine carefully.
    # M0 = (A00+A11)(B00+B11): A factor [1,0,1,1], B factor [1,1,0,0]... 
    
    # Standard Strassen formulas:
    # M0: (A00+A11)(B00+B11)
    # M1: (A10+A11)B00
    # M2: A00(B01-B11)
    # M3: A11(B10-B00)
    # M4: (A00+A01)B11
    # M5: (A10-A00)(B00+B01)
    # M6: (A01-A11)(B10+B11)
    
    # A-coefficients for each M (indexed by [A00, A01, A10, A11]):
    sA_coefs = [
        [1, 0, 0, 1],  # M0: A00+A11
        [0, 0, 1, 1],  # M1: A10+A11
        [1, 0, 0, 0],  # M2: A00
        [0, 0, 0, 1],  # M3: A11
        [1, 1, 0, 0],  # M4: A00+A01
        [1, 0, 1, 0],  # M5: A10-A00
        [0, 1, 0, -1]  # M6: A01-A11
    ]
    
    # B-coefficients for each M (indexed by [B00, B01, B10, B11]):
    sB_coefs = [
        [1, 0, 0, 1],  # M0: B00+B11
        [1, 0, 0, 0],  # M1: B00
        [0, 1, 0, -1], # M2: B01-B11
        [0, 0, 1, -1], # M3: B10-B00
        [0, 0, 0, 1],  # M4: B11
        [1, 1, 0, 0],  # M5: B00+B01
        [0, 0, 1, 1]   # M6: B10+B11
    ]
    
    # C-coefficients for each M in the final sum:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    # C is 2x2, indexed by [C00, C01, C10, C11]:
    sC_coefs = [
        [1, 0, 0, 0],   # M0 contributes to C00
        [0, 0, 1, -1],  # M1: C10 += M1, C11 -= M1
        [0, 1, 0, 1],   # M2: C01 += M2, C11 += M2
        [1, 0, 1, 0],   # M3: C00 += M3, C10 += M3
        [-1, 1, 0, 0],  # M4: C00 -= M4, C01 += M4
        [0, 0, 0, 1],   # M5: C11 += M5
        [1, 0, 0, 0]    # M6: C00 += M6
    ]
    
    # For the 4x4x4 Strassen, we have 7 outer multiplications (the block-level Strassen)
    # and 7 inner Strassen multiplications for each 2x2x2.
    # Total 49 terms.
    
    # For the outer block index (which 2x2 sub-block of A and B), and inner Strassen index:
    # We need to map to the global indices.
    
    # A global indices: 0..15, where A[i,j] is at index i*4+j (i=0..3, j=0..3)
    # B global indices: 0..27, where B[j,k] is at index j*7+k (j=0..3, k=0..6)
    # C global indices: 0..27, where C[i,k] is at index k*4+i (k=0..6, i=0..3)
    #   Wait, the task says W[k*n+i, r] so C index is k*n+i = k*4+i.
    
    # For 4x4x4 part (k=0..3):
    # A block [a00,a01,a10,a11] corresponds to:
    #   a00: A[0:2, 0:2] -> indices 0,1,4,5
    #   a01: A[0:2, 2:4] -> indices 2,3,6,7
    #   a10: A[2:4, 0:2] -> indices 8,9,12,13
    #   a11: A[2:4, 2:4] -> indices 10,11,14,15
    
    # B block for 4x4x4: B[0:2, 0:2], B[0:2, 2:4], B[2:4, 0:2], B[2:4, 2:4]
    #   b00: B[0:2, 0:2] -> indices 0,1,7,8
    #   b01: B[0:2, 2:4] -> indices 2,3,9,10
    #   b10: B[2:4, 0:2] -> indices 14,15,21,22
    #   b11: B[2:4, 2:4] -> indices 16,17,23,24
    
    # C block for 4x4x4: C[0:2, 0:2], C[0:2, 2:4], C[2:4, 0:2], C[2:4, 2:4]
    #   C index k*4+i
    #   c00: C[0:2, 0:2] -> k=0,1; i=0,1 -> indices 0,1,4,5
    #   c01: C[0:2, 2:4] -> k=2,3; i=0,1 -> indices 8,9,12,13
    #   c10: C[2:4, 0:2] -> k=0,1; i=2,3 -> indices 2,3,6,7
    #   c11: C[2:4, 2:4] -> k=2,3; i=2,3 -> indices 10,11,14,15
    
    # A_block_indices: list of 4 lists, each with 4 indices
    A_blocks = [
        [0, 1, 4, 5],    # A00
        [2, 3, 6, 7],    # A01
        [8, 9, 12, 13],  # A10
        [10, 11, 14, 15] # A11
    ]
    
    # B_blocks for 4x4x4 (cols 0-3):
    B_blocks_4 = [
        [0, 1, 7, 8],    # B00
        [2, 3, 9, 10],   # B01
        [14, 15, 21, 22],# B10
        [16, 17, 23, 24] # B11
    ]
    
    # C_blocks for 4x4x4:
    C_blocks_4 = [
        [0, 1, 4, 5],    # C00
        [8, 9, 12, 13],  # C01
        [2, 3, 6, 7],    # C10
        [10, 11, 14, 15] # C11
    ]
    
    # Build Part 1 (Strassen 4x4x4, rank 49)
    U1 = np.zeros((16, 49))
    V1 = np.zeros((28, 49))
    W1 = np.zeros((28, 49))
    
    # For outer Strassen term t (0..6) and inner Strassen term s (0..6):
    # The outer term t uses A-coef sA_coefs[t] and B-coef sB_coefs[t]
    # The inner term s uses A-coef sA_coefs[s] (wait, no)
    
    # Actually, the outer Strassen combines blocks. Let me re-derive.
    # For the 4x4x4 Strassen:
    # We have 7 "multiplications" at the block level.
    # Each block-level multiplication is X_block * Y_block where X is 2x2 and Y is 2x2.
    # Each such 2x2x2 multiplication is decomposed using inner Strassen (rank 7).
    
    # Outer t=0: X = A00+A11 (as a 2x2), Y = B00+B11 (as a 2x2)
    # The 2x2 X has entries: X[0,0]=A[0,0]+A[2,2], X[0,1]=A[0,1]+A[2,3], X[1,0]=A[1,0]+A[3,2], X[1,1]=A[1,1]+A[3,3]
    # So for outer t=0, inner s: 
    #   The A-factor for inner s applies to X's 2x2 entries.
    #   X[0,0] = A00[0,0] + A11[0,0] = A[0,0] + A[2,2]
    #   etc.
    
    # This is getting complicated. Let me just build it directly by tracking which global indices get which coefficients.
    
    # For outer term t, the A-block coefficient vector is sA_coefs[t] = [a00, a01, a10, a11]
    # This means the X 2x2 matrix is:
    #   X[0,0] = a00*A[0,0] + a01*A[0,2] + a10*A[1,0] + a11*A[1,2] ... no wait.
    
    # I think I'm overcomplicating this. Let me use a cleaner formulation.
    
    # For 4x4x4 Strassen, the decomposition is:
    # For each pair (t, s) where t is outer (0..6) and s is inner (0..6):
    #   The contribution to A-index (i,j) is:
    #     outer_A_coef[t][block_of(i,j)] * inner_A_coef[s][local_ij]
    #   where block_of maps (i,j) to {0,1,2,3} (the 2x2 block), and local_ij is within that block.
    
    # Similarly for B and C.
    
    # Let me define the mappings:
    # A: (i,j) -> block = i//2*2 + j//2, local_i = i%2, local_j = j%2
    # A index = i*4+j
    # B: (j,k) for k=0..3 -> block = j//2*2 + k//2, local_j = j%2, local_k = k%2
    # B index = j*7+k
    # C: (i,k) for k=0..3 -> block = k//2*2 + i//2, local_k = k%2, local_i = i%2
    # C index = k*4+i
    
    # A local index within block: local_i*2 + local_j (0,1,2,3)
    # This corresponds to A_block[local_i, local_j]
    
    # sA_coefs[t] gives coefficients for the 4 blocks of the outer A matrix.
    # But actually sA_coefs[t][block] is the coefficient for that entire 2x2 block.
    # Then within that block, the inner Strassen sA_coefs[s][local] gives the coefficient for each local entry.
    
    # So for A-index (i,j) = i*4+j:
    #   block = (i//2)*2 + (j//2)
    #   local = (i%2)*2 + (j%2)
    #   coef = sA_coefs[t][block] * sA_coefs[s][local]
    
    # Wait, that's not quite right either. The outer Strassen forms new 2x2 matrices.
    # For outer t, the X matrix (2x2) has X[bi, bj] = sum_block sA_coefs[t][block] * A_block[block][bi, bj]
    # where bi,bj are local indices within the 2x2.
    # Then inner Strassen applies to this 2x2 X.
    # So X[bi,bj] * Y[ci,cj] where the product contributes to C.
    
    # For the inner Strassen term s:
    #   The A-factor picks X[bi,bj] with coefficient sA_coefs[s][local] where local = bi*2+bj
    #   Similarly for B-factor: sB_coefs[s][local] for Y
    #   C-factor: sC_coefs[s][clocal] for the result block
    
    # So for global A-index (i,j):
    #   block_a = (i//2)*2 + (j//2)
    #   la_i = i%2, la_j = j%2
    #   local_a = la_i*2 + la_j
    #   coef_U = sA_coefs[t][block_a] * sA_coefs[s][local_a]
    
    # For global B-index (j,k) with k in 0..3:
    #   block_b = (j//2)*2 + (k//2)
    #   lb_j = j%2, lb_k = k%2
    #   local_b = lb_j*2 + lb_k
    #   coef_V = sB_coefs[t][block_b] * sB_coefs[s][local_b]
    
    # For global C-index (i,k) with k in 0..3:
    #   block_c = (k//2)*2 + (i//2)
    #   lc_k = k%2, lc_i = i%2
    #   local_c = lc_k*2 + lc_i
    #   coef_W = sC_coefs[t][block_c] * sC_coefs[s][local_c]
    
    # Let's build this:
    for t in range(7):
        for s in range(7):
            r = t * 7 + s
            for i in range(4):
                for j in range(4):
                    block_a = (i//2)*2 + (j//2)
                    local_a = (i%2)*2 + (j%2)
                    coef = sA_coefs[t][block_a] * sA_coefs[s][local_a]
                    U1[i*4+j, r] = coef
            
            for j in range(4):
                for k in range(4):
                    block_b = (j//2)*2 + (k//2)
                    local_b = (j%2)*2 + (k%2)
                    coef = sB_coefs[t][block_b] * sB_coefs[s][local_b]
                    V1[j*7+k, r] = coef
            
            for i in range(4):
                for k in range(4):
                    block_c = (k//2)*2 + (i//2)
                    local_c = (k%2)*2 + (i%2)
                    coef = sC_coefs[t][block_c] * sC_coefs[s][local_c]
                    W1[k*4+i, r] = coef
    
    # Part 2: Schoolbook for 4x4x3 (columns 4-6 of B)
    U2 = np.zeros((16, 48))
    V2 = np.zeros((28, 48))
    W2 = np.zeros((28, 48))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):  # k=0,1,2 correspond to cols 4,5,6
                U2[i*4+j, col] = 1
                V2[j*7+(k+4), col] = 1
                W2[(k+4)*4+i, col] = 1
                col += 1
    
    # Combine
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    
    # Verify
    T = np.zeros((16, 28, 28))
    for i in range(4):
        for j in range(4):
            for k in range(7):
                T[i*4+j, j*7+k, k*4+i] = 1
    
    T_approx = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    if not np.allclose(T_approx, T, atol=1e-9):
        # Fallback to schoolbook
        U = np.zeros((16, 112))
        V = np.zeros((28, 112))
        W = np.zeros((28, 112))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    U[i*4+j, col] = 1
                    V[j*7+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
