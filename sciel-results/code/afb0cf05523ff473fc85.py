"""Rewritten program for matrix multiplication 4x5 by 5x5 (tensor decomposition).

Strategy: Use Strassen's algorithm for the 4x4x4 core (rank 49) plus
schoolbook for the 4x5x1 remainder (rank 20), giving total rank 69.

The 4x5 matrix A is split as [A_left (4x4) | A_right (4x1)].
The 5x5 matrix B is split as [B_top (4x4) | B_right (4x1); B_bottom (1x4) | B_bottom_right (1x1)].

C = A * B = A_left * B_top + A_left * B_right + A_right * B_bottom + A_right * B_bottom_right

Wait, let me be more careful. A is 4x5, B is 5x5, C is 4x5.

A = [A_11 (4x4) | A_12 (4x1)]
B = [B_11 (4x4) | B_12 (4x1)]
    [B_21 (1x4) | B_22 (1x1)]

C = A * B:
C[i,k] for i in 0..3, k in 0..4:
  For k in 0..3: C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k] + A[i,4]*B[4,k]
  For k=4: C[i,4] = sum_{j=0}^{3} A[i,j]*B[j,4] + A[i,4]*B[4,4]

So:
- The 4x4x4 part (A_11 * B_11) contributes to C[0:4, 0:4] with rank 49 (Strassen)
- A_11 * B_12 (4x4 times 4x1) contributes to C[0:4, 4] with rank 16 (schoolbook)
- A_12 * B_21 (4x1 times 1x4) contributes to C[0:4, 0:4] with rank 4 (schoolbook)
- A_12 * B_22 (4x1 times 1x1) contributes to C[0:4, 4] with rank 4 (schoolbook)

Total rank = 49 + 16 + 4 + 4 = 73.

Hmm, that's 73, not 69. Let me reconsider.

Actually, the previous response claimed 49 + 20 = 69. Let me think about what "4x5x1" means.

If we treat the problem as 4x5 * 5x5, we can split B as [B_left (5x4) | B_right (5x1)].
Then C_left (4x4) = A * B_left and C_right (4x1) = A * B_right.

For C_left: A (4x5) * B_left (5x4). This is a 4x5x4 problem.
For C_right: A (4x5) * B_right (5x1). This is a 4x5x1 problem.

The 4x5x1 problem has rank 4*5*1 = 20 (schoolbook).
The 4x5x4 problem: we can further split A as [A_left (4x4) | A_right (4x1)] and B_left as [B_top (4x4); B_bottom (1x4)].
C_left = A_left * B_top + A_right * B_bottom.
- A_left * B_top is 4x4x4, rank 49 (Strassen)
- A_right * B_bottom is 4x1 * 1x4, rank 4 (schoolbook)
Total for C_left: 49 + 4 = 53.

Grand total: 53 + 20 = 73.

So rank 73. Let me verify this is correct and implement it.

Actually wait - can we do better? Let me think about whether Strassen can be applied more cleverly.

For now, let me implement the rank-73 solution and verify it.
"""
import time
import random

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return the Strassen rank-7 decomposition for 2x2 * 2x2.
    
    Returns (U, V, W) where:
    U: 4x7 (for A entries, indexed as A[i,j] -> U[i*2+j])
    V: 4x7 (for B entries, indexed as B[j,k] -> V[j*2+k])  
    W: 4x7 (for C entries, indexed as C[i,k] -> W[k*2+i])
    
    Strassen's 7 multiplications for 2x2:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # A indices: A[0,0]=A11, A[0,1]=A12, A[1,0]=A21, A[1,1]=A22
    # B indices: B[0,0]=B11, B[0,1]=B12, B[1,0]=B21, B[1,1]=B22
    # C indices: C[0,0]=C11, C[0,1]=C12, C[1,0]=C21, C[1,1]=C22
    # W indexed as W[k*2+i] where C[i,k] -> W[k*2+i]
    
    # M1 = (A11+A22)(B11+B22)
    U[0][0] = 1  # A11
    U[3][0] = 1  # A22
    V[0][0] = 1  # B11
    V[3][0] = 1  # B22
    W[0][0] = 1  # C11
    W[3][0] = 1  # C22
    
    # M2 = (A21+A22)B11
    U[2][1] = 1  # A21
    U[3][1] = 1  # A22
    V[0][1] = 1  # B11
    W[2][1] = 1  # C21
    
    # M3 = A11(B21-B22)
    U[0][2] = 1  # A11
    V[2][2] = 1  # B21
    V[3][2] = -1 # B22
    W[1][2] = 1  # C12
    W[3][2] = 1  # C22
    
    # M4 = A22(B12-B11)
    U[3][3] = 1  # A22
    V[1][3] = 1  # B12
    V[0][3] = -1 # B11
    W[0][3] = 1  # C11
    W[2][3] = 1  # C21
    
    # M5 = (A11+A12)B22
    U[0][4] = 1  # A11
    U[1][4] = 1  # A12
    V[3][4] = 1  # B22
    W[0][4] = -1 # C11
    W[1][4] = 1  # C12
    
    # M6 = (A21-A11)(B11+B12)
    U[2][5] = 1  # A21
    U[0][5] = -1 # A11
    V[0][5] = 1  # B11
    V[1][5] = 1  # B12
    W[3][5] = 1  # C22
    
    # M7 = (A12-A22)(B21+B22)
    U[1][6] = 1  # A12
    U[3][6] = -1 # A22
    V[2][6] = 1  # B21
    V[3][6] = 1  # B22
    W[0][6] = 1  # C11
    W[3][6] = 1  # C22
    
    return U, V, W


def _strassen_4x4():
    """Return the Strassen rank-49 decomposition for 4x4 * 4x4.
    
    Applies Strassen recursively: split 4x4 into four 2x2 blocks.
    Each 2x2*2x2 uses rank 7, so 7*7 = 49 total.
    
    Returns (U, V, W) where:
    U: 16x49 (for A entries, indexed as A[i,j] -> U[i*4+j])
    V: 16x49 (for B entries, indexed as B[j,k] -> V[j*4+k])
    W: 16x49 (for C entries, indexed as C[i,k] -> W[k*4+i])
    """
    U2, V2, W2 = _strassen_2x2()
    
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    # The 4x4 matrix A is split into 2x2 blocks:
    # A = [A00 A01]
    #     [A10 A11]
    # where A00 is rows 0-1, cols 0-1; A01 is rows 0-1, cols 2-3; etc.
    
    # Similarly for B and C.
    
    # For each of the 7 Strassen multiplications at the top level,
    # we need to form the 2x2 matrices from the 4x4 blocks,
    # then apply the 2x2 Strassen decomposition.
    
    # Top-level Strassen for 4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mk is a 2x2 matrix computed via Strassen (rank 7).
    # So each Mk contributes 7 columns to the total rank-49 decomposition.
    
    # For each top-level multiplication Mk, we have a 2x2 result.
    # The 2x2 Strassen gives us 7 sub-multiplications.
    # For each sub-multiplication, the U, V, W factors are 2x2 matrices
    # formed from the appropriate 4x4 blocks.
    
    # Let's define the block mapping:
    # A00: A[0:2, 0:2] -> U indices 0-3 (i*4+j for i in 0-1, j in 0-1)
    # A01: A[0:2, 2:4] -> U indices 2,3,6,7 (i*4+j for i in 0-1, j in 2-3)
    # A10: A[2:4, 0:2] -> U indices 8,9,12,13 (i*4+j for i in 2-3, j in 0-1)
    # A11: A[2:4, 2:4] -> U indices 10,11,14,15 (i*4+j for i in 2-3, j in 2-3)
    
    # B00: B[0:2, 0:2] -> V indices 0-3
    # B01: B[0:2, 2:4] -> V indices 2,3,6,7
    # B10: B[2:4, 0:2] -> V indices 8,9,12,13
    # B11: B[2:4, 2:4] -> V indices 10,11,14,15
    
    # C00: C[0:2, 0:2] -> W indices 0-3 (k*4+i for k in 0-1, i in 0-1)
    # C01: C[0:2, 2:4] -> W indices 2,3,6,7
    # C10: C[2:4, 0:2] -> W indices 8,9,12,13
    # C11: C[2:4, 2:4] -> W indices 10,11,14,15
    
    def get_block_indices(block_type, size=2):
        """Get the linear indices for a 2x2 block within a 4x4 matrix.
        
        block_type: '00', '01', '10', '11'
        Returns list of 4 indices (row-major within the block).
        """
        if block_type == '00':
            return [0, 1, 4, 5]  # rows 0-1, cols 0-1: 0*4+0, 0*4+1, 1*4+0, 1*4+1
        elif block_type == '01':
            return [2, 3, 6, 7]  # rows 0-1, cols 2-3
        elif block_type == '10':
            return [8, 9, 12, 13]  # rows 2-3, cols 0-1
        elif block_type == '11':
            return [10, 11, 14, 15]  # rows 2-3, cols 2-3
    
    # For the 2x2 Strassen, the local indices are:
    # A_local: [0,1,2,3] corresponding to A11,A12,A21,A22
    # B_local: [0,1,2,3] corresponding to B11,B12,B21,B22
    # C_local: [0,1,2,3] corresponding to C11,C12,C21,C22
    # But in our 2x2 Strassen function, W is indexed as W[k*2+i].
    # So W[0]=C11, W[1]=C12, W[2]=C21, W[3]=C22.
    
    # Now for each top-level multiplication, we need to map the 2x2 blocks
    # to the 4x4 indices and apply the 2x2 Strassen.
    
    # Top-level multiplication 1: (A00+A11)(B00+B11)
    # The 2x2 matrix for A is A00+A11, for B is B00+B11.
    # In the 2x2 Strassen, U[local_idx] gets contributions from the A blocks.
    
    # Let me define a helper: for a given top-level mult, specify which 4x4 blocks
    # form the 2x2 A and B matrices, and which 4x4 block the 2x2 C goes into.
    
    # Top-level mults:
    # M1: A = A00+A11, B = B00+B11, C -> C00 and C11
    # M2: A = A10+A11, B = B00, C -> C10
    # M3: A = A00, B = B10-B11, C -> C01 and C11
    # M4: A = A11, B = B01-B00, C -> C00 and C10
    # M5: A = A00+A01, B = B11, C -> C00 and C01
    # M6: A = A10-A00, B = B00+B01, C -> C11
    # M7: A = A01-A11, B = B10+B11, C -> C00 and C11
    
    # For each top-level mult, the 2x2 A matrix is formed from two 2x2 blocks of the 4x4.
    # The 2x2 Strassen will have U factors that are 2x2 matrices.
    # Each entry in the 2x2 U corresponds to a coefficient times a 4x4 block entry.
    
    # This is getting complex. Let me use a different approach:
    # Build the full 4x4x49 decomposition by composing the 2x2 decompositions.
    
    # Alternative: directly construct the 49-column decomposition.
    
    # For each of the 7 top-level Strassen mults, and for each of the 7 sub-Strassen mults,
    # we get one column in the rank-49 decomposition.
    
    # Top-level mult t (0..6), sub-mult s (0..6): column index = t*7 + s
    
    # For top-level mult t, we have:
    # - A_blocks: list of (block_name, coefficient) pairs forming the 2x2 A matrix
    # - B_blocks: list of (block_name, coefficient) pairs forming the 2x2 B matrix  
    # - C_blocks: list of (block_name, coefficient) pairs for the 2x2 C matrix
    
    # The 2x2 A matrix has entries A11, A12, A21, A22 (local 2x2 indices).
    # Each local entry is a linear combination of 4x4 block entries.
    
    # For the 2x2 Strassen sub-decomposition, U2[local_A_idx, s] gives the coefficient
    # for that local A entry in sub-mult s.
    
    # So for the full 4x4 decomposition:
    # U[global_A_idx, t*7+s] = sum over local_A_idx of U2[local_A_idx, s] * coeff(global_A_idx, local_A_idx, t)
    
    # Where coeff(global_A_idx, local_A_idx, t) is the coefficient of the global entry
    # in the local entry for top-level mult t.
    
    # Let me define the top-level mults more carefully.
    
    # For top-level mult t, the 2x2 A matrix entries (local) are:
    # local_A[0] = A11 (top-left of the 2x2)
    # local_A[1] = A12 (top-right)
    # local_A[2] = A21 (bottom-left)
    # local_A[3] = A22 (bottom-right)
    
    # Each local_A[i] is a sum of 4x4 block entries with coefficients.
    
    # Top-level mult definitions:
    # M1 (t=0): A = A00+A11, B = B00+B11, C = C00+C11 (with signs from Strassen)
    #   local_A[0] = A00[0,0] + A11[0,0]  (but these are 2x2 blocks, not scalars)
    
    # Wait, I'm confusing myself. Let me restart with a cleaner formulation.
    
    # The 4x4 matrix A is partitioned into four 2x2 blocks:
    # A = [A00 A01]
    #     [A10 A11]
    # where A00 = A[0:2,0:2], A01 = A[0:2,2:4], A10 = A[2:4,0:2], A11 = A[2:4,2:4]
    
    # The 4x4 matrix B is similarly partitioned:
    # B = [B00 B01]
    #     [B10 B11]
    
    # The 4x4 matrix C = A*B is partitioned as:
    # C = [C00 C01]
    #     [C10 C11]
    
    # Strassen for 4x4 (treating blocks as 2x2 matrices):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mk is a 2x2 matrix product. Using Strassen for each 2x2 product gives rank 7.
    # Total rank: 7*7 = 49.
    
    # For each top-level mult Mk, the 2x2 A-matrix and 2x2 B-matrix are formed from
    # the 4x4 blocks. The 2x2 Strassen decomposition gives 7 sub-multiplications.
    
    # For sub-mult s of top-level mult t:
    # The 2x2 A-matrix for this sub-mult is determined by U2[:, s] applied to the
    # 2x2 A-matrix of top-level mult t.
    # Similarly for B and C.
    
    # Let me define for each top-level mult t:
    # A_coeffs[t][local_idx] = list of (global_block, global_pos, coeff)
    # where local_idx is 0,1,2,3 (the 2x2 local positions)
    
    # Actually, let me just directly compute the 49-column decomposition.
    
    # For top-level mult t and sub-mult s:
    # Column index = t*7 + s
    
    # The 2x2 A-matrix for top-level mult t has entries that are 2x2 blocks of the 4x4.
    # Let A_t be the 2x2 "matrix of blocks" for top-level mult t.
    # A_t[i,j] is a 2x2 matrix (block) for i,j in 0,1.
    
    # The 2x2 Strassen U2[i,j,s] gives the coefficient for A_t[i,j] in sub-mult s.
    # Wait, no. U2 is indexed by the local 2x2 A entries.
    
    # U2[local_A_idx, s] where local_A_idx encodes (i,j) of the 2x2 A matrix.
    # In our _strassen_2x2, U is 4x7, indexed as U[i*2+j] for i,j in 0,1.
    # So U[0]=A11, U[1]=A12, U[2]=A21, U[3]=A22.
    
    # For top-level mult t, the 2x2 A-matrix A_t has:
    # A_t[0,0] = some 2x2 block (or sum of blocks)
    # A_t[0,1] = some 2x2 block (or sum of blocks)
    # A_t[1,0] = some 2x2 block (or sum of blocks)
    # A_t[1,1] = some 2x2 block (or sum of blocks)
    
    # For sub-mult s, the contribution to U is:
    # For each local position (i,j) in the 2x2 A-matrix:
    #   U2[i*2+j, s] * A_t[i,j]
    # where A_t[i,j] is a 2x2 block of the 4x4.
    
    # So for each entry in the 2x2 block A_t[i,j], we add U2[i*2+j, s] * coeff to the
    # corresponding global U entry.
    
    # Let me define the block structure for each top-level mult.
    
    # I'll represent each 2x2 block as a 4-element list [top-left, top-right, bottom-left, bottom-right]
    # corresponding to positions (0,0), (0,1), (1,0), (1,1) within the 2x2 block.
    
    # Global 4x4 indices for block (bi, bj) where bi, bj in 0,1:
    # Block (0,0): rows 0-1, cols 0-1 -> global indices 0,1,4,5
    # Block (0,1): rows 0-1, cols 2-3 -> global indices 2,3,6,7
    # Block (1,0): rows 2-3, cols 0-1 -> global indices 8,9,12,13
    # Block (1,1): rows 2-3, cols 2-3 -> global indices 10,11,14,15
    
    block_indices = {
        (0,0): [0, 1, 4, 5],
        (0,1): [2, 3, 6, 7],
        (1,0): [8, 9, 12, 13],
        (1,1): [10, 11, 14, 15]
    }
    
    # For each top-level mult t, define A_t as a 2x2 array of 2x2 blocks.
    # Each block is represented as a list of 4 coefficients (for the 4 positions in the 2x2 block).
    # A_t[i][j] is the 2x2 block at position (i,j) of the 2x2 "matrix of blocks".
    
    # M1: A = A00+A11
    # A_t[0][0] = A00, A_t[0][1] = A00 (wait, no)
    
    # Actually, A00+A11 means:
    # (A00+A11) = [A00[0,0]+A11[0,0], A00[0,1]+A11[0,1]]
    #              [A00[1,0]+A11[1,0], A00[1,1]+A11[1,1]]
    
    # So the 2x2 "matrix of blocks" for M1's A is:
    # [A00+A11]  (it's a single 2x2 block that is the sum of two 2x2 blocks)
    
    # Wait, I think I'm overcomplicating this. Let me think of it differently.
    
    # The 4x4 A matrix, when viewed as a 2x2 matrix of 2x2 blocks, is:
    # A_blocks = [[A00, A01],
    #             [A10, A11]]
    
    # For Strassen at the block level, we form new 2x2 matrices of blocks:
    # M1_A = A00+A11 (element-wise sum of the 2x2 blocks)
    # M1_B = B00+B11
    # etc.
    
    # Each of these is a 2x2 matrix of 2x2 blocks, i.e., a 4x4 matrix.
    
    # But for the tensor decomposition, we need to express each scalar entry of the
    # resulting 4x4 matrices as linear combinations of the original 4x4 entries.
    
    # For M1_A = A00+A11:
    # (M1_A)[i,j] = A00[i,j] + A11[i,j] for i,j in 0,1
    # In global 4x4 indices:
    # (M1_A)[0,0] = A[0,0] + A[2,2]
    # (M1_A)[0,1] = A[0,1] + A[2,3]
    # (M1_A)[1,0] = A[1,0] + A[3,2]
    # (M1_A)[1,1] = A[1,1] + A[3,3]
    
    # So for the 2x2 Strassen applied to M1_A * M1_B, the U factors are:
    # U2[local_idx, s] * (M1_A)[local_i, local_j]
    
    # Where (M1_A)[local_i, local_j] is a linear combination of global A entries.
    
    # Let me define for each top-level mult t a mapping from local 2x2 positions
    # to global 4x4 entries with coefficients.
    
    # For top-level mult t, define:
    # A_map[t][li][lj] = list of (global_idx, coeff) for the entry at local position (li,lj)
    
    top_level_mults = []
    
    # M1: A = A00+A11, B = B00+B11
    # A00 is at global rows 0-1, cols 0-1; A11 is at global rows 2-3, cols 2-3
    A_map_1 = [
        [(0, 1), (8, 1)],    # [0,0]: A[0,0] + A[2,2]
        [(1, 1), (9, 1)],    # [0,1]: A[0,1] + A[2,3]
        [(4, 1), (12, 1)],   # [1,0]: A[1,0] + A[3,2]
        [(5, 1), (13, 1)]    # [1,1]: A[1,1] + A[3,3]
    ]
    B_map_1 = [
        [(0, 1), (8, 1)],
        [(1, 1), (9, 1)],
        [(4, 1), (12, 1)],
        [(5, 1), (13, 1)]
    ]
    C_map_1 = [
        [(0, 1), (8, 1)],
        [(1, 1), (9, 1)],
        [(4, 1), (12, 1)],
        [(5, 1), (13, 1)]
    ]
    top_level_mults.append((A_map_1, B_map_1, C_map_1))
    
    # M2: A = A10+A11, B = B00
    # A10 is at global rows 2-3, cols 0-1; A11 is at global rows 2-3, cols 2-3
    A_map_2 = [
        [(8, 1), (10, 1)],
        [(9, 1), (11, 1)],
        [(12, 1), (14, 1)],
        [(13, 1), (15, 1)]
    ]
    B_map_2 = [
        [(0, 1)],
        [(1, 1)],
        [(4, 1)],
        [(5, 1)]
    ]
    C_map_2 = [
        [(8, 1)],
        [(9, 1)],
        [(12, 1)],
        [(13, 1)]
    ]
    top_level_mults.append((A_map_2, B_map_2, C_map_2))
    
    # M3: A = A00, B = B10-B11
    A_map_3 = [
        [(0, 1)],
        [(1, 1)],
        [(4, 1)],
        [(5, 1)]
    ]
    B_map_3 = [
        [(8, 1), (10, -1)],
        [(9, 1), (11, -1)],
        [(12, 1), (14, -1)],
        [(13, 1), (15, -1)]
    ]
    C_map_3 = [
        [(2, 1)],
        [(3, 1)],
        [(6, 1)],
        [(7, 1)]
    ]
    top_level_mults.append((A_map_3, B_map_3, C_map_3))
    
    # M4: A = A11, B = B01-B00
    A_map_4 = [
        [(10, 1)],
        [(11, 1)],
        [(14, 1)],
        [(15, 1)]
    ]
    B_map_4 = [
        [(2, 1), (0, -1)],
        [(3, 1), (1, -1)],
        [(6, 1), (4, -1)],
        [(7, 1), (5, -1)]
    ]
    C_map_4 = [
        [(0, 1)],
        [(1, 1)],
        [(4, 1)],
        [(5, 1)]
    ]
    top_level_mults.append((A_map_4, B_map_4, C_map_4))
    
    # M5: A = A00+A01, B = B11
    A_map_5 = [
        [(0, 1), (2, 1)],
        [(1, 1), (3, 1)],
        [(4, 1), (6, 1)],
        [(5, 1), (7, 1)]
    ]
    B_map_5 = [
        [(10, 1)],
        [(11, 1)],
        [(14, 1)],
        [(15, 1)]
    ]
    C_map_5 = [
        [(0, 1)],
        [(1, 1)],
        [(4, 1)],
        [(5, 1)]
    ]
    top_level_mults.append((A_map_5, B_map_5, C_map_5))
    
    # M6: A = A10-A00, B = B00+B01
    A_map_6 = [
        [(8, 1), (0, -1)],
        [(9, 1), (1, -1)],
        [(12, 1), (4, -1)],
        [(13, 1), (5, -1)]
    ]
    B_map_6 = [
        [(0, 1), (2, 1)],
        [(1, 1), (3, 1)],
        [(4, 1), (6, 1)],
        [(5, 1), (7, 1)]
    ]
    C_map_6 = [
        [(10, 1)],
        [(11, 1)],
        [(14, 1)],
        [(15, 1)]
    ]
    top_level_mults.append((A_map_6, B_map_6, C_map_6))
    
    # M7: A = A01-A11, B = B10+B11
    A_map_7 = [
        [(2, 1), (10, -1)],
        [(3, 1), (11, -1)],
        [(6, 1), (14, -1)],
        [(7, 1), (15, -1)]
    ]
    B_map_7 = [
        [(8, 1), (10, 1)],
        [(9, 1), (11, 1)],
        [(12, 1), (14, 1)],
        [(13, 1), (15, 1)]
    ]
    C_map_7 = [
        [(0, 1)],
        [(1, 1)],
        [(4, 1)],
        [(5, 1)]
    ]
    top_level_mults.append((A_map_7, B_map_7, C_map_7))
    
    # Now build the full 4x4x49 decomposition.
    # For each top-level mult t (0..6) and sub-mult s (0..6):
    # Column index = t*7 + s
    
    # U2, V2, W2 are from _strassen_2x2()
    # U2[li, s] where li = 0,1,2,3 corresponds to local 2x2 positions [0,0],[0,1],[1,0],[1,1]
    # W2[lk, s] where lk = 0,1,2,3 corresponds to local 2x2 C positions [0,0],[0,1],[1,0],[1,1]
    # But in our _strassen_2x2, W is indexed as W[k*2+i], so:
    # W[0]=C[0,0], W[1]=C[0,1], W[2]=C[1,0], W[3]=C[1,1]
    
    for t in range(7):
        A_map_t, B_map_t, C_map_t = top_level_mults[t]
        for s in range(7):
            col = t * 7 + s
            
            # U contributions
            for li in range(4):
                u_coeff = U2[li][s]
                if u_coeff == 0:
                    continue
                for global_idx, coeff in A_map_t[li]:
                    U[global_idx][col] += u_coeff * coeff
            
            # V contributions
            for li in range(4):
                v_coeff = V2[li][s]
                if v_coeff == 0:
                    continue
                for global_idx, coeff in B_map_t[li]:
                    V[global_idx][col] += v_coeff * coeff
            
            # W contributions
            for li in range(4):
                w_coeff = W2[li][s]
                if w_coeff == 0:
                    continue
                for global_idx, coeff in C_map_t[li]:
                    W[global_idx][col] += w_coeff * coeff
    
    return U, V, W


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    
    Uses Strassen for the 4x4x4 core (rank 49) plus schoolbook for the remaining parts.
    
    A (4x5) = [A_left (4x4) | A_right (4x1)]
    B (5x5) = [B_top (4x4) | B_top_right (4x1)]
               [B_bottom (1x4) | B_bottom_right (1x1)]
    
    C (4x5) = A * B
    
    C[i,k] for k in 0..3: sum_{j=0}^{3} A[i,j]*B[j,k] + A[i,4]*B[4,k]
    C[i,4]: sum_{j=0}^{3} A[i,j]*B[j,4] + A[i,4]*B[4,4]
    
    Parts:
    1. A_left * B_top (4x4 * 4x4) -> contributes to C[0:4, 0:4], rank 49 (Strassen)
    2. A_left * B_top_right (4x4 * 4x1) -> contributes to C[0:4, 4], rank 16 (schoolbook)
    3. A_right * B_bottom (4x1 * 1x4) -> contributes to C[0:4, 0:4], rank 4 (schoolbook)
    4. A_right * B_bottom_right (4x1 * 1x1) -> contributes to C[0:4, 4], rank 4 (schoolbook)
    
    Total rank: 49 + 16 + 4 + 4 = 73
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    
    start_time = time.time()
    deadline = start_time + 115
    
    # Get Strassen 4x4x4 decomposition
    U_str, V_str, W_str = _strassen_4x4()
    rank_str = 49
    
    # Total rank
    rank_total = 49 + 16 + 4 + 4  # 73
    
    U = [[0] * rank_total for _ in range(nm)]
    V = [[0] * rank_total for _ in range(mp)]
    W = [[0] * rank_total for _ in range(pn)]
    
    # Part 1: A_left * B_top -> C[0:4, 0:4]
    # A_left[i,j] for i in 0..3, j in 0..3 -> U[i*5+j]
    # B_top[j,k] for j in 0..3, k in 0..3 -> V[j*5+k]
    # C[i,k] for i in 0..3, k in 0..3 -> W[k*4+i]
    
    for i in range(4):
        for j in range(4):
            for s in range(49):
                val = U_str[i*4+j][s]
                if val != 0:
                    U[i*5+j][s] = val
    
    for j in range(4):
        for k in range(4):
            for s in range(49):
                val = V_str[j*4+k][s]
                if val != 0:
                    V[j*5+k][s] = val
    
    for i in range(4):
        for k in range(4):
            for s in range(49):
                val = W_str[k*4+i][s]
                if val != 0:
                    W[k*4+i][s] = val
    
    # Part 2: A_left * B_top_right -> C[0:4, 4]
    # A_left[i,j] for i in 0..3, j in 0..3 -> U[i*5+j]
    # B_top_right[j,4] for j in 0..3 -> V[j*5+4]
    # C[i,4] for i in 0..3 -> W[4*4+i] = W[16+i]
    
    col_offset = 49
    for i in range(4):
        for j in range(4):
            col = col_offset + i*4 + j
            U[i*5+j][col] = 1
            V[j*5+4][col] = 1
            W[16+i][col] = 1
    
    # Part 3: A_right * B_bottom -> C[0:4, 0:4]
    # A_right[i,4] for i in 0..3 -> U[i*5+4]
    # B_bottom[4,k] for k in 0..3 -> V[4*5+k] = V[20+k]
    # C[i,k] for i in 0..3, k in 0..3 -> W[k*4+i]
    
    col_offset = 49 + 16  # 65
    for i in range(4):
        for k in range(4):
            col = col_offset + i*4 + k
            U[i*5+4][col] = 1
            V[20+k][col] = 1
            W[k*4+i][col] = 1
    
    # Part 4: A_right * B_bottom_right -> C[0:4, 4]
    # A_right[i,4] for i in 0..3 -> U[i*5+4]
    # B_bottom_right[4,4] -> V[4*5+4] = V[24]
    # C[i,4] for i in 0..3 -> W[16+i]
    
    col_offset = 49 + 16 + 4  # 69
    for i in range(4):
        col = col_offset + i
        U[i*5+4][col] = 1
        V[24][col] = 1
        W[16+i][col] = 1
    
    # Verify
    def verify(U, V, W, tol=1e-9):
        r = rank_total
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r_idx in range(r):
                        s += U[i*m+j][r_idx] * V[j*p+k][r_idx] * W[k*n+i][r_idx]
                    if abs(s - 1.0) > tol:
                        return False, (i, j, k, s)
        return True, None
    
    valid, info = verify(U, V, W)
    if not valid:
        # Fallback to schoolbook
        schoolbook_rank = n * m * p  # 100
        U_school = [[0] * schoolbook_rank for _ in range(nm)]
        V_school = [[0] * schoolbook_rank for _ in range(mp)]
        W_school = [[0] * schoolbook_rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j][col] = 1
                    V_school[j*p+k][col] = 1
                    W_school[k*n+i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
