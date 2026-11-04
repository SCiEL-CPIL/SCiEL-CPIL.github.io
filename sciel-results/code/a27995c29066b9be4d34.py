"""Rewritten program for 6x6 matrix multiplication tensor decomposition.

Strategy: Use the known rank-19 decomposition for 3x3 matrix multiplication
(Liu's algorithm) and apply it to the 6x6 case by treating the 6x6 matrices
as 2x2 blocks of 3x3 matrices, where each 3x3 block multiplication uses
the rank-19 scheme. This gives rank 4*19 = 76 for the 6x6 case.

Actually, let me reconsider. The 6x6 matrix multiplication can be viewed
as multiplying two 6x6 matrices. We can partition each into 2x2 blocks 
of 3x3 matrices. The 2x2 block multiplication uses Strassen's algorithm 
(rank 7 for 2x2 scalar), but here each "scalar" is a 3x3 matrix. 

Using Strassen for the 2x2 block structure with 3x3 matrix entries:
- We need 7 products of 3x3 matrices
- Each 3x3 product uses rank 19 (Liu's algorithm)
- Total rank: 7 * 19 = 133

Alternatively, we could use a direct approach. Let me think about what's 
actually achievable.

For 3x3 matrix multiplication, the best known rank is 19 (Liu 1998).
For 6x6, we can use the tensor product structure.

Let me implement Liu's rank-19 decomposition for 3x3 and combine with 
Strassen for 2x2 blocks to get rank 7*19 = 133 for 6x6.

Actually, the simplest correct approach that improves on 216:
- Use Strassen (rank 7) for 2x2 block structure where each block is 3x3
- Use schoolbook (rank 27) for each 3x3 block product
- Total: 7 * 27 = 189

This was measured at 1.143 in the knowledge. Let me implement this correctly.
"""

import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def _schoolbook_3x3_factors():
    """Return U, V, W for 3x3 matrix multiplication (rank 27).
    
    Convention: A is 3x3 (i,j), B is 3x3 (j,k), C = A*B is 3x3 (i,k).
    U has shape (3*3, 27) = (9, 27), V has shape (3*3, 27) = (9, 27), 
    W has shape (3*3, 27) = (9, 27).
    
    Indexing: U[i*3+j, c], V[j*3+k, c], W[k*3+i, c] for the trace convention.
    """
    n = 3
    rank = n * n * n  # 27
    U = np.zeros((n * n, rank))
    V = np.zeros((n * n, rank))
    W = np.zeros((n * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                U[i * n + j, col] = 1
                V[j * n + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2_indices():
    """
    Strassen's algorithm for 2x2 matrix multiplication.
    
    A = [[A11, A12], [A21, A22]]
    B = [[B11, B12], [B21, B22]]
    C = A*B = [[C11, C12], [C21, C22]]
    
    Strassen products (7 multiplications):
    M1 = (A11 + A22) * (B11 + B22)
    M2 = (A21 + A22) * B11
    M3 = A11 * (B12 - B22)
    M4 = A22 * (B21 - B11)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)
    
    Results:
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # For each Strassen product M_t, we need to express it as a product
    # of a linear combination of A blocks and a linear combination of B blocks.
    # Each A block is a 3x3 matrix, each B block is a 3x3 matrix.
    # The product of two 3x3 matrices uses rank 27 (schoolbook).
    
    # U factors: for each of the 7 Strassen products, and for each of the 27 
    # schoolbook terms within, we specify which A block entries are used.
    # V factors: similarly for B block entries.
    # W factors: how the results combine into C blocks.
    
    # Let's build this step by step.
    # 
    # For the full 6x6 problem:
    # A is 6x6, partitioned into 2x2 blocks of 3x3:
    #   A = [[A11, A12], [A21, A22]]
    # B is 6x6, partitioned into 2x2 blocks of 3x3:
    #   B = [[B11, B12], [B21, B22]]
    # C = A*B = [[C11, C12], [C21, C22]]
    #
    # Each block is 3x3. The input to U is the 36 entries of A (6*6),
    # indexed as A[row, col] where row, col in 0..5.
    # Block A11 = A[0:3, 0:3], A12 = A[0:3, 3:6], A21 = A[3:6, 0:3], A22 = A[3:6, 3:6]
    #
    # For the schoolbook 3x3 multiplication within each Strassen product:
    # If we're computing X * Y where X and Y are 3x3 matrices (linear combos of blocks),
    # then for each (i,j,k) in the 3x3 product, we have:
    #   U entry: coefficient for X[i,j]
    #   V entry: coefficient for Y[j,k]  
    #   W entry: coefficient for C_block[i,k]
    
    # Let's define the Strassen linear combinations.
    # A-blocks: 1=A11, 2=A12, 3=A21, 4=A22
    # B-blocks: 1=B11, 2=B12, 3=B21, 4=B22
    
    # For each Strassen product t=1..7, define:
    #   A_coeffs[t]: dict from block_id -> coefficient
    #   B_coeffs[t]: dict from block_id -> coefficient
    #   C_contribs[t]: dict from (result_block, i, k) -> coefficient
    #     where result_block is which C block (1=C11, 2=C12, 3=C21, 4=C22)
    
    # Strassen M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # A block definitions (which 3x3 submatrix of the 6x6 A):
    # Block 1 (A11): rows 0-2, cols 0-2
    # Block 2 (A12): rows 0-2, cols 3-5
    # Block 3 (A21): rows 3-5, cols 0-2
    # Block 4 (A22): rows 3-5, cols 3-5
    
    # B block definitions:
    # Block 1 (B11): rows 0-2, cols 0-2
    # Block 2 (B12): rows 0-2, cols 3-5
    # Block 3 (B21): rows 3-5, cols 0-2
    # Block 4 (B22): rows 3-5, cols 3-5
    
    # C block definitions:
    # Block 1 (C11): rows 0-2, cols 0-2
    # Block 2 (C12): rows 0-2, cols 3-5
    # Block 3 (C21): rows 3-5, cols 0-2
    # Block 4 (C22): rows 3-5, cols 3-5
    
    # For the 3x3 schoolbook within each block product:
    # X[i,j] * Y[j,k] contributes to Z[i,k]
    # In the 6x6 indexing:
    #   X[i,j] where X is a block: if block is at position (br, bc) in 2x2,
    #     then X[i,j] = A[br*3+i, bc*3+j]
    #   Y[j,k] where Y is a block at position (br, bc):
    #     then Y[j,k] = B[br*3+j, bc*3+k]
    #   Z[i,k] where Z is a block at position (br, bc):
    #     then Z[i,k] = C[br*3+i, bc*3+k]
    
    # Now let's build the full decomposition.
    
    # A_block_positions: block_id -> (row_offset, col_offset) in the 6x6 A
    A_block_pos = {
        1: (0, 0),  # A11
        2: (0, 3),  # A12
        3: (3, 0),  # A21
        4: (3, 3),  # A22
    }
    
    # B_block_positions: block_id -> (row_offset, col_offset) in the 6x6 B
    B_block_pos = {
        1: (0, 0),  # B11
        2: (0, 3),  # B12
        3: (3, 0),  # B21
        4: (3, 3),  # B22
    }
    
    # C_block_positions: block_id -> (row_offset, col_offset) in the 6x6 C
    C_block_pos = {
        1: (0, 0),  # C11
        2: (0, 3),  # C12
        3: (3, 0),  # C21
        4: (3, 3),  # C22
    }
    
    # Strassen products:
    # t: (A_coeffs, B_coeffs)
    # A_coeffs: {block_id: coeff}
    # B_coeffs: {block_id: coeff}
    
    strassen_A = [
        {1: 1, 4: 1},       # M1: A11 + A22
        {3: 1, 4: 1},       # M2: A21 + A22
        {1: 1},             # M3: A11
        {4: 1},             # M4: A22
        {1: 1, 2: 1},       # M5: A11 + A12
        {3: 1, 1: -1},      # M6: A21 - A11
        {2: 1, 4: -1},      # M7: A12 - A22
    ]
    
    strassen_B = [
        {1: 1, 4: 1},       # M1: B11 + B22
        {1: 1},             # M2: B11
        {2: 1, 4: -1},      # M3: B12 - B22
        {3: 1, 1: -1},      # M4: B21 - B11
        {4: 1},             # M5: B22
        {1: 1, 2: 1},       # M6: B11 + B12
        {3: 1, 4: 1},       # M7: B21 + B22
    ]
    
    # C contributions: for each Strassen product t, which C blocks does it contribute to, and with what coefficient
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # c_contribs[t][c_block_id] = coefficient
    c_contribs = [
        {1: 1, 4: 1},       # M1 -> C11 (coeff 1), C22 (coeff 1)
        {3: 1, 4: -1},      # M2 -> C21 (coeff 1), C22 (coeff -1)
        {2: 1, 4: 1},       # M3 -> C12 (coeff 1), C22 (coeff 1)
        {1: 1, 3: 1},       # M4 -> C11 (coeff 1), C21 (coeff 1)
        {1: -1, 2: 1},      # M5 -> C11 (coeff -1), C12 (coeff 1)
        {4: 1},             # M6 -> C22 (coeff 1)
        {1: 1},             # M7 -> C11 (coeff 1)
    ]
    
    # Now build the full factor matrices.
    # Total rank = 7 * 27 = 189
    total_rank = 7 * 27
    n = 6
    m = 6
    p = 6
    
    U = np.zeros((n * m, total_rank))  # (36, 189)
    V = np.zeros((m * p, total_rank))  # (36, 189)
    W = np.zeros((p * n, total_rank))  # (36, 189)
    
    for t in range(7):  # Strassen product index 0-6
        A_coeffs = strassen_A[t]
        B_coeffs = strassen_B[t]
        C_coeffs = c_contribs[t]
        
        # For each of the 27 schoolbook terms in the 3x3 product:
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    col = t * 27 + (i * 3 + j) * 3 + k
                    
                    # U: coefficient for A entries
                    # X[i,j] = sum over A blocks b of A_coeffs[b] * A_block[b][i,j]
                    # A_block[b][i,j] = A[A_block_pos[b][0] + i, A_block_pos[b][1] + j]
                    for b, coeff in A_coeffs.items():
                        row_off, col_off = A_block_pos[b]
                        a_row = row_off + i
                        a_col = col_off + j
                        U[a_row * m + a_col, col] += coeff
                    
                    # V: coefficient for B entries
                    # Y[j,k] = sum over B blocks b of B_coeffs[b] * B_block[b][j,k]
                    # B_block[b][j,k] = B[B_block_pos[b][0] + j, B_block_pos[b][1] + k]
                    for b, coeff in B_coeffs.items():
                        row_off, col_off = B_block_pos[b]
                        b_row = row_off + j
                        b_col = col_off + k
                        V[b_row * p + b_col, col] += coeff
                    
                    # W: coefficient for C entries
                    # Z[i,k] contributes to C blocks
                    # C_block[c][i,k] = C[C_block_pos[c][0] + i, C_block_pos[c][1] + k]
                    for c, coeff in C_coeffs.items():
                        row_off, col_off = C_block_pos[c]
                        c_row = row_off + i
                        c_col = col_off + k
                        # W index: W[k*n + i, col] per the trace convention
                        # Wait, let me re-read the convention.
                        # "The output factor follows the official trace convention W[k*n+i,r]"
                        # So W is indexed as W[k*n + i, r] where k is the column index of C and i is the row index?
                        # 
                        # In standard matrix multiplication C[i,k] = sum_j A[i,j] * B[j,k]
                        # The tensor T[i,j,k] = A[i,j] * B[j,k] * C[i,k]
                        # U[i*m+j, r], V[j*p+k, r], W[?, r]
                        # 
                        # The trace convention W[k*n+i, r] means:
                        # W is indexed by (k, i) where k is the column of C and i is the row of C.
                        # So W[k*n + i, r] corresponds to C[i, k].
                        
                        # C[i,k] where i is the row in the 3x3 block and k is the column in the 3x3 block.
                        # The full C index: row = c_row, col = c_col
                        # So i_full = c_row, k_full = c_col
                        # W index: k_full * n + i_full
                        W[c_col * n + c_row, col] += coeff
    
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0  # seconds, with safety margin
    
    try:
        U, V, W = _strassen_2x2_indices()
        
        # Convert to lists of lists for the expected format
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        # Ensure entries are integers (they should be, given our construction)
        # But let's verify and clean up
        U_clean = []
        for row in U_list:
            U_clean.append([int(round(x)) for x in row])
        V_clean = []
        for row in V_list:
            V_clean.append([int(round(x)) for x in row])
        W_clean = []
        for row in W_list:
            W_clean.append([int(round(x)) for x in row])
        
        elapsed = time.time() - start_time
        if elapsed > deadline:
            # Should not happen for this static construction
            pass
        
        return {"U": U_clean, "V": V_clean, "W": W_clean}
    
    except Exception as e:
        # Fallback to schoolbook if something goes wrong
        n, m, p = DIMENSIONS
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
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
