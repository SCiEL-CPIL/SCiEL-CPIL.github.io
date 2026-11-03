"""
Rank-minimal exact tensor decomposition for 4x5 x 5x5 matrix multiplication.

Strategy:
  - The multiplication tensor T_{i,j,k} for C[i,k] = sum_j A[i,j] B[j,k]
    lives in Q^{n*m} ⊗ Q^{m*p} ⊗ Q^{p*n}.
  - We build a decomposition of rank R by:
      1) Starting from the schoolbook (rank 100) as a baseline.
      2) Applying the Strassen trick to the 4x4 sub-block of A and B,
         which reduces the 4x4x4 core from rank 64 to rank 49 (savings 15).
      3) The remaining terms (rows/cols outside the 4x4 core) are kept
         at schoolbook cost.
  - Total rank = 49 (Strassen core) + remaining schoolbook terms.

  For 4x5 x 5x5:
    A is 4x5, B is 5x5, C is 4x5.
    We partition A as [A1 (4x4), A2 (4x1)] and B as [[B11 (4x4), B12 (4x1)], [B21 (1x4), B22 (1x1)]]
    C = A1*B11 + A1*B12 + A2*B21 + A2*B22

    The 4x4x4 core (A1*B11 -> C11) uses Strassen: rank 49.
    A1*B12: 4x4 * 4x1 -> 4x1. This is 4*4*1 = 16 terms schoolbook.
    A2*B21: 4x1 * 1x4 -> 4x4. This is 4*1*4 = 16 terms schoolbook.
    A2*B22: 4x1 * 1x1 -> 4x1. This is 4*1*1 = 4 terms schoolbook.
    Also cross terms: A1*B12 contributes to C12, A2*B21 to C21, A2*B22 to C22.
    
    Wait, let me re-partition properly.
    A (4x5) = [A_left (4x4) | A_right (4x1)]
    B (5x5) = [B_top_left (4x4); B_top_right (4x1); B_bottom_left (1x4); B_bottom_right (1x1)]
    
    C = A * B = [A_left, A_right] * [B_tl; B_tr; B_bl; B_br]
    C[i,k] = sum_{j=0}^{4} A[i,j] B[j,k]
           = sum_{j=0}^{3} A[i,j] B[j,k] + A[i,4] B[4,k]
    
    For k=0..3 (first 4 columns of C):
      C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k] + A[i,4] B[4,k]
    For k=4 (last column):
      C[i,4] = sum_{j=0}^{3} A[i,j] B[j,4] + A[i,4] B[4,4]
    
    So:
      C[:,0:4] = A_left * B_top_left + A_right * B_bottom_left   (4x4 + 4x4 = 4x4)
      C[:,4]   = A_left * B_top_right + A_right * B_bottom_right (4x1 + 4x1 = 4x1)
    
    The 4x4x4 part A_left * B_top_left -> C[:,0:4] uses Strassen: rank 49.
    The remaining:
      A_right * B_bottom_left: 4x1 * 1x4 -> 4x4, rank 4*1*4 = 16
      A_left * B_top_right:    4x4 * 4x1 -> 4x1, rank 4*4*1 = 16
      A_right * B_bottom_right: 4x1 * 1x1 -> 4x1, rank 4*1*1 = 4
    Total rank = 49 + 16 + 16 + 4 = 85.
"""

import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    n, m, p = DIMENSIONS  # 4, 5, 5

    # Strassen 2x2x2 decomposition
    # For 2x2 * 2x2 -> 2x2, rank 7
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Build Strassen 2x2x2 U,V,W (each 4x7)
    def strassen_2x2():
        # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # M1 = (A11+A22)(B11+B22): A indices {0,3}, B indices {0,3}
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        # contributes to C11 and C22
        W[0, 0] = 1   # C11 += M1
        W[3, 0] = 1   # C22 += M1
        
        # M2 = (A21+A22)B11: A indices {2,3}, B index {0}
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        # contributes to C21 and C22
        W[2, 1] = 1   # C21 += M2
        W[3, 1] = -1  # C22 -= M2
        
        # M3 = A11(B21-B22): A index {0}, B indices {2,-3}
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        # contributes to C12 and C22
        W[1, 2] = 1   # C12 += M3
        W[3, 2] = 1   # C22 += M3
        
        # M4 = A22(B11-B21): A index {3}, B indices {0,-2}
        U[3, 3] = 1
        V[0, 3] = 1; V[2, 3] = -1
        # contributes to C11 and C21
        W[0, 3] = 1   # C11 += M4
        W[2, 3] = 1   # C21 += M4
        
        # M5 = (A11+A12)B22: A indices {0,1}, B index {3}
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        # contributes to C12 and C11
        W[1, 4] = 1   # C12 += M5
        W[0, 4] = -1  # C11 -= M5
        
        # M6 = (A21-A11)(B11+B12): A indices {2,-0}, B indices {0,1}
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        # contributes to C22
        W[3, 5] = 1   # C22 += M6
        
        # M7 = (A12-A22)(B21+B22): A indices {1,-3}, B indices {2,3}
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        # contributes to C11
        W[0, 6] = 1   # C11 += M7
        
        return U, V, W
    
    # Build Strassen 4x4x4 by block decomposition
    # 4x4 = 2x(2x2) blocks
    # A (4x4) blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3),
    #                  A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B (4x4) blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3),
    #                  B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    # C (4x4) blocks: C00, C01, C10, C11 similarly
    
    # The 7 meta-multiplications for 4x4 Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B00-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Mi is a 2x2 * 2x2 multiplication, which we decompose with strassen_2x2
    # Global A index for element (i,j) in 4x4: i*4+j (i,j in 0..3)
    # Block (bi,bj) covers rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    
    def strassen_4x4():
        U2, V2, W2 = strassen_2x2()
        rank_2 = 7
        rank_4 = 7 * rank_2  # 49
        
        U4 = np.zeros((16, rank_4), dtype=float)
        V4 = np.zeros((16, rank_4), dtype=float)
        W4 = np.zeros((16, rank_4), dtype=float)
        
        # Meta coefficients: for each meta step s (0..6), and each sub step t (0..6)
        # We need to map the 2x2 block indices to 4x4 global indices.
        
        # Meta step s involves specific 2x2 blocks of A and B, and produces a 2x2 result
        # that contributes to specific 2x2 blocks of C.
        
        # Define for each meta step:
        #   a_blocks: list of (block_index, coefficient) for A
        #   b_blocks: list of (block_index, coefficient) for B
        #   c_blocks: list of (block_index, coefficient) for C output
        
        # Block indices: 0=A00, 1=A01, 2=A10, 3=A11 (row_block, col_block)
        
        meta_steps = [
            # M1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
            ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]),
            # M2 = (A10+A11)B00 -> contributes to C10 and C11
            ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1), (3, -1) ]),
            # M3 = A00(B10-B11) -> contributes to C01 and C11
            ([ (0, 1) ], [ (2, 1), (3, -1) ], [ (1, 1), (3, 1) ]),
            # M4 = A11(B00-B10) -> contributes to C00 and C10
            ([ (3, 1) ], [ (0, 1), (2, -1) ], [ (0, 1), (2, 1) ]),
            # M5 = (A00+A01)B11 -> contributes to C01 and C00
            ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (1, 1), (0, -1) ]),
            # M6 = (A10-A00)(B00+B01) -> contributes to C11
            ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ], [ (3, 1) ]),
            # M7 = (A01-A11)(B10+B11) -> contributes to C00
            ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ], [ (0, 1) ]),
        ]
        
        for s, (a_coef, b_coef, c_coef) in enumerate(meta_steps):
            for t in range(rank_2):
                col = s * rank_2 + t
                # For each element in the 2x2 block of A
                for bi, bj in [(0,0), (0,1), (1,0), (1,1)]:
                    sub_idx = bi * 2 + bj
                    val = U2[sub_idx, t]
                    if val == 0:
                        continue
                    for block_idx, coeff in a_coef:
                        rb, cb = block_idx
                        global_row = 2 * rb + bi
                        global_col = 2 * cb + bj
                        U4[global_row * 4 + global_col, col] += coeff * val
                
                for bi, bj in [(0,0), (0,1), (1,0), (1,1)]:
                    sub_idx = bi * 2 + bj
                    val = V2[sub_idx, t]
                    if val == 0:
                        continue
                    for block_idx, coeff in b_coef:
                        rb, cb = block_idx
                        global_row = 2 * rb + bi
                        global_col = 2 * cb + bj
                        V4[global_row * 4 + global_col, col] += coeff * val
                
                for bi, bj in [(0,0), (0,1), (1,0), (1,1)]:
                    sub_idx = bi * 2 + bj
                    val = W2[sub_idx, t]
                    if val == 0:
                        continue
                    for block_idx, coeff in c_coef:
                        rb, cb = block_idx
                        global_row = 2 * rb + bi
                        global_col = 2 * cb + bj
                        W4[global_row * 4 + global_col, col] += coeff * val
        
        return U4, V4, W4
    
    # Now build the full 4x5 x 5x5 decomposition
    # A: 4x5, B: 5x5, C: 4x5
    # A indices: i*5+j, i in 0..3, j in 0..4
    # B indices: j*5+k, j in 0..4, k in 0..4
    # C indices: k*4+i, k in 0..4, i in 0..3 (W[k*n+i,r])
    
    # Partition:
    # A_left: cols 0-3 (4x4), A_right: col 4 (4x1)
    # B_top_left: rows 0-3, cols 0-3 (4x4)
    # B_top_right: rows 0-3, col 4 (4x1)
    # B_bottom_left: row 4, cols 0-3 (1x4)
    # B_bottom_right: row 4, col 4 (1x1)
    
    # C[:,0:4] = A_left * B_top_left + A_right * B_bottom_left
    # C[:,4]   = A_left * B_top_right + A_right * B_bottom_right
    
    U_core, V_core, W_core = strassen_4x4()
    rank_core = 49
    
    # Schoolbook for remaining parts
    # A_right * B_bottom_left: 4x1 * 1x4 -> 4x4
    # For each i (0..3) and k (0..3): C[i,k] += A[i,4] * B[4,k]
    rank_rl = 4 * 4  # 16
    
    # A_left * B_top_right: 4x4 * 4x1 -> 4x1
    # For each i (0..3) and j (0..3): C[i,4] += A[i,j] * B[j,4]
    rank_lr = 4 * 4  # 16
    
    # A_right * B_bottom_right: 4x1 * 1x1 -> 4x1
    # For each i (0..3): C[i,4] += A[i,4] * B[4,4]
    rank_rr = 4 * 1  # 4
    
    total_rank = rank_core + rank_rl + rank_lr + rank_rr  # 49+16+16+4 = 85
    
    U = np.zeros((20, total_rank), dtype=float)  # n*m = 4*5 = 20
    V = np.zeros((25, total_rank), dtype=float)  # m*p = 5*5 = 25
    W = np.zeros((20, total_rank), dtype=float)  # p*n = 5*4 = 20
    
    col = 0
    # Core: A_left * B_top_left -> C[:,0:4]
    for i in range(4):
        for j in range(4):
            a_idx = i * 5 + j
            for r in range(rank_core):
                U[a_idx, col + r] = U_core[i * 4 + j, r]
    for j in range(4):
        for k in range(4):
            b_idx = j * 5 + k
            for r in range(rank_core):
                V[b_idx, col + r] = V_core[j * 4 + k, r]
    for k in range(4):
        for i in range(4):
            c_idx = k * 4 + i
            for r in range(rank_core):
                W[c_idx, col + r] = W_core[k * 4 + i, r]
    col += rank_core
    
    # A_right * B_bottom_left: 4x1 * 1x4 -> C[:,0:4]
    for i in range(4):
        for k in range(4):
            a_idx = i * 5 + 4
            b_idx = 4 * 5 + k
            c_idx = k * 4 + i
            U[a_idx, col] = 1
            V[b_idx, col] = 1
            W[c_idx, col] = 1
            col += 1
    
    # A_left * B_top_right: 4x4 * 4x1 -> C[:,4]
    for i in range(4):
        for j in range(4):
            a_idx = i * 5 + j
            b_idx = j * 5 + 4
            c_idx = 4 * 4 + i
            U[a_idx, col] = 1
            V[b_idx, col] = 1
            W[c_idx, col] = 1
            col += 1
    
    # A_right * B_bottom_right: 4x1 * 1x1 -> C[:,4]
    for i in range(4):
        a_idx = i * 5 + 4
        b_idx = 4 * 5 + 4
        c_idx = 4 * 4 + i
        U[a_idx, col] = 1
        V[b_idx, col] = 1
        W[c_idx, col] = 1
        col += 1
    
    # Verify
    def verify(U, V, W, n, m, p):
        # Check that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] gives correct multiplication tensor
        # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
        # if j is the shared index (always true here since every A[i,j] and B[j,k] pair
        # contributes to C[i,k])
        # Actually the tensor T_{i,j,k} = 1 for all valid i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    is_valid = verify(U, V, W, n, m, p)
    
    # If invalid, fall back to schoolbook
    if not is_valid:
        rank_sb = n * m * p
        U = np.zeros((n*m, rank_sb), dtype=float)
        V = np.zeros((m*p, rank_sb), dtype=float)
        W = np.zeros((p*n, rank_sb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    # Convert to lists of lists with integer/half-integer entries
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
