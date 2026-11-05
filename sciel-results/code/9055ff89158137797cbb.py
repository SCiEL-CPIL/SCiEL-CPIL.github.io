"""
Tensor decomposition for 4x4 * 4x7 matrix multiplication.

The tensor T has indices (i, j, k) where:
  i in 0..3 (row of A, 4 choices)
  j in 0..3 (column of A / row of B, 4 choices)
  k in 0..6 (column of B, 7 choices)

Standard decomposition: rank = 4*4*7 = 112.

We can improve this by using Strassen's algorithm for the 4x4x4 sub-problem.
Strassen decomposes 2x2x2 with rank 7 instead of 8.
For 4x4x4, we can recursively apply Strassen:
  - Split 4x4x4 into 8 sub-problems of 2x2x2 (block Strassen), each rank 7 -> total 56
  - Or recursively: 4x4x4 -> 8 sub-problems of 2x2x2, but we can also do
    4x4x4 as two 2x2x4 + ... no, the recursive Strassen for 4x4x4:
    Split into quadrants. Actually, the known result is that 4x4x4 can be done
    with rank 49 (recursive Strassen: 7^log2(4) = 7^2 = 49).

For 4x4x7:
  Split B into B1 (4x4) and B2 (4x3).
  A*B1: 4x4x4, use recursive Strassen -> rank 49
  A*B2: 4x4x3, standard -> rank 48
  Total: 49 + 48 = 97

But we can do better by splitting differently. Let's think about 4x4x7:
  Split B into 4 + 3 columns.
  Alternatively, split into 7 columns, each 4x1. But that's worse.

Actually, let me reconsider. The known optimal or near-optimal for 4x4x7:
  We can use the fact that 4x4x7 = 4x4x4 + 4x4x3.
  
  For 4x4x4: recursive Strassen gives rank 49.
  For 4x4x3: We can split this further. 4x4x3 = 4x4x2 + 4x4x1.
    4x4x2: can be done with 2 * (4x4x1) = 2*4 = 8? No.
    Actually, 4x4x1 has rank 4 (standard). 4x4x2 has rank at most 8 (standard) 
    but can be improved. Actually for m=1 or p=1, the rank is n*m*p (standard).
    
  Hmm, let me just use the straightforward approach:
  - 4x4x4 with recursive Strassen: rank 49
  - 4x4x3 with standard: rank 48
  Total: 97

  Can we do 4x4x3 better? 
  4x4x3: split into 4x2x3 + 2x4x3? No, that doesn't help directly.
  
  Actually, let me think about this differently. The tensor for 4x4x7:
  We can use block decomposition. Let A be 4x4, B be 4x7.
  Split A into 2x2 blocks, B into 2x7 blocks.
  Then we get a 2x2 block matrix times 2x7 block matrix.
  Using Strassen for the 2x2 block multiplication:
  Each block multiplication involves 2x2 * 2x7 or 2x2 * 2x7 etc.
  
  This is getting complex. Let me just implement:
  1. Recursive Strassen for 4x4x4 (rank 49)
  2. Standard for 4x4x3 (rank 48)
  Total rank 97.

  Actually, I recall that for 4x4x3, we might be able to do better.
  Let me check: the rank of 4x4x3 tensor. 
  Lower bound: max(4*4, 4*3, 4*3) = 16. Upper bound: 48 (standard).
  Known results: I don't have a specific reference, but let's try to construct.
  
  For now, let's go with rank 97 (49 + 48).

  Wait, I should double-check the recursive Strassen for 4x4x4.
  Strassen for 2x2x2: rank 7.
  For 4x4x4: we can view it as 8 sub-problems of 2x2x2, but Strassen reduces to 7 multiplications.
  However, each "multiplication" in Strassen for n x n is actually an n/2 x n/2 multiplication.
  So for 4x4x4 with Strassen: 7 multiplications of 2x2x2, each of rank 7.
  Total: 7 * 7 = 49. Yes, that's correct.

  So total rank = 49 + 48 = 97.

Let me implement this.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2x2.
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    Indices: i,j in 0..1 for A (2x2), k in 0..1 for B (2x2)
    W[k*2+i, j] convention? Let me check the task.
    
    Task says: W[k*n+i, r] where n is the number of rows of A.
    For 2x2x2: n=2, so W[k*2+i, r].
    U[i*m+j, r] where m is columns of A = rows of B. For 2x2x2, m=2, so U[i*2+j, r].
    V[j*p+k, r] where p is columns of B. For 2x2x2, p=2, so V[j*2+k, r].
    
    Strassen 2x2x2:
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11)B00
    M3 = A00(B10 - B11)
    M4 = A11(B01 - B00)
    M5 = (A00 + A01)B11
    M6 = (A10 - A00)(B00 + B01)
    M7 = (A01 - A11)(B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
    W[0*2+0, 0] = 1; W[1*2+1, 0] = 1  # C00 gets M1, C11 gets M1
    
    # M2 = (A10 + A11)B00
    U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    W[1*2+0, 1] = 1  # C10 gets M2
    
    # M3 = A00(B10 - B11)
    U[0*2+0, 2] = 1
    V[1*2+0, 2] = 1; V[1*2+1, 2] = -1
    W[0*2+1, 2] = 1; W[1*2+1, 2] = 1  # C01 gets M3, C11 gets M3
    
    # M4 = A11(B01 - B00)
    U[1*2+1, 3] = 1
    V[0*2+1, 3] = 1; V[0*2+0, 3] = -1
    W[0*2+0, 3] = 1; W[1*2+0, 3] = 1  # C00 gets M4, C10 gets M4
    
    # M5 = (A00 + A01)B11
    U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    W[0*2+0, 4] = -1; W[0*2+1, 4] = 1  # C00 gets -M5, C01 gets M5
    
    # M6 = (A10 - A00)(B00 + B01)
    U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1; V[0*2+1, 5] = 1
    W[1*2+1, 5] = 1  # C11 gets M6
    
    # M7 = (A01 - A11)(B10 + B11)
    U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
    V[1*2+0, 6] = 1; V[1*2+1, 6] = 1
    W[0*2+0, 6] = 1  # C00 gets M7
    
    return U, V, W

def _recursive_strassen_4x4x4():
    """
    Recursive Strassen for 4x4x4.
    Split into 2x2 blocks, use Strassen 2x2x2 for each block multiplication.
    This gives rank 7 * 7 = 49.
    
    Actually, let me think more carefully.
    A is 4x4, B is 4x4.
    Split A into 4 blocks of 2x2: A00, A01, A10, A11
    Split B into 4 blocks of 2x2: B00, B01, B10, B11
    
    Using Strassen at the block level:
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11)B00
    M3 = A00(B10 - B11)
    M4 = A11(B01 - B00)
    M5 = (A00 + A01)B11
    M6 = (A10 - A00)(B00 + B01)
    M7 = (A01 - A11)(B10 + B11)
    
    Each Mi is a product of 2x2 matrices, so each has rank 7 (using Strassen 2x2x2).
    Total rank = 7 * 7 = 49.
    
    The output C is 4x4, split into 4 blocks of 2x2:
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Get the 2x2x2 Strassen decomposition
    U2, V2, W2 = _strassen_2x2x2()
    
    # For each block-level multiplication, we need to map the block indices
    # to the full indices.
    # A[i,j] where i,j in 0..3
    # A[i,j] is in block (i//2, j//2) at position (i%2, j%2)
    # B[j,k] where j,k in 0..3
    # B[j,k] is in block (j//2, k//2) at position (j%2, k%2)
    # C[i,k] where i,k in 0..3
    # C[i,k] is in block (i//2, k//2) at position (i%2, k%2)
    
    # For block (bi, bj) in A and block (bj, bk) in B:
    # A global index: (bi*2 + i_local, bj*2 + j_local)
    # B global index: (bj*2 + j_local, bk*2 + k_local)
    # C global index: (bi*2 + i_local, bk*2 + k_local)
    
    # The 7 Strassen multiplications at block level:
    # M1: blocks (0,0) and (1,1) of A, blocks (0,0) and (1,1) of B
    #   A blocks: (0,0) and (1,1) -> global i in {0,1} U {2,3}, j in {0,1} U {2,3}
    #   B blocks: (0,0) and (1,1) -> global j in {0,1} U {2,3}, k in {0,1} U {2,3}
    #   But wait, Strassen M1 = (A00 + A11)(B00 + B11)
    #   A00 is block (0,0), A11 is block (1,1)
    #   B00 is block (0,0), B11 is block (1,1)
    
    # Let me define the block-level Strassen and then expand.
    # For each of the 7 block multiplications, we have a 2x2x2 sub-problem.
    # The sub-problem has its own U, V, W of size (4,7), (4,7), (4,7).
    # We need to map these to the global indices.
    
    # Block multiplication r (r in 0..6) involves:
    #   A blocks: determined by Strassen formula
    #   B blocks: determined by Strassen formula
    #   C blocks: determined by Strassen formula
    
    # Let me just hard-code the expansion.
    # For simplicity, let me define the block-level indices.
    
    # Strassen block-level:
    # M1: A blocks (0,0)+(1,1), B blocks (0,0)+(1,1)
    # M2: A blocks (1,0)+(1,1), B block (0,0)
    # M3: A block (0,0), B blocks (1,0)-(1,1)
    # M4: A block (1,1), B blocks (0,1)-(0,0)
    # M5: A blocks (0,0)+(0,1), B block (1,1)
    # M6: A blocks (1,0)-(0,0), B blocks (0,0)+(0,1)
    # M7: A blocks (0,1)-(1,1), B blocks (1,0)+(1,1)
    
    # For each block multiplication, the 2x2x2 sub-problem maps:
    #   local i, j, k in 0..1
    #   global i = block_row_A * 2 + i
    #   global j = block_col_A * 2 + j  (same as block_row_B)
    #   global k = block_col_B * 2 + k
    
    # But each block multiplication involves multiple blocks of A and B.
    # For example, M1 involves A blocks (0,0) and (1,1).
    # So for A, the global indices are:
    #   From block (0,0): i in 0,1; j in 0,1
    #   From block (1,1): i in 2,3; j in 2,3
    # The 2x2x2 sub-problem for M1 has:
    #   local A indices (i, j) -> global depends on which block
    #   This is tricky because the sub-problem treats the sum as a single 2x2 matrix.
    
    # Actually, I think the cleaner way is to just expand each of the 49 rank-1 terms.
    # Each rank-1 term corresponds to one of the 7 block-level multiplications 
    # times one of the 7 sub-multiplications within that block multiplication.
    
    # Let me define the block-level Strassen more carefully.
    # For M1 = (A00 + A11)(B00 + B11):
    #   The "A" part is a 2x2 matrix where:
    #     element (0,0) = A00[0,0] + A11[0,0]
    #     element (0,1) = A00[0,1] + A11[0,1]
    #     element (1,0) = A00[1,0] + A11[1,0]
    #     element (1,1) = A00[1,1] + A11[1,1]
    #   Similarly for B.
    
    # So for the 2x2x2 sub-problem, U[local_i * 2 + local_j, sub_rank] gives the coefficient
    # for A[local_i, local_j] in the sub-problem.
    # But A[local_i, local_j] in the sub-problem corresponds to:
    #   If local_i, local_j refer to the block-level 2x2, then:
    #   A_sub[0,0] = A00 + A11 (block-wise)
    #   So A_sub[0,0][i,j] = A00[i,j] + A11[i,j] for i,j in 0,1
    #   Global: A00[i,j] is at (i, j), A11[i,j] is at (i+2, j+2)
    
    # This is getting quite complex. Let me just build it by iterating.
    
    # For each block-level multiplication r in 0..6:
    #   Determine which A blocks and B blocks are involved, with signs.
    #   For each such block, and for each element in the block,
    #   add the contribution to U, V, W.
    
    # Block-level Strassen definitions:
    # r=0: M1: A blocks: {(0,0,+1), (1,1,+1)}, B blocks: {(0,0,+1), (1,1,+1)}
    # r=1: M2: A blocks: {(1,0,+1), (1,1,+1)}, B blocks: {(0,0,+1)}
    # r=2: M3: A blocks: {(0,0,+1)}, B blocks: {(1,0,+1), (1,1,-1)}
    # r=3: M4: A blocks: {(1,1,+1)}, B blocks: {(0,1,+1), (0,0,-1)}
    # r=4: M5: A blocks: {(0,0,+1), (0,1,+1)}, B blocks: {(1,1,+1)}
    # r=5: M6: A blocks: {(1,0,+1), (0,0,-1)}, B blocks: {(0,0,+1), (0,1,+1)}
    # r=6: M7: A blocks: {(0,1,+1), (1,1,-1)}, B blocks: {(1,0,+1), (1,1,+1)}
    
    a_blocks = [
        [(0,0,1), (1,1,1)],  # r=0
        [(1,0,1), (1,1,1)],  # r=1
        [(0,0,1)],           # r=2
        [(1,1,1)],           # r=3
        [(0,0,1), (0,1,1)],  # r=4
        [(1,0,1), (0,0,-1)], # r=5
        [(0,1,1), (1,1,-1)]  # r=6
    ]
    b_blocks = [
        [(0,0,1), (1,1,1)],  # r=0
        [(0,0,1)],           # r=1
        [(1,0,1), (1,1,-1)], # r=2
        [(0,1,1), (0,0,-1)], # r=3
        [(1,1,1)],           # r=4
        [(0,0,1), (0,1,1)],  # r=5
        [(1,0,1), (1,1,1)]   # r=6
    ]
    
    # For each block-level multiplication r, and for each sub-rank s in 0..6:
    #   The global rank is r * 7 + s
    #   U contribution: for each A block (bi, bj, sign) in a_blocks[r]:
    #     for local_i, local_j in 0..1:
    #       global_i = bi * 2 + local_i
    #       global_j = bj * 2 + local_j
    #       U[global_i * 4 + global_j, r*7+s] += sign * U2[local_i * 2 + local_j, s]
    #   V contribution: for each B block (bj, bk, sign) in b_blocks[r]:
    #     for local_j, local_k in 0..1:
    #       global_j = bj * 2 + local_j
    #       global_k = bk * 2 + local_k
    #       V[global_j * 4 + global_k, r*7+s] += sign * V2[local_j * 2 + local_k, s]
    #   W contribution: for C blocks... but C is the output, and W is the output matrix.
    #   The W matrix for the sub-problem gives the coefficients for the output.
    #   For the block-level, C blocks are determined by the Strassen formula.
    
    # Actually, W for the full 4x4x4:
    #   W[k*4+i, rank] where k is output column, i is output row.
    #   The output C is 4x4.
    #   For block-level, C blocks:
    #   C00 = M1 + M4 - M5 + M7 -> blocks (0,0)
    #   C01 = M3 + M5 -> blocks (0,1)
    #   C10 = M2 + M4 -> blocks (1,0)
    #   C11 = M1 - M2 + M3 + M6 -> blocks (1,1)
    
    c_blocks = [
        [(0,0,1), (1,1,1)],  # M1: C00 and C11
        [(1,0,1)],           # M2: C10
        [(0,1,1), (1,1,1)],  # M3: C01 and C11
        [(0,0,1), (1,0,1)],  # M4: C00 and C10
        [(0,0,-1), (0,1,1)], # M5: -C00 and C01
        [(1,1,1)],           # M6: C11
        [(0,0,1)]            # M7: C00
    ]
    
    for r in range(7):
        for s in range(7):
            global_rank = r * 7 + s
            # U
            for (bi, bj, sign) in a_blocks[r]:
                for local_i in range(2):
                    for local_j in range(2):
                        global_i = bi * 2 + local_i
                        global_j = bj * 2 + local_j
                        U[global_i * 4 + global_j, global_rank] += sign * U2[local_i * 2 + local_j, s]
            # V
            for (bj, bk, sign) in b_blocks[r]:
                for local_j in range(2):
                    for local_k in range(2):
                        global_j = bj * 2 + local_j
                        global_k = bk * 2 + local_k
                        V[global_j * 4 + global_k, global_rank] += sign * V2[local_j * 2 + local_k, s]
            # W
            for (bi, bk, sign) in c_blocks[r]:
                for local_i in range(2):
                    for local_k in range(2):
                        global_i = bi * 2 + local_i
                        global_k = bk * 2 + local_k
                        W[global_k * 4 + global_i, global_rank] += sign * W2[local_k * 2 + local_i, s]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    deadline = time.time() + 120
    
    # Split B into B1 (4x4) and B2 (4x3)
    # A*B1: 4x4x4, use recursive Strassen -> rank 49
    # A*B2: 4x4x3, standard -> rank 48
    # Total rank: 97
    
    # Part 1: 4x4x4 with recursive Strassen
    U1, V1, W1 = _recursive_strassen_4x4x4()
    
    # Part 2: 4x4x3 standard
    n2, m2, p2 = 4, 4, 3
    rank2 = n2 * m2 * p2
    U2 = np.zeros((n2 * m2, rank2), dtype=float)
    V2 = np.zeros((m2 * p2, rank2), dtype=float)
    W2 = np.zeros((p2 * n2, rank2), dtype=float)
    col = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U2[i * m2 + j, col] = 1
                V2[j * p2 + k, col] = 1
                W2[k * n2 + i, col] = 1
                col += 1
    
    # Combine:
    # U: shape (4*4, 97)
    #   First 49 columns: U1 (for B1, columns 0-3 of B)
    #   Next 48 columns: U2 (for B2, columns 4-6 of B)
    #   But U2 is for 4x4x3, so U2 has the same A indices.
    #   U_full[i*4+j, col] = U1[i*4+j, col] for col < 49
    #   U_full[i*4+j, 49+col] = U2[i*4+j, col] for col < 48
    
    # V: shape (4*7, 97)
    #   First 49 columns: V1, but V1 is for 4x4 (m=4, p=4), so V1[j*4+k, col]
    #   For the full V, j in 0..3, k in 0..6
    #   For B1 (k in 0..3): V_full[j*7+k, col] = V1[j*4+k, col] for col < 49
    #   For B2 (k in 4..6): V_full[j*7+k, 49+col] = V2[j*3+(k-4), col] for col < 48
    
    # W: shape (7*4, 97)
    #   First 49 columns: W1, W1[k*4+i, col] for k in 0..3
    #   For the full W, k in 0..6
    #   For B1 (k in 0..3): W_full[k*4+i, col] = W1[k*4+i, col] for col < 49
    #   For B2 (k in 4..6): W_full[k*4+i, 49+col] = W2[(k-4)*4+i, col] for col < 48
    
    rank_total = 49 + 48
    U = np.zeros((4 * 4, rank_total), dtype=float)
    V = np.zeros((4 * 7, rank_total), dtype=float)
    W = np.zeros((7 * 4, rank_total), dtype=float)
    
    # Fill U
    U[:, :49] = U1
    U[:, 49:] = U2
    
    # Fill V
    for j in range(4):
        for k in range(4):
            V[j * 7 + k, :49] = V1[j * 4 + k, :49]
        for k2 in range(3):
            k = 4 + k2
            V[j * 7 + k, 49:] = V2[j * 3 + k2, :]
    
    # Fill W
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, :49] = W1[k * 4 + i, :49]
    for k2 in range(3):
        k = 4 + k2
        for i in range(4):
            W[k * 4 + i, 49:] = W2[k2 * 4 + i, :]
    
    # Convert to lists of lists (integer/half-integer)
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
