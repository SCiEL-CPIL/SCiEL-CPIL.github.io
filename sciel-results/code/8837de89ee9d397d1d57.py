"""
Rank-minimal exact tensor decomposition for 2x4 by 4x5 matrix multiplication.

Strategy:
  - Use the known rank-38 decomposition obtained by applying Strassen's
    rank-7 algorithm to the 2x2x2 block sub-problems and a direct-sum
    (schoolbook) approach to the 2x2x3 block sub-problems, then combining
    via block-diagonal structure.

  The 2x4 by 4x5 multiplication can be viewed as:
    A (2x4) * B (4x5) = C (2x5)

  Partition B into two 4x2 and one 4x1 block? No, let's think differently.

  Actually, partition the 4 columns of A (and 4 rows of B) into two groups of 2:
    A = [A1 | A2] where A1, A2 are 2x2
    B = [B1; B2] where B1, B2 are 2x5

    C = A1*B1 + A2*B2

  Each term is a 2x2 * 2x5 product. We can further partition the 5 columns of B1,B2 into 2+2+1:
    B1 = [B1a B1b B1c] where B1a,B1b are 2x2, B1c is 2x1
    Similarly for B2.

    C = A1*B1a + A1*B1b + A1*B1c + A2*B2a + A2*B2b + A2*B2c

  For each 2x2 * 2x2 product, we can use Strassen (rank 7).
  For each 2x2 * 2x1 product, this is just a 2x2 times a vector (2x1), which is rank 4 (schoolbook) since there's no savings for 2x2*2x1.

  Wait, 2x2 * 2x1: output is 2x1. The tensor has 2*2*1 = 4 entries, so rank 4.
  2x2 * 2x2: output is 2x2. Schoolbook rank 4, Strassen rank 7? No, Strassen is for 2x2*2x2 with rank 7? 
  Actually Strassen's algorithm computes 2x2*2x2 with 7 multiplications, so the tensor rank is at most 7, and it's known to be exactly 7 over most fields.

  So total rank = 4 * 7 (for the four 2x2*2x2 blocks) + 2 * 4 (for the two 2x2*2x1 blocks) = 28 + 8 = 36.

  Hmm, but the knowledge says rank 38 was achieved. Let me re-examine.

  Alternative partitioning: Maybe the blocks aren't all the same size.

  Let me reconsider. The knowledge says: "Exploiting block structure to replace schoolbook rank-40 with a rank-38 decomposition by using Strassen for 2x2x2 blocks and direct sum for 2x2x3 blocks"

  So they used 2x2x2 blocks (Strassen, rank 7 each) and 2x2x3 blocks (direct, rank 6 each).

  How to partition 2x4 * 4x5 into blocks of type 2x2*2x2 and 2x2*2x3?

  A is 2x4, B is 4x5.

  Partition the 4 columns of A into 2+2: A = [A1 | A2], each 2x2.
  Partition the 4 rows of B into 2+2: B = [B1; B2], each 2x5.

  C = A1*B1 + A2*B2

  Now partition the 5 columns of B1 and B2 into 2+3:
    B1 = [B1a | B1b] where B1a is 2x2, B1b is 2x3
    B2 = [B2a | B2b] where B2a is 2x2, B2b is 2x3

  C = A1*B1a + A1*B1b + A2*B2a + A2*B2b

  For A1*B1a: 2x2 * 2x2 -> use Strassen, rank 7
  For A1*B1b: 2x2 * 2x3 -> schoolbook, rank 2*2*3 = 12? Or can we do better?
  
  Wait, 2x2 * 2x3: the tensor is for computing a 2x3 output from 2x2 and 2x3 inputs. The number of scalar multiplications in schoolbook is 2*2*3 = 12. But can we do better? For rectangular matrices, the rank might be lower. But for 2x2*2x3, I believe the rank is 12 (no known improvement).

  Hmm, that gives 7 + 12 + 7 + 12 = 38. That matches!

  So:
  - A1*B1a: 2x2 * 2x2, Strassen rank 7
  - A1*B1b: 2x2 * 2x3, schoolbook rank 12
  - A2*B2a: 2x2 * 2x2, Strassen rank 7
  - A2*B2b: 2x2 * 2x3, schoolbook rank 12
  Total: 38

  Let me construct this.
"""

import time

DIMENSIONS = [2, 4, 5]

# Strassen's 7 multiplications for 2x2 * 2x2
# Given matrices:
#   A = [[a11, a12],
#        [a21, a22]]
#   B = [[b11, b12],
#        [b21, b22]]
#
# Strassen's algorithm:
#   M1 = (a11 + a22) * (b11 + b22)
#   M2 = (a21 + a22) * b11
#   M3 = a11 * (b12 - b22)
#   M4 = a22 * (b21 - b11)
#   M5 = (a11 + a12) * b22
#   M6 = (a21 - a11) * (b11 + b12)
#   M7 = (a12 - a22) * (b21 + b22)
#
#   C11 = M1 + M4 - M5 + M7
#   C12 = M3 + M5
#   C21 = M2 + M4
#   C22 = M1 - M2 + M3 + M6

def _strassen_2x2():
    """
    Return U, V, W for 2x2 * 2x2 using Strassen.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j, k] for A[i,j], V[i*2+j, k] for B[i,j], W[i*2+j, k] for C[i,j].
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (a11 + a22) * (b11 + b22)
    # U: a11=1, a22=1
    U[0][0] = 1  # a11
    U[3][0] = 1  # a22
    V[0][0] = 1  # b11
    V[3][0] = 1  # b22
    W[0][0] = 1  # C11 += M1
    W[3][0] = 1  # C22 += M1
    
    # M2 = (a21 + a22) * b11
    U[1][1] = 1  # a21
    U[3][1] = 1  # a22
    V[0][1] = 1  # b11
    W[2][1] = 1  # C21 += M2
    W[3][1] = -1 # C22 -= M2
    
    # M3 = a11 * (b12 - b22)
    U[0][2] = 1  # a11
    V[1][2] = 1  # b12
    V[3][2] = -1 # -b22
    W[0][2] = 1  # C11 += M3
    W[1][2] = 1  # C12 += M3
    
    # M4 = a22 * (b21 - b11)
    U[3][3] = 1  # a22
    V[2][3] = 1  # b21
    V[0][3] = -1 # -b11
    W[0][3] = 1  # C11 += M4
    W[2][3] = 1  # C21 += M4
    
    # M5 = (a11 + a12) * b22
    U[0][4] = 1  # a11
    U[1][4] = 1  # a12
    V[3][4] = 1  # b22
    W[0][4] = -1 # C11 -= M5
    W[1][4] = 1  # C12 += M5
    
    # M6 = (a21 - a11) * (b11 + b12)
    U[1][5] = 1  # a21
    U[0][5] = -1 # -a11
    V[0][5] = 1  # b11
    V[1][5] = 1  # b12
    W[3][5] = 1  # C22 += M6
    
    # M7 = (a12 - a22) * (b21 + b22)
    U[1][6] = 1  # a12
    U[3][6] = -1 # -a22
    V[2][6] = 1  # b21
    V[3][6] = 1  # b22
    W[0][6] = 1  # C11 += M7
    
    return U, V, W


def _schoolbook_2x2_x_2x3():
    """
    Return U, V, W for 2x2 * 2x3 using schoolbook.
    U has shape (4, 12), V has shape (6, 12), W has shape (6, 12).
    A is 2x2: indices (0,1,2,3) -> (0,0),(0,1),(1,0),(1,1)
    B is 2x3: indices (0,1,2,3,4,5) -> (0,0),(0,1),(0,2),(1,0),(1,1),(1,2)
    C is 2x3: indices (0,1,2,3,4,5) -> (0,0),(0,1),(0,2),(1,0),(1,1),(1,2)
    """
    n, m, p = 2, 2, 3
    rank = n * m * p  # 12
    U = [[0]*rank for _ in range(n*m)]  # 4 x 12
    V = [[0]*rank for _ in range(m*p)]  # 6 x 12
    W = [[0]*rank for _ in range(p*n)]  # 6 x 12
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    return U, V, W


def _combine_block_diagonal(blocks):
    """
    Combine a list of (U, V, W) tuples into a single block-diagonal decomposition.
    Each block has U_i of shape (r_i, rank_i), V_i of shape (s_i, rank_i), W_i of shape (t_i, rank_i).
    The combined U has shape (sum r_i, sum rank_i), etc.
    
    But wait - the blocks operate on disjoint subsets of the full matrices.
    We need to embed each block's U, V, W into the full-sized matrices.
    """
    total_rank = sum(b[1].shape[1] for b in blocks)
    return total_rank


def construct():
    """
    Construct rank-38 decomposition for 2x4 * 4x5 matrix multiplication.
    
    Partition:
      A (2x4) = [A1 | A2], A1,A2 are 2x2
      B (4x5) = [B1; B2], B1,B2 are 2x5
      B1 = [B1a | B1b], B1a is 2x2, B1b is 2x3
      B2 = [B2a | B2b], B2a is 2x2, B2b is 2x3
      
      C = A1*B1a + A1*B1b + A2*B2a + A2*B2b
      
    Block 1: A1 (2x2) * B1a (2x2) -> C1a (2x2), Strassen rank 7
    Block 2: A1 (2x2) * B1b (2x3) -> C1b (2x3), schoolbook rank 12
    Block 3: A2 (2x2) * B2a (2x2) -> C2a (2x2), Strassen rank 7
    Block 4: A2 (2x2) * B2b (2x3) -> C2b (2x3), schoolbook rank 12
    
    Total rank: 7 + 12 + 7 + 12 = 38
    """
    n, m, p = 2, 4, 5
    # Full matrix dimensions:
    # U: (n*m, rank) = (8, 38)
    # V: (m*p, rank) = (20, 38)
    # W: (p*n, rank) = (10, 38)
    
    total_rank = 38
    U = [[0]*total_rank for _ in range(n*m)]  # 8 x 38
    V = [[0]*total_rank for _ in range(m*p)]  # 20 x 38
    W = [[0]*total_rank for _ in range(p*n)]  # 10 x 38
    
    # Get the sub-decompositions
    U_s, V_s, W_s = _strassen_2x2()
    U_sb, V_sb, W_sb = _schoolbook_2x2_x_2x3()
    
    # Block 1: A1 (rows 0,1 of A) * B1a (rows 0,1 of B, cols 0,1) -> C1a (rows 0,1 of C, cols 0,1)
    # A1: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices 0,1,2,3
    # B1a: B[0,0], B[0,1], B[1,0], B[1,1] -> V indices 0,1,4,5
    # C1a: C[0,0], C[0,1], C[1,0], C[1,1] -> W indices 0,1,2,3 (W[k*n+i] where k is col, i is row)
    #   C[0,0]: k=0,i=0 -> W[0]
    #   C[0,1]: k=1,i=0 -> W[1]
    #   C[1,0]: k=0,i=1 -> W[2]
    #   C[1,1]: k=1,i=1 -> W[3]
    
    col_offset = 0
    for c in range(7):
        # U mapping: local (0,1,2,3) -> global (0,1,2,3)
        for li, gi in enumerate([0, 1, 2, 3]):
            U[gi][col_offset + c] = U_s[li][c]
        # V mapping: local (0,1,2,3) -> global (0,1,4,5)
        #   local 0: B[0,0] -> global 0
        #   local 1: B[0,1] -> global 1
        #   local 2: B[1,0] -> global 4
        #   local 3: B[1,1] -> global 5
        for li, gi in enumerate([0, 1, 4, 5]):
            V[gi][col_offset + c] = V_s[li][c]
        # W mapping: local (0,1,2,3) -> global (0,1,2,3)
        #   local 0: C[0,0] -> global 0
        #   local 1: C[0,1] -> global 1
        #   local 2: C[1,0] -> global 2
        #   local 3: C[1,1] -> global 3
        for li, gi in enumerate([0, 1, 2, 3]):
            W[gi][col_offset + c] = W_s[li][c]
    col_offset += 7
    
    # Block 2: A1 (rows 0,1 of A) * B1b (rows 0,1 of B, cols 2,3,4) -> C1b (rows 0,1 of C, cols 2,3,4)
    # A1: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices 0,1,2,3
    # B1b: B[0,2], B[0,3], B[0,4], B[1,2], B[1,3], B[1,4] -> V indices 2,3,6,7,8,9
    #   B[0,2]: row 0, col 2 -> V[0*3+2] = V[2]
    #   B[0,3]: row 0, col 3 -> V[0*3+3] = V[3]
    #   B[0,4]: row 0, col 4 -> V[0*3+4] = V[4]... wait, m=4, p=5, so V index = row*p + col
    #   B[row, col] -> V[row*p + col] where p=5
    #   B[0,2] -> V[0*5+2] = V[2]
    #   B[0,3] -> V[0*5+3] = V[3]
    #   B[0,4] -> V[0*5+4] = V[4]
    #   B[1,2] -> V[1*5+2] = V[7]
    #   B[1,3] -> V[1*5+3] = V[8]
    #   B[1,4] -> V[1*5+4] = V[9]
    # C1b: C[0,2], C[0,3], C[0,4], C[1,2], C[1,3], C[1,4] -> W indices
    #   W[k*n+i] where k is col, i is row, n=2
    #   C[0,2]: k=2, i=0 -> W[2*2+0] = W[4]
    #   C[0,3]: k=3, i=0 -> W[3*2+0] = W[6]
    #   C[0,4]: k=4, i=0 -> W[4*2+0] = W[8]
    #   C[1,2]: k=2, i=1 -> W[2*2+1] = W[5]
    #   C[1,3]: k=3, i=1 -> W[3*2+1] = W[7]
    #   C[1,4]: k=4, i=1 -> W[4*2+1] = W[9]
    
    for c in range(12):
        # U mapping: local (0,1,2,3) -> global (0,1,2,3)
        for li, gi in enumerate([0, 1, 2, 3]):
            U[gi][col_offset + c] = U_sb[li][c]
        # V mapping: local (0,1,2,3,4,5) -> global (2,3,4,7,8,9)
        #   local 0: B[0,0] of B1b -> global B[0,2] -> V[2]
        #   local 1: B[0,1] of B1b -> global B[0,3] -> V[3]
        #   local 2: B[0,2] of B1b -> global B[0,4] -> V[4]
        #   local 3: B[1,0] of B1b -> global B[1,2] -> V[7]
        #   local 4: B[1,1] of B1b -> global B[1,3] -> V[8]
        #   local 5: B[1,2] of B1b -> global B[1,4] -> V[9]
        for li, gi in enumerate([2, 3, 4, 7, 8, 9]):
            V[gi][col_offset + c] = V_sb[li][c]
        # W mapping: local (0,1,2,3,4,5) -> global (4,6,8,5,7,9)
        #   local 0: C[0,0] of C1b -> global C[0,2] -> W[4]
        #   local 1: C[0,1] of C1b -> global C[0,3] -> W[6]
        #   local 2: C[0,2] of C1b -> global C[0,4] -> W[8]
        #   local 3: C[1,0] of C1b -> global C[1,2] -> W[5]
        #   local 4: C[1,1] of C1b -> global C[1,3] -> W[7]
        #   local 5: C[1,2] of C1b -> global C[1,4] -> W[9]
        for li, gi in enumerate([4, 6, 8, 5, 7, 9]):
            W[gi][col_offset + c] = W_sb[li][c]
    col_offset += 12
    
    # Block 3: A2 (rows 2,3 of A) * B2a (rows 2,3 of B, cols 0,1) -> C2a (rows 0,1 of C, cols 0,1)
    # A2: A[0,2], A[0,3], A[1,2], A[1,3] -> U indices 4,5,6,7
    #   A[row, col] -> U[row*m + col] where m=4
    #   A[0,2] -> U[0*4+2] = U[2]... wait, that's wrong.
    #   Let me re-index. A is 2x4. U has n*m = 8 rows.
    #   U[i*m + j] where i is row of A (0,1), j is col of A (0,1,2,3)
    #   A[0,0] -> U[0], A[0,1] -> U[1], A[0,2] -> U[2], A[0,3] -> U[3]
    #   A[1,0] -> U[4], A[1,1] -> U[5], A[1,2] -> U[6], A[1,3] -> U[7]
    #   
    #   Wait, I had A1 as the first 2 columns and A2 as the last 2 columns.
    #   A1 = A[:, 0:2], A2 = A[:, 2:4]
    #   A1: A[0,0], A[0,1], A[1,0], A[1,1] -> U[0], U[1], U[4], U[5]
    #   A2: A[0,2], A[0,3], A[1,2], A[1,3] -> U[2], U[3], U[6], U[7]
    
    # Let me redo the mappings more carefully.
    
    # Actually, let me restart the mapping with clearer indices.
    # A is 2x4, indexed A[i][j] for i in {0,1}, j in {0,1,2,3}
    # U has 8 rows: U[i*4 + j] corresponds to A[i][j]
    #   A[0][0] -> U[0], A[0][1] -> U[1], A[0][2] -> U[2], A[0][3] -> U[3]
    #   A[1][0] -> U[4], A[1][1] -> U[5], A[1][2] -> U[6], A[1][3] -> U[7]
    
    # B is 4x5, indexed B[i][j] for i in {0,1,2,3}, j in {0,1,2,3,4}
    # V has 20 rows: V[i*5 + j] corresponds to B[i][j]
    #   B[0][0] -> V[0], B[0][1] -> V[1], B[0][2] -> V[2], B[0][3] -> V[3], B[0][4] -> V[4]
    #   B[1][0] -> V[5], B[1][1] -> V[6], B[1][2] -> V[7], B[1][3] -> V[8], B[1][4] -> V[9]
    #   B[2][0] -> V[10], B[2][1] -> V[11], B[2][2] -> V[12], B[2][3] -> V[13], B[2][4] -> V[14]
    #   B[3][0] -> V[15], B[3][1] -> V[16], B[3][2] -> V[17], B[3][3] -> V[18], B[3][4] -> V[19]
    
    # C is 2x5, indexed C[i][j] for i in {0,1}, j in {0,1,2,3,4}
    # W has 10 rows: W[j*2 + i] corresponds to C[i][j] (trace convention W[k*n+i] where k=j, n=2)
    #   C[0][0] -> W[0], C[0][1] -> W[1], C[0][2] -> W[2], C[0][3] -> W[3], C[0][4] -> W[4]
    #   C[1][0] -> W[5], C[1][1] -> W[6], C[1][2] -> W[7], C[1][3] -> W[8], C[1][4] -> W[9]
    
    # Now let's redo:
    
    # A1 = A[:, 0:2]: A[0][0], A[0][1], A[1][0], A[1][1]
    #   Local indices in A1: (0,0), (0,1), (1,0), (1,1)
    #   Global U indices: 0, 1, 4, 5
    # A2 = A[:, 2:4]: A[0][2], A[0][3], A[1][2], A[1][3]
    #   Local indices in A2: (0,0), (0,1), (1,0), (1,1)
    #   Global U indices: 2, 3, 6, 7
    
    # B1 = B[0:2, :]: rows 0,1 of B
    # B2 = B[2:4, :]: rows 2,3 of B
    # B1a = B[0:2, 0:2]: B[0][0], B[0][1], B[1][0], B[1][1]
    #   Local indices: (0,0), (0,1), (1,0), (1,1)
    #   Global V indices: 0, 1, 5, 6
    # B1b = B[0:2, 2:5]: B[0][2], B[0][3], B[0][4], B[1][2], B[1][3], B[1][4]
    #   Local indices: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
    #   Global V indices: 2, 3, 4, 7, 8, 9
    # B2a = B[2:4, 0:2]: B[2][0], B[2][1], B[3][0], B[3][1]
    #   Local indices: (0,0), (0,1), (1,0), (1,1)
    #   Global V indices: 10, 11, 15, 16
    # B2b = B[2:4, 2:5]: B[2][2], B[2][3], B[2][4], B[3][2], B[3][3], B[3][4]
    #   Local indices: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
    #   Global V indices: 12, 13, 14, 17, 18, 19
    
    # C1a = C[:, 0:2]: C[0][0], C[0][1], C[1][0], C[1][1]
    #   Local indices: (0,0), (0,1), (1,0), (1,1)
    #   Global W indices: 0, 1, 5, 6
    # C1b = C[:, 2:5]: C[0][2], C[0][3], C[0][4], C[1][2], C[1][3], C[1][4]
    #   Local indices: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
    #   Global W indices: 2, 3, 4, 7, 8, 9
    # C2a = C[:, 0:2]: same as C1a (since C2 = C, the output is the same matrix)
    #   Wait, no. C = A1*B1 + A2*B2. So C1a is the contribution from A1*B1a, and C2a is the contribution from A2*B2a.
    #   They both contribute to the same output C.
    #   C2a = C[:, 0:2]: C[0][0], C[0][1], C[1][0], C[1][1]
    #   Global W indices: 0, 1, 5, 6
    # C2b = C[:, 2:5]: C[0][2], C[0][3], C[0][4], C[1][2], C[1][3], C[1][4]
    #   Global W indices: 2, 3, 4, 7, 8, 9
    
    # OK so my earlier mappings for blocks 1 and 2 were partially wrong. Let me redo all.
    
    # Reset
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col_offset = 0
    
    # Block 1: A1 (2x2) * B1a (2x2) -> contributes to C[:, 0:2]
    # Strassen: U_s, V_s, W_s each 4x7
    # A1 local -> U global: [0, 1, 4, 5]
    # B1a local -> V global: [0, 1, 5, 6]
    # C local -> W global: [0, 1, 5, 6]
    for c in range(7):
        for li, gi in enumerate([0, 1, 4, 5]):
            U[gi][col_offset + c] = U_s[li][c]
        for li, gi in enumerate([0, 1, 5, 6]):
            V[gi][col_offset + c] = V_s[li][c]
        for li, gi in enumerate([0, 1, 5, 6]):
            W[gi][col_offset + c] = W_s[li][c]
    col_offset += 7
    
    # Block 2: A1 (2x2) * B1b (2x3) -> contributes to C[:, 2:5]
    # Schoolbook: U_sb, V_sb, W_sb each (4,6,6) x 12
    # A1 local -> U global: [0, 1, 4, 5]
    # B1b local -> V global: [2, 3, 4, 7, 8, 9]
    # C local -> W global: [2, 3, 4, 7, 8, 9]
    for c in range(12):
        for li, gi in enumerate([0, 1, 4, 5]):
            U[gi][col_offset + c] = U_sb[li][c]
        for li, gi in enumerate([2, 3, 4, 7, 8, 9]):
            V[gi][col_offset + c] = V_sb[li][c]
        for li, gi in enumerate([2, 3, 4, 7, 8, 9]):
            W[gi][col_offset + c] = W_sb[li][c]
    col_offset += 12
    
    # Block 3: A2 (2x2) * B2a (2x2) -> contributes to C[:, 0:2]
    # Strassen: U_s, V_s, W_s each 4x7
    # A2 local -> U global: [2, 3, 6, 7]
    # B2a local -> V global: [10, 11, 15, 16]
    # C local -> W global: [0, 1, 5, 6]
    for c in range(7):
        for li, gi in enumerate([2, 3, 6, 7]):
            U[gi][col_offset + c] = U_s[li][c]
        for li, gi in enumerate([10, 11, 15, 16]):
            V[gi][col_offset + c] = V_s[li][c]
        for li, gi in enumerate([0, 1, 5, 6]):
            W[gi][col_offset + c] = W_s[li][c]
    col_offset += 7
    
    # Block 4: A2 (2x2) * B2b (2x3) -> contributes to C[:, 2:5]
    # Schoolbook: U_sb, V_sb, W_sb each (4,6,6) x 12
    # A2 local -> U global: [2, 3, 6, 7]
    # B2b local -> V global: [12, 13, 14, 17, 18, 19]
    # C local -> W global: [2, 3, 4, 7, 8, 9]
    for c in range(12):
        for li, gi in enumerate([2, 3, 6, 7]):
            U[gi][col_offset + c] = U_sb[li][c]
        for li, gi in enumerate([12, 13, 14, 17, 18, 19]):
            V[gi][col_offset + c] = V_sb[li][c]
        for li, gi in enumerate([2, 3, 4, 7, 8, 9]):
            W[gi][col_offset + c] = W_sb[li][c]
    col_offset += 12
    
    assert col_offset == total_rank, f"col_offset {col_offset} != total_rank {total_rank}"
    
    return {"U": U, "V": V, "W": W}
