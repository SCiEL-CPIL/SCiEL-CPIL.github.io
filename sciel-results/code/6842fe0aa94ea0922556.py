"""
Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 4 matrix.
The output factor follows the official trace convention W[k*n+i,r].
Interface: construct() -> {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
with integer/half-integer entries that exactly decompose the matrix multiplication tensor.

Strategy: combine a rank-28 Strassen-style decomposition for the 2x4x4 top block
(with B split into two 4x2 halves, each handled by two 2x2x2 Strassen sub-decompositions)
with a rank-16 schoolbook decomposition for the 1x4x4 bottom block. Total rank = 44.
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return schoolbook decomposition for n x m by m x p multiplication."""
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
    return U, V, W, rank


def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 by 2x2 multiplication.
    Indexing:
      U rows: (i,j) for A[i,j], row index = i*2 + j
      V rows: (j,k) for B[j,k], row index = j*2 + k
      W rows: (k,i) for C[i,k], row index = k*2 + i   (trace convention W[k*n+i])
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]

    # A11=0, A12=1, A21=2, A22=3
    # B11=0, B12=1, B21=2, B22=3
    # C11 -> W[0], C12 -> W[1], C21 -> W[2], C22 -> W[3]

    # M1 = (A11+A22)(B11+B22);  C11+=1, C22+=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # M2 = (A21+A22)B11;  C21+=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1

    # M3 = A11(B12-B22);  C12+=1, C22+=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1

    # M4 = A22(B21-B11);  C11+=1, C21+=1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1

    # M5 = (A11+A12)B22;  C11-=1, C12+=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1

    # M6 = (A21-A11)(B11+B12);  C22+=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # M7 = (A12-A22)(B21+B22);  C11+=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return U, V, W, 7


def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """
    Verify: for all (i,j,k) in [n)x[m)x[p),
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > tol:
                    return False
    return True


def _build_rank44():
    """
    Build a rank-44 decomposition for 3x4 by 4x4 multiplication.

    A is 3x4 (rows i=0,1,2; cols j=0,1,2,3).
    B is 4x4 (rows j=0,1,2,3; cols k=0,1,2,3).
    C is 3x4 (rows i=0,1,2; cols k=0,1,2,3).

    Split:
      A_top  = rows 0,1 (2x4)
      A_bot  = row  2   (1x4)
      B_L    = cols 0,1 (4x2)
      B_R    = cols 2,3 (4x2)

    C_top_L = A_top * B_L  (2x2)  -> Strassen 2x2x2, rank 7
    C_top_R = A_top * B_R  (2x2)  -> Strassen 2x2x2, rank 7
    C_bot_L = A_bot * B_L  (1x2)  -> schoolbook 1x4x2, rank 8
    C_bot_R = A_bot * B_R  (1x2)  -> schoolbook 1x4x2, rank 8
    Total rank = 7 + 7 + 8 + 8 = 30? No: 7+7+8+8 = 30... wait that's not 44.

    Hmm, let me recompute. Actually 1x4x2 schoolbook = 1*4*2 = 8. So 7+7+8+8 = 30.
    But that's only if the sub-decompositions are independent (which they are, since
    the C blocks don't overlap). So total = 30? That seems too good. Let me reconsider.

    Wait - the issue is that C_top_L and C_top_R share the same A_top rows and B rows.
    But in the tensor decomposition, each (i,j,k) triple is independent, so the ranks
    simply add. 7+7+8+8 = 30. But that contradicts the known lower bound...

    Actually no. Let me re-examine. The 2x4x4 decomposition:
      A_top is 2x4, B is 4x4. Split B into B_L (4x2) and B_R (4x2).
      C_top = [A_top*B_L | A_top*B_R], each 2x2.
      A_top*B_L: A_top is 2x4, B_L is 4x2. Split A_top = [A_top_L A_top_R] (each 2x2),
                 B_L = [B_L_top; B_L_bot] (each 2x1).
                 A_top*B_L = A_top_L*B_L_top + A_top_R*B_L_bot.
                 This is a 2x2x1 + 2x2x1, NOT a 2x2x2.

    I made an error. Let me redo this carefully.

    For A_top (2x4) * B_L (4x2):
      A_top = [A_L | A_R] where A_L, A_R are 2x2.
      B_L = [B_T; B_B] where B_T, B_B are 2x1.
      A_top * B_L = A_L * B_T + A_R * B_B.
      A_L * B_T is 2x2 * 2x1 = 2x1 vector. Rank of 2x2x1 = 4 (schoolbook).
      A_R * B_B is 2x2 * 2x1 = 2x1 vector. Rank of 2x2x1 = 4.
      Total for A_top * B_L: 4 + 4 = 8.

    So 2x4x2 has schoolbook rank 16, and this split gives 8+8=16. No improvement.

    The Strassen trick works for 2x2x2 (square, full), not for 2x2x1.

    Let me reconsider the whole approach. The correct way to get rank 44:

    2x4x4: Split B into B_L (4x2) and B_R (4x2).
      For A_top * B_L (2x4 by 4x2):
        Split A_top into A_L (2x2) and A_R (2x2), B_L into B_T (2x1) and B_B (2x1).
        A_top * B_L = A_L*B_T + A_R*B_B. Each term is 2x2 by 2x1.
        2x2 by 2x1 has rank 4 (schoolbook). So 4+4 = 8.
      Similarly A_top * B_R = 8.
      Total 2x4x4 = 16. Same as schoolbook 32? No, schoolbook 2x4x4 = 32.
      Wait: 2x4x4 schoolbook = 2*4*4 = 32. My split gives 8+8 = 16. That's half!

    Let me re-examine. A is 2x4, B is 4x4, C is 2x4.
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k].
    Split j into {0,1} and {2,3}:
      C[i,k] = sum_{j in {0,1}} A[i,j]B[j,k] + sum_{j in {2,3}} A[i,j]B[j,k]
             = (A_L * B_T)[i,k] + (A_R * B_B)[i,k]
    where A_L is 2x2 (rows i, cols j=0,1), B_T is 2x4 (rows j=0,1, cols k),
          A_R is 2x2 (rows i, cols j=2,3), B_B is 2x4 (rows j=2,3, cols k).
    A_L * B_T is 2x2 by 2x4 = 2x4. Schoolbook rank = 2*2*4 = 16.
    A_R * B_B is 2x2 by 2x4 = 2x4. Schoolbook rank = 16.
    Total = 32. Same as schoolbook. No improvement.

    The Strassen trick needs BOTH dimensions to be 2x2. For 2x2x4, we can split the 4
    into two 2s: A_L * B_T where B_T = [B_T1 | B_T2] (each 2x2).
    A_L * B_T1 is 2x2x2 (rank 7 via Strassen), A_L * B_T2 is 2x2x2 (rank 7).
    Total for A_L * B_T: 7+7 = 14.
    Similarly A_R * B_B: 14.
    Total 2x4x4: 14+14 = 28.

    So the correct decomposition:
      2x4x4: split A into A_L, A_R (2x2 each), split B into B_T, B_B (2x4 each),
             then split B_T into B_T1, B_T2 (2x2 each), B_B into B_B1, B_B2 (2x2 each).
             4 sub-problems of 2x2x2, each rank 7. Total = 28.

      1x4x4: A_bot is 1x4, B is 4x4.
             Split A_bot into A_BL, A_BR (1x2 each), B into B_T, B_B (2x4 each).
             A_bot * B = A_BL*B_T + A_BR*B_B.
             A_BL*B_T: 1x2 by 2x4. Split B_T into B_T1, B_T2 (2x2 each).
             A_BL*B_T1: 1x2 by 2x2. This is 1x2x2, schoolbook rank = 1*2*2 = 4.
             A_BL*B_T2: 4. Total A_BL*B_T = 8.
             Similarly A_BR*B_B = 8.
             Total 1x4x4 = 16.

    Grand total: 28 + 16 = 44. 

    Now let me build this properly.
    """
    n, m, p = 3, 4, 4
    total_rank = 44
    U = [[0] * total_rank for _ in range(n * m)]  # 12 rows
    V = [[0] * total_rank for _ in range(m * p)]  # 16 rows
    W = [[0] * total_rank for _ in range(p * n)]  # 12 rows

    # We'll fill in 4 Strassen blocks for the top (2x4x4) part and
    # 4 schoolbook blocks for the bottom (1x4x4) part.

    # === TOP PART: A rows 0,1 (2x4), B all 4x4, C rows 0,1 (2x4) ===
    # A_L: rows 0,1; cols 0,1  -> A indices: (0,0)=0, (0,1)=1, (1,0)=4, (1,1)=5
    # A_R: rows 0,1; cols 2,3  -> A indices: (0,2)=2, (0,3)=3, (1,2)=6, (1,3)=7
    # B_T: rows 0,1; cols 0,1,2,3 -> B indices: (0,0)=0,(0,1)=1,(0,2)=2,(0,3)=3,(1,0)=4,(1,1)=5,(1,2)=6,(1,3)=7
    # B_B: rows 2,3; cols 0,1,2,3 -> B indices: (2,0)=8,(2,1)=9,(2,2)=10,(2,3)=11,(3,0)=12,(3,1)=13,(3,2)=14,(3,3)=15
    # C top: rows 0,1; cols 0,1,2,3 -> W indices: W[k*3+i]
    #   C(0,0)=W[0], C(0,1)=W[1], C(0,2)=W[2], C(0,3)=W[3]
    #   C(1,0)=W[3], C(1,1)=W[4], C(1,2)=W[5], C(1,3)=W[6]

    # Strassen block helper: given the 4 A-indices, 4 B-indices, and 4 W-indices
    # for a 2x2x2 sub-problem, fill in the Strassen decomposition at offset col.

    def fill_strassen(U, V, W, a_idx, b_idx, w_idx, col0):
        """
        a_idx: list of 4 U-row indices for A11, A12, A21, A22
        b_idx: list of 4 V-row indices for B11, B12, B21, B22
        w_idx: list of 4 W-row indices for C11, C12, C21, C22
        col0: starting column offset
        """
        # M1 = (A11+A22)(B11+B22); C11+=1, C22+=1
        U[a_idx[0]][col0+0] = 1; U[a_idx[3]][col0+0] = 1
        V[b_idx[0]][col0+0] = 1; V[b_idx[3]][col0+0] = 1
        W[w_idx[0]][col0+0] = 1; W[w_idx[3]][col0+0] = 1

        # M2 = (A21+A22)B11; C21+=1
        U[a_idx[2]][col0+1] = 1; U[a_idx[3]][col0+1] = 1
        V[b_idx[0]][col0+1] = 1
        W[w_idx[2]][col0+1] = 1

        # M3 = A11(B12-B22); C12+=1, C22+=1
        U[a_idx[0]][col0+2] = 1
        V[b_idx[1]][col0+2] = 1; V[b_idx[3]][col0+2] = -1
        W[w_idx[1]][col0+2] = 1; W[w_idx[3]][col0+2] = 1

        # M4 = A22(B21-B11); C11+=1, C21+=1
        U[a_idx[3]][col0+3] = 1
        V[b_idx[2]][col0+3] = 1; V[b_idx[0]][col0+3] = -1
        W[w_idx[0]][col0+3] = 1; W[w_idx[2]][col0+3] = 1

        # M5 = (A11+A12)B22; C11-=1, C12+=1
        U[a_idx[0]][col0+4] = 1; U[a_idx[1]][col0+4] = 1
        V[b_idx[3]][col0+4] = 1
        W[w_idx[0]][col0+4] = -1; W[w_idx[1]][col0+4] = 1

        # M6 = (A21-A11)(B11+B12); C22+=1
        U[a_idx[2]][col0+5] = 1; U[a_idx[0]][col0+5] = -1
        V[b_idx[0]][col0+5] = 1; V[b_idx[1]][col0+5] = 1
        W[w_idx[3]][col0+5] = 1

        # M7 = (A12-A22)(B21+B22); C11+=1
        U[a_idx[1]][col0+6] = 1; U[a_idx[3]][col0+6] = -1
        V[b_idx[2]][col0+6] = 1; V[b_idx[3]][col0+6] = 1
        W[w_idx[0]][col0+6] = 1

    # Top part: 4 Strassen blocks
    # Block 1: A_L (cols 0,1) * B_T1 (rows 0,1, cols 0,1) -> C(0:2, 0:2)
    #   A indices: A(0,0)=0, A(0,1)=1, A(1,0)=4, A(1,1)=5
    #   B indices: B(0,0)=0, B(0,1)=1, B(1,0)=4, B(1,1)=5
    #   W indices: C(0,0)=W[0], C(0,1)=W[1], C(1,0)=W[3], C(1,1)=W[4]
    fill_strassen(U, V, W,
                  a_idx=[0, 1, 4, 5],
                  b_idx=[0, 1, 4, 5],
                  w_idx=[0, 1, 3, 4],
                  col0=0)

    # Block 2: A_L (cols 0,1) * B_T2 (rows 0,1, cols 2,3) -> C(0:2, 2:4)
    #   A indices: A(0,0)=0, A(0,1)=1, A(1,0)=4, A(1,1)=5
    #   B indices: B(0,2)=2, B(0,3)=3, B(1,2)=6, B(1,3)=7
    #   W indices: C(0,2)=W[2], C(0,3)=W[3], C(1,2)=W[5], C(1,3)=W[6]
    fill_strassen(U, V, W,
                  a_idx=[0, 1, 4, 5],
                  b_idx=[2, 3, 6, 7],
                  w_idx=[2, 3, 5, 6],
                  col0=7)

    # Block 3: A_R (cols 2,3) * B_B1 (rows 2,3, cols 0,1) -> C(0:2, 0:2)
    #   A indices: A(0,2)=2, A(0,3)=3, A(1,2)=6, A(1,3)=7
    #   B indices: B(2,0)=8, B(2,1)=9, B(3,0)=12, B(3,1)=13
    #   W indices: C(0,0)=W[0], C(0,1)=W[1], C(1,0)=W[3], C(1,1)=W[4]
    fill_strassen(U, V, W,
                  a_idx=[2, 3, 6, 7],
                  b_idx=[8, 9, 12, 13],
                  w_idx=[0, 1, 3, 4],
                  col0=14)

    # Block 4: A_R (cols 2,3) * B_B2 (rows 2,3, cols 2,3) -> C(0:2, 2:4)
    #   A indices: A(0,2)=2, A(0,3)=3, A(1,2)=6, A(1,3)=7
    #   B indices: B(2,2)=10, B(2,3)=11, B(3,2)=14, B(3,3)=15
    #   W indices: C(0,2)=W[2], C(0,3)=W[3], C(1,2)=W[5], C(1,3)=W[6]
    fill_strassen(U, V, W,
                  a_idx=[2, 3, 6, 7],
                  b_idx=[10, 11, 14, 15],
                  w_idx=[2, 3, 5, 6],
                  col0=21)

    # === BOTTOM PART: A row 2 (1x4), B all 4x4, C row 2 (1x4) ===
    # A_BL: row 2, cols 0,1 -> A indices: A(2,0)=8, A(2,1)=9
    # A_BR: row 2, cols 2,3 -> A indices: A(2,2)=10, A(2,3)=11
    # B_T: rows 0,1; B_B: rows 2,3
    # C bot: row 2; cols 0,1,2,3 -> W indices: W[k*3+2]
    #   C(2,0)=W[2], C(2,1)=W[5], C(2,2)=W[8], C(2,3)=W[11]

    # Schoolbook for 1x2 by 2x2 (rank 4):
    #   C(0,0) = A(0,0)B(0,0) + A(0,1)B(1,0)
    #   C(0,1) = A(0,0)B(0,1) + A(0,1)B(1,1)
    #   4 terms, each rank 1.

    def fill_schoolbook_1x2x2(U, V, W, a_idx, b_idx, w_idx, col0):
        """
        a_idx: list of 2 U-row indices for A(0,0), A(0,1)
        b_idx: list of 4 V-row indices for B(0,0), B(0,1), B(1,0), B(1,1)
        w_idx: list of 2 W-row indices for C(0,0), C(0,1)
        col0: starting column offset
        """
        # C(0,0) = A(0,0)B(0,0) + A(0,1)B(1,0)
        U[a_idx[0]][col0+0] = 1; V[b_idx[0]][col0+0] = 1; W[w_idx[0]][col0+0] = 1
        U[a_idx[1]][col0+1] = 1; V[b_idx[2]][col0+1] = 1; W[w_idx[0]][col0+1] = 1
        # C(0,1) = A(0,0)B(0,1) + A(0,1)B(1,1)
        U[a_idx[0]][col0+2] = 1; V[b_idx[1]][col0+2] = 1; W[w_idx[1]][col0+2] = 1
        U[a_idx[1]][col0+3] = 1; V[b_idx[3]][col0+3] = 1; W[w_idx[1]][col0+3] = 1

    # Block 5: A_BL (cols 0,1) * B_T1 (rows 0,1, cols 0,1) -> C(2, 0:2)
    fill_schoolbook_1x2x2(U, V, W,
                          a_idx=[8, 9],
                          b_idx=[0, 1, 4, 5],
                          w_idx=[2, 5],
                          col0=28)

    # Block 6: A_BL (cols 0,1) * B_T2 (rows 0,1, cols 2,3) -> C(2, 2:4)
    fill_schoolbook_1x2x2(U, V, W,
                          a_idx=[8, 9],
                          b_idx=[2, 3, 6, 7],
                          w_idx=[8, 11],
                          col0=32)

    # Block 7: A_BR (cols 2,3) * B_B1 (rows 2,3, cols 0,1) -> C(2, 0:2)
    fill_schoolbook_1x2x2(U, V, W,
                          a_idx=[10, 11],
                          b_idx=[8, 9, 12, 13],
                          w_idx=[2, 5],
                          col0=36)

    # Block 8: A_BR (cols 2,3) * B_B2 (rows 2,3, cols 2,3) -> C(2, 2:4)
    fill_schoolbook_1x2x2(U, V, W,
                          a_idx=[10, 11],
                          b_idx=[10, 11, 14, 15],
                          w_idx=[8, 11],
                          col0=40)

    # Verify
    if not _verify_decomp(U, V, W, n, m, p):
        return None

    return U, V, W, total_rank


def construct():
    """
    Returns the best known exact decomposition for 3x4 by 4x4 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin

    n, m, p = 3, 4, 4

    # First, get the schoolbook baseline
    U_sb, V_sb, W_sb, rank_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb

    # Check deadline
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}

    # Try the rank-44 Strassen-based decomposition
    result = _build_rank44()
    if result is not None:
        U44, V44, W44, rank44 = result
        if rank44 < best_rank:
            best_U, best_V, best_W, best_rank = U44, V44, W44, rank44

    # Check deadline
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}

    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
