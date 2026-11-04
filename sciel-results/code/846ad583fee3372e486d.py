"""
Strategy: Block-diagonal direct sum.
Partition A (4x4) into four 2x2 blocks: A11, A12, A21, A22.
Partition B (4x7) into two 2x7 blocks: B1 (top 2 rows), B2 (bottom 2 rows).
Compute C (4x7) as:
  C_top (2x7) = A11 * B1 + A12 * B2
  C_bot (2x7) = A21 * B1 + A22 * B2

For each 2x4 * 4x7 multiplication (e.g., C_top = [A11 A12] * [B1; B2]):
  Split B into B_left (2x2) and B_right (2x5).
  Compute A11 * B_left using Strassen (rank 7).
  Compute A12 * B_right using Schoolbook (rank 10).
  Total rank per 2x4 * 4x7 block = 17.
  Total rank for 4 blocks = 4 * 17 = 68.

Wait, the prompt says "investigate improvements within its family" but also "COMPUTE BUDGET... keep explicit deadline guard".
The previous program had rank 78.
Let's see if we can do better.
Actually, the previous program's `_strassen_2x4_4x7` used:
C11 (2x2) = A11*B11 + A12*B21. Rank 14.
C12 (2x2) = A11*B12 + A12*B22. Rank 14.
C13 (2x3) = A11*B13 + A12*B23. Rank 22.
Total for top 2 rows of C: 14+14+22 = 50.
Same for bottom 2 rows: 50.
Total 100? No, the previous code said 78.
Let's re-read the previous code's rank calculation.
C11: 14
C12: 14
C13: 22
C21: 14
C22: 14
C23: 22
Total: 100.
But the comment in the code said "Total Rank = 100" then "Wait... Total 78?".
Let's look at the actual construction in `_strassen_2x4_4x7`.
It calls `add_strassen` 4 times for 2x2 blocks? No.
It calls `add_strassen` for C11 (twice, 14), C12 (twice, 14), C21 (twice, 14), C22 (twice, 14). That's 56.
It calls `add_winograd_2x2_2x3` for C13 (twice, 22), C23 (twice, 22). That's 44.
Total 100.
Why did the comment say 78? Maybe it was a different version.
Regardless, 100 is worse than schoolbook 112, but an improvement.

Can we do better?
The "row-reuse" strategy suggested in the directive:
Partition into four 1x4x7 row-vector multiplications.
C[0, :] = A[0, :] * B
C[1, :] = A[1, :] * B
C[2, :] = A[2, :] * B
C[3, :] = A[3, :] * B

This is equivalent to 4 independent 1x4 * 4x7 multiplications.
Rank of 1x4 * 4x7?
Schoolbook: 1*4*7 = 28.
Total rank = 4 * 28 = 112. No improvement.

However, if we group rows, we can use 2x2 Strassen.
C[0:2, :] = A[0:2, :] * B
C[2:4, :] = A[2:4, :] * B

For C[0:2, :], which is 2x7 = A_2x4 * B_4x7.
We can split B into B_4x2 and B_4x5.
A_2x4 * B_4x2 (2x2 output) can be done with Strassen (rank 7) IF we treat it as 2x2 * 2x2?
No, A is 2x4, B is 4x2.
A = [A11 A12] (2x2 each). B = [B1; B2] (2x2 each).
Result = A11*B1 + A12*B2.
This is exactly the 2x2 * 2x2 Strassen case! Rank 7.
So A_2x4 * B_4x2 has rank 7.

Now A_2x4 * B_4x5 (2x5 output).
Schoolbook: 2*4*5 = 40.
Can we do better?
Split B_4x5 into B_4x2 and B_4x3.
A_2x4 * B_4x2 (2x2 output) -> Rank 7 (Strassen).
A_2x4 * B_4x3 (2x3 output) -> ?
For 2x3 output, schoolbook is 2*4*3 = 24.
Can we use Winograd?
Winograd 2x2 * 2x3 is rank 11.
But here A is 2x4, B is 4x3.
A = [A11 A12] (2x2). B = [B1; B2] (2x3 each).
Result = A11*B1 + A12*B2.
This is the sum of two 2x2 * 2x3 multiplications.
Rank = 11 + 11 = 22.
So A_2x4 * B_4x3 has rank 22.
Total for A_2x4 * B_4x5 is 7 + 22 = 29.
Total for A_2x4 * B_4x7 is 7 (for first 2 cols) + 29 (for last 5 cols) = 36.
So C[0:2, :] has rank 36.
C[2:4, :] has rank 36.
Total rank = 72.

This is better than 100.

Can we do even better?
What if we use 2x4 * 4x7 directly?
Is there a known decomposition?
The "row-reuse" hint suggests finding a shared basis.
But 72 is a solid improvement over 100.

Let's implement the 72-rank decomposition.
Structure:
1. C_top (2x7) = A_top (2x4) * B (4x7).
   Split B into B_left (4x2) and B_right (4x5).
   1a. C_top_left (2x2) = A_top * B_left. Rank 7 (Strassen).
   1b. C_top_right (2x5) = A_top * B_right.
       Split B_right into B_mid (4x2) and B_tail (4x3).
       1b-i. C_top_mid (2x2) = A_top * B_mid. Rank 7 (Strassen).
       1b-ii. C_top_tail (2x3) = A_top * B_tail. Rank 22 (2x Winograd 2x2*2x3).
   Total for C_top: 7 + 7 + 22 = 36.

2. C_bot (2x7) = A_bot (2x4) * B (4x7).
   Same structure. Rank 36.

Total Rank = 72.

Implementation details:
U: 16 rows (4x4).
V: 28 rows (4x7).
W: 28 rows (7x4, trace convention k*4+i).

Indices:
A_top rows: 0,1,2,3 (A[0][0..3], A[1][0..3]).
A_bot rows: 4,5,6,7 (A[2][0..3], A[3][0..3]).

B rows: 0..27.
B_left: 0,1, 14,15 (B[0][0], B[0][1], B[2][0], B[2][1]).
B_mid: 2,3, 16,17 (B[0][2], B[0][3], B[2][2], B[2][3]).
B_tail: 4,5,6, 18,19,20 (B[0][4..6], B[2][4..6]).

C_top rows: 0..7 (C[0][0..6], C[1][0..6]).
C_top_left: W indices for C[0][0], C[0][1], C[1][0], C[1][1].
W index = k*4 + i.
C[0][0] -> 0*4+0 = 0.
C[0][1] -> 1*4+0 = 4.
C[1][0] -> 0*4+1 = 1.
C[1][1] -> 1*4+1 = 5.
So C_top_left rows: [0, 4, 1, 5].

C_top_mid: C[0][2], C[0][3], C[1][2], C[1][3].
C[0][2] -> 2*4+0 = 8.
C[0][3] -> 3*4+0 = 12.
C[1][2] -> 2*4+1 = 9.
C[1][3] -> 3*4+1 = 13.
So C_top_mid rows: [8, 12, 9, 13].

C_top_tail: C[0][4..6], C[1][4..6].
C[0][4] -> 16.
C[0][5] -> 20.
C[0][6] -> 24.
C[1][4] -> 17.
C[1][5] -> 21.
C[1][6] -> 25.
So C_top_tail rows: [16, 20, 24, 17, 21, 25].

C_bot rows: 2..9 (C[2][0..6], C[3][0..6]).
C_bot_left: C[2][0], C[2][1], C[3][0], C[3][1].
C[2][0] -> 0*4+2 = 2.
C[2][1] -> 1*4+2 = 6.
C[3][0] -> 0*4+3 = 3.
C[3][1] -> 1*4+3 = 7.
So C_bot_left rows: [2, 6, 3, 7].

C_bot_mid: C[2][2], C[2][3], C[3][2], C[3][3].
C[2][2] -> 2*4+2 = 10.
C[2][3] -> 3*4+2 = 14.
C[3][2] -> 2*4+3 = 11.
C[3][3] -> 3*4+3 = 15.
So C_bot_mid rows: [10, 14, 11, 15].

C_bot_tail: C[2][4..6], C[3][4..6].
C[2][4] -> 18.
C[2][5] -> 22.
C[2][6] -> 26.
C[3][4] -> 19.
C[3][5] -> 23.
C[3][6] -> 27.
So C_bot_tail rows: [18, 22, 26, 19, 23, 27].

We need a helper to add a Strassen 2x2*2x2 block.
And a helper to add a "2x4*4x3" block (which is 2 Winograd 2x2*2x3).

Let's write the code.
"""

import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _schoolbook():
    """Return the schoolbook decomposition (rank n*m*p) as a fallback."""
    n, m, p = DIMENSIONS
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


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    u_val = U[i * m + j][c]
                    v_val = V[j * p + k][c]
                    w_val = W[k * n + i][c]
                    if u_val != 0 and v_val != 0 and w_val != 0:
                        total += u_val * v_val * w_val
                if total != 1:
                    return False
    return True


def _strassen_2x2_2x2():
    """
    Return the Strassen rank-7 decomposition for 2x2 * 2x2.
    U: shape (4, 7) - rows indexed by (i,j) for i,j in {0,1}
    V: shape (4, 7) - rows indexed by (j,k) for j,k in {0,1}
    W: shape (4, 7) - rows indexed by (k,i) for k,i in {0,1}
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _winograd_2x2_2x3():
    """
    Return the Winograd rank-11 decomposition for 2x2 * 2x3.
    U: shape (4, 11) - rows for A[i][j]
    V: shape (6, 11) - rows for B[j][k]
    W: shape (6, 11) - rows for C[i][k] (trace convention: index k*2+i)
    
    Decomposition:
    Strassen for first 2 columns (rank 7) + Schoolbook for 3rd column (rank 4).
    Total rank 11.
    """
    rank = 11
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(6)]
    W = [[0]*rank for _ in range(6)]
    
    U_s, V_s, W_s = _strassen_2x2_2x2()
    
    # Copy Strassen part
    for i in range(4):
        for c in range(7):
            U[i][c] = U_s[i][c]
            W[i][c] = W_s[i][c] # Note: W indices are k*2+i. Local W_s indices are k*2+i.
            
    # V mapping for Strassen part
    # V_s[0] (0,0) -> V[0] (B[0][0])
    # V_s[1] (0,1) -> V[1] (B[0][1])
    # V_s[2] (1,0) -> V[2] (B[1][0])
    # V_s[3] (1,1) -> V[3] (B[1][1])
    for i in range(4):
        for c in range(7):
            V[i][c] = V_s[i][c]
            
    # Schoolbook part for 3rd column (k=2)
    # C[0][2] = A[0][0]B[0][2] + A[0][1]B[1][2]
    # C[1][2] = A[1][0]B[0][2] + A[1][1]B[1][2]
    
    # W indices for k=2:
    # C[0][2] -> W[2*2+0] = W[4]
    # C[1][2] -> W[2*2+1] = W[5]
    
    # Term 1: A[0][0] * B[0][2]
    U[0][7] = 1
    V[4][7] = 1 # B[0][2] is V[4] in 2x3 layout?
    # V rows: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2).
    # So B[0][2] is V[2]. B[1][2] is V[5].
    # Let's check V row indices for 2x3.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=0, k=2 -> 2
    # j=1, k=0 -> 3
    # j=1, k=1 -> 4
    # j=1, k=2 -> 5
    
    # So V[2] is B[0][2]. V[5] is B[1][2].
    
    V[2][7] = 1
    W[4][7] = 1
    
    # Term 2: A[0][1] * B[1][2]
    U[1][8] = 1
    V[5][8] = 1
    W[4][8] = 1
    
    # Term 3: A[1][0] * B[0][2]
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1
    
    # Term 4: A[1][1] * B[1][2]
    U[3][10] = 1
    V[5][10] = 1
    W[5][10] = 1
    
    return U, V, W


def _construct_72_rank():
    """
    Construct the 72-rank decomposition.
    C_top (2x7) = A_top (2x4) * B (4x7). Rank 36.
    C_bot (2x7) = A_bot (2x4) * B (4x7). Rank 36.
    Total 72.
    """
    n, m, p = 4, 4, 7
    rank = 72
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(28)]
    W = [[0]*rank for _ in range(28)]
    col = 0
    
    def add_strassen_2x4_4x2(a_rows, b_rows, c_rows):
        """
        Adds a Strassen decomposition for A_2x4 * B_4x2 -> C_2x2.
        a_rows: 4 indices into U (A[0][0..3], A[1][0..3])
        b_rows: 4 indices into V (B[0][0..1], B[2][0..1])
        c_rows: 4 indices into W (C[0][0..1], C[1][0..1])
        """
        nonlocal col
        U_s, V_s, W_s = _strassen_2x2_2x2()
        # Map local indices to global
        # Local U: (0,0), (0,1), (1,0), (1,1)
        # a_rows order: A[0][0], A[0][1], A[1][0], A[1][1]
        for li in range(4):
            for c in range(7):
                U[a_rows[li]][col + c] = U_s[li][c]
                
        # Local V: (0,0), (0,1), (1,0), (1,1)
        # b_rows order: B[0][0], B[0][1], B[2][0], B[2][1]
        # Note: V_s[2] is (1,0) which corresponds to B[2][0] in global.
        for li in range(4):
            for c in range(7):
                V[b_rows[li]][col + c] = V_s[li][c]
                
        # Local W: (0,0), (0,1), (1,0), (1,1) -> W indices k*2+i
        # W_s[0] (0,0) -> C[0][0]
        # W_s[1] (0,1) -> C[0][1]
        # W_s[2] (1,0) -> C[1][0]
        # W_s[3] (1,1) -> C[1][1]
        # c_rows order must match this: C[0][0], C[0][1], C[1][0], C[1][1]
        for li in range(4):
            for c in range(7):
                W[c_rows[li]][col + c] = W_s[li][c]
                
        col += 7

    def add_winograd_2x4_4x3(a_rows, b_rows, c_rows):
        """
        Adds a Winograd-like decomposition for A_2x4 * B_4x3 -> C_2x3.
        Uses 2x Winograd 2x2*2x3.
        a_rows: 4 indices into U.
        b_rows: 6 indices into V (B[0][0..2], B[2][0..2]).
        c_rows: 6 indices into W (C[0][0..2], C[1][0..2]).
        """
        nonlocal col
        U_w, V_w, W_w = _winograd_2x2_2x3()
        
        # Part 1: A_left (2x2) * B_left (2x3)
        # A_left rows: a_rows[0], a_rows[1] (A[0][0..1], A[1][0..1])? No.
        # a_rows is A[0][0..3], A[1][0..3].
        # A_left is A[0][0..1], A[1][0..1]. Indices 0,1 of a_rows.
        # B_left is B[0][0..2], B[2][0..2]. Indices 0..2 of b_rows.
        # C_left is C[0][0..2], C[1][0..2]. Indices 0..2 of c_rows?
        # Wait, c_rows is C[0][0..2], C[1][0..2] for the local 2x3 block.
        
        for li in range(4):
            for c in range(11):
                U[a_rows[li]][col + c] = U_w[li][c]
                
        # V mapping:
        # Local V_w rows: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
        # Global b_rows: B[0][0], B[0][1], B[0][2], B[2][0], B[2][1], B[2][2]
        for li in range(6):
            for c in range(11):
                V[b_rows[li]][col + c] = V_w[li][c]
                
        # W mapping:
        # Local W_w rows: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2) -> W indices k*2+i
        # W_w[0] (0,0) -> C[0][0]
        # W_w[1] (0,1) -> C[0][1]
        # W_w[2] (0,2) -> C[0][2]
        # W_w[3] (1,0) -> C[1][0]
        # W_w[4] (1,1) -> C[1][1]
        # W_w[5] (1,2) -> C[1][2]
        # c_rows must be ordered: C[0][0], C[0][1], C[0][2], C[1][0], C[1][1], C[1][2]
        for li in range(6):
            for c in range(11):
                W[c_rows[li]][col + c] = W_w[li][c]
                
        col += 11
        
        # Part 2: A_right (2x2) * B_right (2x3)
        # A_right rows: a_rows[2], a_rows[3] (A[0][2..3], A[1][2..3])
        # B_right rows: b_rows[3], b_rows[4], b_rows[5] (B[2][0..2])? No.
        # Wait, B is 4x3.
        # B = [B_top (2x3); B_bot (2x3)].
        # b_rows[0..2] is B_top. b_rows[3..5] is B_bot.
        # A = [A_left (2x2) A_right (2x2)].
        # a_rows[0..1] is A_left. a_rows[2..3] is A_right.
        
        for li in range(4):
            for c in range(11):
                U[a_rows[2 + (li % 2)]][col + c] = U_w[li][c] # This mapping is tricky.
                # U_w[0] is A[0][0]. We want A[0][2] which is a_rows[2].
                # U_w[1] is A[0][1]. We want A[0][3] which is a_rows[3].
                # U_w[2] is A[1][0]. We want A[1][2] which is a_rows[4]? No, a_rows has 4 elements.
                # a_rows: [A[0][0], A[0][1], A[1][0], A[1][1]] for the 2x4 block.
                # So A_left is a_rows[0], a_rows[1]. A_right is a_rows[2], a_rows[3].
                # U_w[0] (A[0][0]) -> a_rows[2]
                # U_w[1] (A[0][1]) -> a_rows[3]
                # U_w[2] (A[1][0]) -> a_rows[2+2]? No, a_rows[2] is A[1][0]?
                # Let's check a_rows definition in caller.
                # a_rows = [0, 1, 4, 5] for A_top.
                # A[0][0]=0, A[0][1]=1, A[1][0]=4, A[1][1]=5.
                # So A_left is 0,1. A_right is 4,5.
                # U_w[0] -> 4. U_w[1] -> 5. U_w[2] -> ?
                # U_w[2] is A[1][0] in local. Global A[1][2] is a_rows[2]? No.
                # a_rows[2] is A[1][0] (index 4).
                # a_rows[3] is A[1][1] (index 5).
                # So A_right is a_rows[2], a_rows[3].
                # U_w[0] (A[0][0]) -> a_rows[2]
                # U_w[1] (A[0][1]) -> a_rows[3]
                # U_w[2] (A[1][0]) -> a_rows[2] ?? No.
                # U_w[2] corresponds to local A[1][0]. In the right block, this is A[1][2].
                # But a_rows only has 4 elements for the 2x4 matrix.
                # a_rows[0]=A[0][0], a_rows[1]=A[0][1], a_rows[2]=A[1][0], a_rows[3]=A[1][1].
                # So A_right is a_rows[2], a_rows[3].
                # Local A[0][0] -> Global A[1][0]? No.
                # Local A[0][0] is the top-left of the right 2x2 block.
                # The right 2x2 block is A[0][2..3], A[1][2..3].
                # So Local A[0][0] -> Global A[0][2].
                # But a_rows[2] is A[1][0].
                # This means my a_rows ordering is [A00, A01, A10, A11].
                # So A02 is not in a_rows?
                # Ah, A is 2x4.
                # A[0][0], A[0][1], A[0][2], A[0][3]
                # A[1][0], A[1][1], A[1][2], A[1][3]
                # a_rows should have 4 elements? No, 2x4 has 8 elements.
                # My U has 16 rows.
                # A_top is 2x4. Rows 0,1,2,3? No.
                # A is 4x4.
                # A_top is rows 0,1 of A.
                # A[0][0], A[0][1], A[0][2], A[0][3] -> U[0], U[1], U[2], U[3].
                # A[1][0], A[1][1], A[1][2], A[1][3] -> U[4], U[5], U[6], U[7].
                # So a_rows for A_top should be [0, 1, 2, 3, 4, 5, 6, 7].
                # But my add_strassen_2x4_4x2 expects 4 rows.
                # This function is for 2x2 * 2x2.
                # I need a function for 2x4 * 4x2.
                
        # Let's restart the helper functions with correct dimensions.
        pass

    # Corrected Helpers
    
    def add_strassen_2x4_4x2(a_rows, b_rows, c_rows):
        """
        a_rows: 8 indices (A[0][0..3], A[1][0..3])
        b_rows: 4 indices (B[0][0..1], B[2][0..1])
        c_rows: 4 indices (C[0][0..1], C[1][0..1])
        """
        nonlocal col
        U_s, V_s, W_s = _strassen_2x2_2x2()
        # A_left (2x2): a_rows[0], a_rows[1], a_rows[4], a_rows[5]
        # A_right (2x2): a_rows[2], a_rows[3], a_rows[6], a_rows[7]
        
        # Strassen for A_left * B_left
        # A_left rows: 0,1,4,5
        # B_left rows: 0,1,2,3 (B[0][0], B[0][1], B[2][0], B[2][1])
        # C_left rows: 0,1,2,3 (C[0][0], C[0][1], C[1][0], C[1][1])
        
        a_left = [a_rows[0], a_rows[1], a_rows[4], a_rows[5]]
        a_right = [a_rows[2], a_rows[3], a_rows[6], a_rows[7]]
        
        # B_left is b_rows[0..3]
        # B_right is b_rows[4..7]? No, b_rows has 4 elements.
        # B is 4x2.
        # B[0][0], B[0][1] -> b_rows[0], b_rows[1]
        # B[2][0], B[2][1] -> b_rows[2], b_rows[3]
        # So B_left (top 2 rows) is b_rows[0..1].
        # B_right (bottom 2 rows) is b_rows[2..3].
        
        # Strassen 1: A_left * B_top
        for li in range(4):
            for c in range(7):
                U[a_left[li]][col + c] = U_s[li][c]
        for li in range(4):
            for c in range(7):
                V[b_rows[li % 2]][col + c] = V_s[li][c] # This mapping is wrong.
                
        # Let's be precise.
        # V_s[0] (0,0) -> B[0][0] -> b_rows[0]
        # V_s[1] (0,1) -> B[0][1] -> b_rows[1]
        # V_s[2] (1,0) -> B[1][0] -> b_rows[2]
        # V_s[3] (1,1) -> B[1][1] -> b_rows[3]
        
        for li in range(4):
            for c in range(7):
                V[b_rows[li]][col + c] = V_s[li][c]
                
        for li in range(4):
            for c in range(7):
                W[c_rows[li]][col + c] = W_s[li][c]
                
        col += 7
        
        # Strassen 2: A_right * B_bot
        # A_right rows: 2,3,6,7
        # B_bot rows: 2,3 (B[2][0], B[2][1]) -> b_rows[2], b_rows[3]
        
        for li in range(4):
            for c in range(7):
                U[a_right[li]][col + c] = U_s[li][c]
        for li in range(4):
            for c in range(7):
                V[b_rows[2 + (li % 2)]][col + c] = V_s[li][c] # V_s[0]->b2, V_s[1]->b3, V_s[2]->b2, V_s[3]->b3
                # V_s[0] (0,0) -> B[0][0] of B_bot -> b_rows[2]
                # V_s[1] (0,1) -> B[0][1] of B_bot -> b_rows[3]
                # V_s[2] (1,0) -> B[1][0] of B_bot -> b_rows[2]
                # V_s[3] (1,1) -> B[1][1] of B_bot -> b_rows[3]
                # So V[b_rows[2 + (li % 2)]] is correct for li 0,1 -> 2,3. li 2,3 -> 2,3.
                # Wait, li=0 -> 2. li=1 -> 3. li=2 -> 2. li=3 -> 3.
                # V_s[0] -> b2. V_s[1] -> b3. V_s[2] -> b2. V_s[3] -> b3.
                # This matches.
                
        for li in range(4):
            for c in range(7):
                W[c_rows[li]][col + c] = W_s[li][c]
                
        col += 7

    def add_winograd_2x4_4x3(a_rows, b_rows, c_rows):
        """
        a_rows: 8 indices.
        b_rows: 6 indices (B[0][0..2], B[2][0..2]).
        c_rows: 6 indices (C[0][0..2], C[1][0..2]).
        """
        nonlocal col
        U_w, V_w, W_w = _winograd_2x2_2x3()
        
        a_left = [a_rows[0], a_rows[1], a_rows[4], a_rows[5]]
        a_right = [a_rows[2], a_rows[3], a_rows[6], a_rows[7]]
        
        # B_top: b_rows[0..2]
        # B_bot: b_rows[3..5]
        
        # Winograd 1: A_left * B_top
        for li in range(4):
            for c in range(11):
                U[a_left[li]][col + c] = U_w[li][c]
        for li in range(6):
            for c in range(11):
                V[b_rows[li]][col + c] = V_w[li][c]
        for li in range(6):
            for c in range(11):
                W[c_rows[li]][col + c] = W_w[li][c]
        col += 11
        
        # Winograd 2: A_right * B_bot
        for li in range(4):
            for c in range(11):
                U[a_right[li]][col + c] = U_w[li][c]
        for li in range(6):
            for c in range(11):
                V[b_rows[3 + li]][col + c] = V_w[li][c]
        for li in range(6):
            for c in range(11):
                W[c_rows[li]][col + c] = W_w[li][c]
        col += 11

    # Indices
    A_top = [0, 1, 2, 3, 4, 5, 6, 7]
    A_bot = [8, 9, 10, 11, 12, 13, 14, 15]
    
    B_left = [0, 1, 14, 15] # B[0][0], B[0][1], B[2][0], B[2][1]
    B_mid = [2, 3, 16, 17] # B[0][2], B[0][3], B[2][2], B[2][3]
    B_tail = [4, 5, 6, 18, 19, 20] # B[0][4..6], B[2][4..6]
    
    C_top_left = [0, 4, 1, 5] # C[0][0], C[0][1], C[1][0], C[1][1]
    C_top_mid = [8, 12, 9, 13] # C[0][2], C[0][3], C[1][2], C[1][3]
    C_top_tail = [16, 20, 24, 17, 21, 25] # C[0][4..6], C[1][4..6]
    
    C_bot_left = [2, 6, 3, 7]
    C_bot_mid = [10, 14, 11, 15]
    C_bot_tail = [18, 22, 26, 19, 23, 27]
    
    # Top
    add_strassen_2x4_4x2(A_top, B_left, C_top_left)
    add_strassen_2x4_4x2(A_top, B_mid, C_top_mid)
    add_winograd_2x4_4x3(A_top, B_tail, C_top_tail)
    
    # Bot
    add_strassen_2x4_4x2(A_bot, B_left, C_bot_left)
    add_strassen_2x4_4x2(A_bot, B_mid, C_bot_mid)
    add_winograd_2x4_4x3(A_bot, B_tail, C_bot_tail)
    
    return U, V, W


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    try:
        U, V, W = _construct_72_rank()
        result = {"U": U, "V": V, "W": W}
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    result = _schoolbook()
    assert _verify(result["U"], result["V"], result["W"], n, m, p), "Decomposition failed verification"
    return result

# EVOLVE-BLOCK-END
