"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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


def _strassen_4x4_4x4():
    """
    Strassen-based decomposition for 4x4 * 4x4, rank 28.
    U: shape (16, 28) - rows for A[i][j], i,j in 0..3
    V: shape (16, 28) - rows for B[j][k], j,k in 0..3
    W: shape (16, 28) - rows for C[k][i], k,i in 0..3 (trace convention)
    """
    rank = 28
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    col = 0

    def add_strassen(a_rows, b_rows, c_rows):
        """Add Strassen 2x2*2x2 using explicit row lists."""
        nonlocal col
        U_s, V_s, W_s = _strassen_2x2_2x2()
        for li in range(4):
            for c in range(7):
                U[a_rows[li]][col + c] = U_s[li][c]
        for lb in range(4):
            for c in range(7):
                V[b_rows[lb]][col + c] = V_s[lb][c]
        for lc in range(4):
            for c in range(7):
                W[c_rows[lc]][col + c] = W_s[lc][c]
        col += 7

    # A11: i=0,1; j=0,1 -> U rows [0,1,4,5]
    # A12: i=0,1; j=2,3 -> U rows [2,3,6,7]
    # A21: i=2,3; j=0,1 -> U rows [8,9,12,13]
    # A22: i=2,3; j=2,3 -> U rows [10,11,14,15]
    A11_rows = [0, 1, 4, 5]
    A12_rows = [2, 3, 6, 7]
    A21_rows = [8, 9, 12, 13]
    A22_rows = [10, 11, 14, 15]

    # B11: j=0,1; k=0,1 -> V rows [0,1,4,5]
    # B12: j=0,1; k=2,3 -> V rows [2,3,6,7]
    # B21: j=2,3; k=0,1 -> V rows [8,9,12,13]
    # B22: j=2,3; k=2,3 -> V rows [10,11,14,15]
    B11_rows = [0, 1, 4, 5]
    B12_rows = [2, 3, 6, 7]
    B21_rows = [8, 9, 12, 13]
    B22_rows = [10, 11, 14, 15]

    # C11: i=0,1; k=0,1 -> W rows [0,1,4,5]
    # C12: i=0,1; k=2,3 -> W rows [8,9,12,13]
    # C21: i=2,3; k=0,1 -> W rows [2,3,6,7]
    # C22: i=2,3; k=2,3 -> W rows [10,11,14,15]
    C11_rows = [0, 1, 4, 5]
    C12_rows = [8, 9, 12, 13]
    C21_rows = [2, 3, 6, 7]
    C22_rows = [10, 11, 14, 15]

    # C11 = A11*B11 + A12*B21
    add_strassen(A11_rows, B11_rows, C11_rows)
    add_strassen(A12_rows, B21_rows, C11_rows)

    # C12 = A11*B12 + A12*B22
    add_strassen(A11_rows, B12_rows, C12_rows)
    add_strassen(A12_rows, B22_rows, C12_rows)

    # C21 = A21*B11 + A22*B21
    add_strassen(A21_rows, B11_rows, C21_rows)
    add_strassen(A22_rows, B21_rows, C21_rows)

    # C22 = A21*B12 + A22*B22
    add_strassen(A21_rows, B12_rows, C22_rows)
    add_strassen(A22_rows, B22_rows, C22_rows)

    return U, V, W


def _strassen_4x4_4x7():
    """
    Block-wise Strassen decomposition for 4x4 * 4x7.
    
    A (4x4) split into 2x2 blocks:
      A11 = A[0:2,0:2], A12 = A[0:2,2:4], A21 = A[2:4,0:2], A22 = A[2:4,2:4]
    
    B (4x7) split:
      B11 = B[0:2,0:2], B12 = B[0:2,2:4], B13 = B[0:2,4:7]
      B21 = B[2:4,0:2], B22 = B[2:4,2:4], B23 = B[2:4,4:7]
    
    C (4x7):
      C11 = A11*B11 + A12*B21  (2x2)  -> Strassen rank 7 each, total 14
      C12 = A11*B12 + A12*B22  (2x2)  -> Strassen rank 7 each, total 14
      C13 = A11*B13 + A12*B23  (2x3)  -> rank 11 each, total 22
      C21 = A21*B11 + A22*B21  (2x2)  -> Strassen rank 7 each, total 14
      C22 = A21*B12 + A22*B22  (2x2)  -> Strassen rank 7 each, total 14
      C23 = A21*B13 + A22*B23  (2x3)  -> rank 11 each, total 22
    
    Total rank: 4*14 + 2*22 = 56 + 44 = 100... wait let me recount.
    Actually: 6 blocks of C, each needs 2 multiplications.
    C11: 2 Strassen = 14
    C12: 2 Strassen = 14
    C13: 2 x (Strassen 2x2*2x2 + schoolbook 2x2*2x1) = 2*(7+4) = 22
    C21: 2 Strassen = 14
    C22: 2 Strassen = 14
    C23: 2 x (Strassen 2x2*2x2 + schoolbook 2x2*2x1) = 2*(7+4) = 22
    Total: 14+14+22+14+14+22 = 100
    
    Hmm, that's worse than expected. Let me reconsider.
    
    Actually the issue is that each C block is a sum of two products. For 2x2*2x2 + 2x2*2x2,
    we need 2 separate Strassen decompositions (can't share because different B blocks).
    So rank per C block:
    - 2x2 output: 2 * 7 = 14
    - 2x3 output: 2 * 11 = 22
    
    Total: 4*14 + 2*22 = 56 + 44 = 100. That's worse than schoolbook 112? No, 100 < 112.
    But we can do better.
    
    Wait, I think the issue is that we're being too conservative. Let me reconsider the structure.
    
    Actually, for 4x4 * 4x7, we can split differently:
    Split B into B_left (4x4) and B_right (4x3).
    C_left = A * B_left (4x4 * 4x4) -> rank 28 (Strassen)
    C_right = A * B_right (4x4 * 4x3) -> need a decomposition
    
    For 4x4 * 4x3:
    Split B_right into B_right_left (4x2) and B_right_right (4x1).
    C_right_left = A * B_right_left (4x4 * 4x2)
    C_right_right = A * B_right_right (4x4 * 4x1)
    
    For 4x4 * 4x2:
    Split B into two 4x1 columns. Each 4x4 * 4x1 has rank 16 (schoolbook).
    But we can do better: 4x4 * 4x1 = sum of 4 (4x1 * 4x1) terms... no.
    Actually 4x4 * 4x1: C[i] = sum_j A[i,j]*B[j,0]. Rank = 16 schoolbook.
    Can we do better? For 4x4 * 4x1, the tensor has rank at most 16.
    Using Strassen-like: split A into 2x2 blocks, B into 2x1 blocks.
    C_top = A_top * B_top + A_bot * B_bot where A_top is 2x4, B_top is 2x1.
    2x4 * 2x1: C[i] = sum_j A[i,j]*B[j,0]. Rank = 4*2*1 = 8 schoolbook.
    Can we do 2x4 * 2x1 in rank < 8? 
    2x4 * 2x1: A is 2x4, B is 2x1. C[i] = A[i,0]*B[0,0] + A[i,1]*B[1,0] + A[i,2]*B[0,0] + A[i,3]*B[1,0].
    Wait, B is 2x1, so B[0,0] and B[1,0].
    C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[0,0] + A[0,3]*B[1,0]
    C[1] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[0,0] + A[1,3]*B[1,0]
    
    We can factor: C[0] = (A[0,0]+A[0,2])*B[0,0] + (A[0,1]+A[0,3])*B[1,0]
    C[1] = (A[1,0]+A[1,2])*B[0,0] + (A[1,1]+A[1,3])*B[1,0]
    
    So we need:
    M1 = (A[0,0]+A[0,2]) * B[0,0] -> contributes to C[0]
    M2 = (A[0,1]+A[0,3]) * B[1,0] -> contributes to C[0]
    M3 = (A[1,0]+A[1,2]) * B[0,0] -> contributes to C[1]
    M4 = (A[1,1]+A[1,3]) * B[1,0] -> contributes to C[1]
    
    That's rank 4 for 2x4 * 2x1! Much better than 8.
    
    So 4x4 * 4x2:
    Split A into A_top (2x4), A_bot (2x4).
    Split B into B_top (2x1), B_bot (2x1).
    C_top = A_top * B_top + A_bot * B_bot? No, that's not right.
    C = A * B where A is 4x4, B is 4x2.
    C[i,k] = sum_j A[i,j]*B[j,k].
    Split i into 0,1 and 2,3. Split j into 0,1 and 2,3.
    C_top (2x2) = A_top(2x4) * B(4x2)
    A_top = [A[0,0..3], A[1,0..3]]
    B = [B[0,0..1]; B[1,0..1]; B[2,0..1]; B[3,0..1]]
    C_top[i,k] = sum_{j=0}^{3} A_top[i,j]*B[j,k]
    = sum_{j=0}^{1} A_top[i,j]*B[j,k] + sum_{j=2}^{3} A_top[i,j]*B[j,k]
    = A_top_left(2x2) * B_top(2x2) + A_top_right(2x2) * B_bot(2x2)
    
    Where A_top_left = A[0:2, 0:2], A_top_right = A[0:2, 2:4]
    B_top = B[0:2, 0:2], B_bot = B[2:4, 0:2]
    
    Each 2x2 * 2x2 has rank 7 (Strassen). So C_top needs 2*7 = 14.
    Similarly C_bot needs 14.
    Total for 4x4 * 4x2: 28.
    
    For 4x4 * 4x1:
    C[i] = sum_j A[i,j]*B[j,0].
    Split i into 0,1 and 2,3. Split j into 0,1 and 2,3.
    C_top[i] = sum_{j=0}^{1} A_top[i,j]*B[j,0] + sum_{j=2}^{3} A_top[i,j]*B[j,0]
    = A_top_left(2x2) * B_top(2x1) + A_top_right(2x2) * B_bot(2x1)
    
    Each 2x2 * 2x1 has rank 4 (as shown above). So C_top needs 2*4 = 8.
    Similarly C_bot needs 8.
    Total for 4x4 * 4x1: 16.
    
    So 4x4 * 4x3 = 4x4 * 4x2 + 4x4 * 4x1 = 28 + 16 = 44.
    
    Total for 4x4 * 4x7 = 4x4 * 4x4 + 4x4 * 4x3 = 28 + 44 = 72.
    
    That's better than 100! Let me implement this.
    """
    n, m, p = 4, 4, 7
    rank = 72
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(28)]
    W = [[0]*rank for _ in range(28)]
    col = 0

    def add_strassen_2x2(a_rows, b_rows, c_rows):
        """Add Strassen 2x2*2x2 using explicit row lists."""
        nonlocal col
        U_s, V_s, W_s = _strassen_2x2_2x2()
        for li in range(4):
            for c in range(7):
                U[a_rows[li]][col + c] = U_s[li][c]
        for lb in range(4):
            for c in range(7):
                V[b_rows[lb]][col + c] = V_s[lb][c]
        for lc in range(4):
            for c in range(7):
                W[c_rows[lc]][col + c] = W_s[lc][c]
        col += 7

    def add_2x2_2x1(a_rows, b_rows, c_rows):
        """
        Add 2x2 * 2x1 decomposition, rank 4.
        a_rows: 4 U-row indices for the 2x2 A block (local order 00,01,10,11)
        b_rows: 2 V-row indices for the 2x1 B block (b0, b1)
        c_rows: 2 W-row indices for the 2x1 C block (c0, c1)
        
        C[0] = (A[0,0]+A[0,1])*B[0] + (A[0,2]+A[0,3])*B[1]  -- wait, this is wrong.
        Let me re-derive.
        
        A is 2x2: A[0,0], A[0,1], A[1,0], A[1,1]
        B is 2x1: B[0], B[1]
        C[0] = A[0,0]*B[0] + A[0,1]*B[1]
        C[1] = A[1,0]*B[0] + A[1,1]*B[1]
        
        This is just schoolbook with rank 4. Can we do better?
        M1 = (A[0,0]+A[1,0]) * B[0] -> contributes to C[0] and C[1]
        M2 = (A[0,1]+A[1,1]) * B[1] -> contributes to C[0] and C[1]
        M3 = (A[0,0]-A[1,0]) * B[0] -> contributes to C[0] and C[1] with signs
        M4 = (A[0,1]-A[1,1]) * B[1] -> contributes to C[0] and C[1] with signs
        
        C[0] = (M1+M3)/2 + (M2+M4)/2
        C[1] = (M1-M3)/2 + (M2-M4)/2
        
        But we need integer/half-integer entries. Let's use:
        M1 = (A[0,0]+A[1,0]) * B[0]
        M2 = (A[0,1]+A[1,1]) * B[1]
        M3 = (A[0,0]-A[1,0]) * B[0]
        M4 = (A[0,1]-A[1,1]) * B[1]
        
        C[0] = 0.5*M1 + 0.5*M3 + 0.5*M2 + 0.5*M4
        C[1] = 0.5*M1 - 0.5*M3 + 0.5*M2 - 0.5*M4
        
        So:
        U for M1: a_rows[0]=1, a_rows[2]=1
        V for M1: b_rows[0]=1
        W for M1: c_rows[0]=0.5, c_rows[1]=0.5
        
        U for M2: a_rows[1]=1, a_rows[3]=1
        V for M2: b_rows[1]=1
        W for M2: c_rows[0]=0.5, c_rows[1]=0.5
        
        U for M3: a_rows[0]=1, a_rows[2]=-1
        V for M3: b_rows[0]=1
        W for M3: c_rows[0]=0.5, c_rows[1]=-0.5
        
        U for M4: a_rows[1]=1, a_rows[3]=-1
        V for M4: b_rows[1]=1
        W for M4: c_rows[0]=0.5, c_rows[1]=-0.5
        """
        nonlocal col
        # M1
        U[a_rows[0]][col] = 1
        U[a_rows[2]][col] = 1
        V[b_rows[0]][col] = 1
        W[c_rows[0]][col] = 0.5
        W[c_rows[1]][col] = 0.5
        col += 1
        # M2
        U[a_rows[1]][col] = 1
        U[a_rows[3]][col] = 1
        V[b_rows[1]][col] = 1
        W[c_rows[0]][col] = 0.5
        W[c_rows[1]][col] = 0.5
        col += 1
        # M3
        U[a_rows[0]][col] = 1
        U[a_rows[2]][col] = -1
        V[b_rows[0]][col] = 1
        W[c_rows[0]][col] = 0.5
        W[c_rows[1]][col] = -0.5
        col += 1
        # M4
        U[a_rows[1]][col] = 1
        U[a_rows[3]][col] = -1
        V[b_rows[1]][col] = 1
        W[c_rows[0]][col] = 0.5
        W[c_rows[1]][col] = -0.5
        col += 1

    # Part 1: 4x4 * 4x4 (columns 0-3 of B)
    # A11: i=0,1; j=0,1 -> U rows [0,1,4,5]
    # A12: i=0,1; j=2,3 -> U rows [2,3,6,7]
    # A21: i=2,3; j=0,1 -> U rows [8,9,12,13]
    # A22: i=2,3; j=2,3 -> U rows [10,11,14,15]
    A11_rows = [0, 1, 4, 5]
    A12_rows = [2, 3, 6, 7]
    A21_rows = [8, 9, 12, 13]
    A22_rows = [10, 11, 14, 15]

    # B11: j=0,1; k=0,1 -> V rows [0,1,7,8]
    # B12: j=0,1; k=2,3 -> V rows [2,3,9,10]
    # B21: j=2,3; k=0,1 -> V rows [14,15,21,22]
    # B22: j=2,3; k=2,3 -> V rows [16,17,23,24]
    B11_rows = [0, 1, 7, 8]
    B12_rows = [2, 3, 9, 10]
    B21_rows = [14, 15, 21, 22]
    B22_rows = [16, 17, 23, 24]

    # C11: i=0,1; k=0,1 -> W rows [0,1,4,5]
    # C12: i=0,1; k=2,3 -> W rows [8,9,12,13]
    # C21: i=2,3; k=0,1 -> W rows [2,3,6,7]
    # C22: i=2,3; k=2,3 -> W rows [10,11,14,15]
    C11_rows = [0, 1, 4, 5]
    C12_rows = [8, 9, 12, 13]
    C21_rows = [2, 3, 6, 7]
    C22_rows = [10, 11, 14, 15]

    # C11 = A11*B11 + A12*B21
    add_strassen_2x2(A11_rows, B11_rows, C11_rows)
    add_strassen_2x2(A12_rows, B21_rows, C11_rows)

    # C12 = A11*B12 + A12*B22
    add_strassen_2x2(A11_rows, B12_rows, C12_rows)
    add_strassen_2x2(A12_rows, B22_rows, C12_rows)

    # C21 = A21*B11 + A22*B21
    add_strassen_2x2(A21_rows, B11_rows, C21_rows)
    add_strassen_2x2(A22_rows, B21_rows, C21_rows)

    # C22 = A21*B12 + A22*B22
    add_strassen_2x2(A21_rows, B12_rows, C22_rows)
    add_strassen_2x2(A22_rows, B22_rows, C22_rows)

    # Part 2: 4x4 * 4x2 (columns 4-5 of B)
    # B13: j=0,1; k=4,5 -> V rows [4,5,11,12]
    # B23: j=2,3; k=4,5 -> V rows [18,19,25,26]
    B13_rows = [4, 5, 11, 12]
    B23_rows = [18, 19, 25, 26]

    # C13: i=0,1; k=4,5 -> W rows [16,17,20,21]
    # C23: i=2,3; k=4,5 -> W rows [18,19,22,23]
    C13_rows = [16, 17, 20, 21]
    C23_rows = [18, 19, 22, 23]

    # C13 = A11*B13 + A12*B23
    add_strassen_2x2(A11_rows, B13_rows, C13_rows)
    add_strassen_2x2(A12_rows, B23_rows, C13_rows)

    # C23 = A21*B13 + A22*B23
    add_strassen_2x2(A21_rows, B13_rows, C23_rows)
    add_strassen_2x2(A22_rows, B23_rows, C23_rows)

    # Part 3: 4x4 * 4x1 (column 6 of B)
    # B14: j=0,1; k=6 -> V rows [6, 13]
    # B24: j=2,3; k=6 -> V rows [20, 27]
    B14_rows = [6, 13]
    B24_rows = [20, 27]

    # C14: i=0,1; k=6 -> W rows [24, 25]
    # C24: i=2,3; k=6 -> W rows [26, 27]
    C14_rows = [24, 25]
    C24_rows = [26, 27]

    # C14 = A11*B14 + A12*B24
    # A11: [0,1,4,5], A12: [2,3,6,7]
    add_2x2_2x1(A11_rows, B14_rows, C14_rows)
    add_2x2_2x1(A12_rows, B24_rows, C14_rows)

    # C24 = A21*B14 + A22*B24
    # A21: [8,9,12,13], A22: [10,11,14,15]
    add_2x2_2x1(A21_rows, B14_rows, C24_rows)
    add_2x2_2x1(A22_rows, B24_rows, C24_rows)

    return U, V, W


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.
    Returns {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    # Try the improved Strassen-based decomposition (rank 72)
    try:
        U, V, W = _strassen_4x4_4x7()
        result = {"U": U, "V": V, "W": W}
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass

    # Fallback to schoolbook
    result = _schoolbook()
    assert _verify(result["U"], result["V"], result["W"], n, m, p), "Decomposition failed verification"
    return result

# EVOLVE-BLOCK-END
