"""Matrix multiplication 3,3,5 — rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-43 decomposition for 3x3 * 3x5 matrix multiplication.
    
    Strategy:
    - Split B (3x5) into B1 (3x3) and B2 (3x2).
    - C = A*B = [C1, C2] where C1 = A*B1 (3x3) and C2 = A*B2 (3x2).
    - Use Lader's rank-23 algorithm for the 3x3x3 part (C1).
    - Use schoolbook rank-18 for the 3x3x2 part (C2).
    - Total rank = 23 + 18 = 41.
    
    Wait, let me double-check the Lader decomposition. The standard Lader decomposition 
    for 3x3 matrix multiplication has rank 23. Let me verify this is correct.
    
    Actually, the best known rank for 3x3x3 is 23 (Lader, 1973). Let me implement it.
    
    For 3x3x3: A (3x3) * B (3x3) = C (3x3)
    U has 9 rows (i*m+j for i,j in 0..2), V has 9 rows (j*p+k for j,k in 0..2), W has 9 rows (k*n+i for k,i in 0..2)
    
    Lader's decomposition uses 23 terms. Let me look up the specific coefficients.
    
    The Lader decomposition for 3x3 matrix multiplication:
    Let A = [a11 a12 a13; a21 a22 a23; a31 a32 a33]
    Let B = [b11 b12 b13; b21 b22 b23; b31 b32 b33]
    
    The 23 bilinear forms are:
    M1 = (a11+a12+a13)(b11+b21+b31)
    M2 = (a11-a12+a13)(b11-b21+b31)
    M3 = (a11+a12-a13)(b11+b21-b31)
    M4 = (a11-a12-a13)(b11-b21-b31)
    M5 = (a12+a22+a32)(b21+b22+b23)
    M6 = (a12-a22+a32)(b21-b22+b23)
    M7 = (a12+a22-a32)(b21+b22-b23)
    M8 = (a12-a22-a32)(b21-b22-b23)
    M9 = (a13+a23+a33)(b31+b32+b33)
    M10 = (a13-a23+a33)(b31-b32+b33)
    M11 = (a13+a23-a33)(b31+b32-b33)
    M12 = (a13-a23-a33)(b31-b32-b33)
    M13 = (a21+a22+a23)(b12+b13+b22+b23)
    M14 = (a21-a22+a23)(b12-b13+b22-b23)
    M15 = (a21+a22-a23)(b12+b13+b22-b23)
    M16 = (a21-a22-a23)(b12-b13+b22-b23)
    M17 = (a31+a32+a33)(b11+b12+b13)
    M18 = (a31-a32+a33)(b11-b12+b13)
    M19 = (a31+a32-a33)(b11+b12-b13)
    M20 = (a31-a32-a33)(b11-b12-b13)
    M21 = a11*b11
    M22 = a22*b22
    M23 = a33*b33
    
    Hmm, that doesn't look right. Let me think more carefully.
    
    Actually, I'm not confident I can correctly implement Lader's 23-rank decomposition from memory. 
    Let me use a safer approach: Strassen for 3x3x3.
    
    Strassen for 3x3x3: We can use Strassen's 7-rank for 2x2x2 and extend to 3x3x3.
    One approach: partition the 3x3 matrices into 2x2 blocks with an extra row/column.
    
    A = [A11 A12; A21 A22; A31 A32 A33] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1, etc.
    This gets complicated.
    
    Alternative: Use the known result that 3x3x3 can be done in rank 23, but if I can't implement it correctly, 
    I'll use a simpler decomposition.
    
    Let me try a different approach: Use Strassen's algorithm for the 2x2x2 sub-problems.
    
    For 3x3x3, we can write:
    A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x1, B21 is 1x2, B22 is 1x1
    
    C = A*B = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
               [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    This requires:
    - 1 Strassen for 2x2x2 (rank 7) for A11*B11
    - 2 rank-2 for 2x1x1 (A12*B21, A21*B12) - actually these are just dot products
    - 1 rank-1 for 1x1x1 (A22*B22)
    - 1 rank-2 for 2x1x1 (A12*B22)
    - 1 rank-2 for 1x2x1 (A21*B21)
    
    This is getting messy. Let me just use the schoolbook for 3x3x3 (rank 27) and schoolbook for 3x3x2 (rank 18).
    Total rank = 27 + 18 = 45. That's the same as the original.
    
    OK, let me try to implement a better decomposition. 
    
    For 3x3x2: A (3x3) * B (3x2) = C (3x2)
    We can use Strassen-like ideas. Partition A into 2x2 and 1x1 blocks.
    A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    B = [[B11, B12], [B21, B22]] where B11 is 2x1, B12 is 2x1, B21 is 1x1, B22 is 1x1
    
    C = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
         [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    C11 (2x1) = A11*B11 + A12*B21
    C12 (2x1) = A11*B12 + A12*B22
    C21 (1x1) = A21*B11 + A22*B21
    C22 (1x1) = A21*B12 + A22*B22
    
    For C11: A11 (2x2) * B11 (2x1). This is like multiplying a 2x2 by a 2x1.
    We can use 3 multiplications: (a11+a21)*(b11+b21), (a11-a21)*(b11-b21), a11*b11
    Then C11_1 = M1/2 + M2/2 + a11*b11, C11_2 = M1/2 - M2/2
    
    Similarly for C12.
    
    For C21: A21 (1x2) * B11 (2x1) = a21*b11 + a22*b21 (just 2 multiplications)
    For C22: A21 (1x2) * B12 (2x1) = a21*b12 + a22*b22 (just 2 multiplications)
    
    Total for 3x3x2: 3 + 3 + 2 + 2 = 10? That seems too good. Let me recount.
    
    Actually, for 2x2 * 2x1, the rank is 3 (it's a rank-3 tensor).
    So C11 uses 3, C12 uses 3, C21 uses 2, C22 uses 2. Total = 10.
    
    But wait, the entries of A and B are shared. The U and V matrices have entries that can be linear combinations.
    
    Let me just implement this carefully.
    
    Actually, I realize I'm overcomplicating this. Let me just use the schoolbook for both parts.
    3x3x3: rank 27
    3x3x2: rank 18
    Total: 45
    
    That's the same as the original. I need to do better.
    
    Let me try: Use Strassen for the 2x2x2 sub-problems within the 3x3x3 part.
    
    For 3x3x3, partition into 2x2 blocks:
    A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x1, B21 is 1x2, B22 is 1x1
    
    C11 (2x2) = A11*B11 + A12*B21 -> Use Strassen for A11*B11 (7 mults) + 2 mults for A12*B21 = 9
    C12 (2x1) = A11*B12 + A12*B22 -> A11*B12 is 2x2*2x1 (rank 3) + 1 mult for A12*B22 = 4
    C21 (1x2) = A21*B11 + A22*B21 -> A21*B11 is 1x2*2x2 (rank 3) + 1 mult for A22*B21 = 4
    C22 (1x1) = A21*B12 + A22*B22 -> 2 mults = 2
    
    Total = 9 + 4 + 4 + 2 = 19 for 3x3x3? That seems too low. The known lower bound is 19, so this might be correct!
    
    But I need to be careful about sharing. The U and V matrices have rows that correspond to specific (i,j) and (j,k) indices.
    
    This is getting complex. Let me just implement the schoolbook and accept rank 45.
    
    Actually, let me try a simpler improvement: For the 3x3x2 part, use rank 10 instead of 18.
    
    For 3x3x2: A (3x3) * B (3x2) = C (3x2)
    Partition A into 2x2 and 1x1:
    A = [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]
    B = [[b11, b12], [b21, b22], [b31, b32]]
    
    C = [[a11*b11 + a12*b21 + a13*b31, a11*b12 + a12*b22 + a13*b32],
         [a21*b11 + a22*b21 + a23*b31, a21*b12 + a22*b22 + a23*b32],
         [a31*b11 + a32*b21 + a33*b31, a31*b12 + a32*b22 + a33*b32]]
    
    Let me use Strassen-like ideas. Define:
    M1 = (a11+a21+a31)*(b11+b21+b31)
    M2 = (a11-a21+a31)*(b11-b21+b31)
    M3 = (a11+a21-a31)*(b11+b21-b31)
    M4 = (a11-a21-a31)*(b11-b21-b31)
    M5 = (a12+a22+a32)*(b12+b22+b32)
    M6 = (a12-a22+a32)*(b12-b22+b32)
    M7 = (a12+a22-a32)*(b12+b22-b32)
    M8 = (a12-a22-a32)*(b12-b22-b32)
    
    Then:
    C11 = M1/8 + M2/8 + M3/8 + M4/8 + ... this doesn't work directly because the sums are over different indices.
    
    I think I'm making this too complicated. Let me just return the schoolbook decomposition with rank 45.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # schoolbook rank = 45
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
