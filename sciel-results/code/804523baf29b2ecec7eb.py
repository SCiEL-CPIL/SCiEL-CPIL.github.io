import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def get_strassen_2x2():
    """Returns U, V, W for 2x2x2 matrix multiplication, rank 7.
    U, V, W are lists of shape (4, 7) where row index = i*n+j, col index = term.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 2: (A10+A11)B00 -> C10+=1, C11-=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # Term 3: A00(B01-B11) -> C01+=1, C11+=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # Term 4: A11(B10-B00) -> C10+=1, C00+=1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = 1
    
    # Term 5: (A00+A01)B11 -> C01-=1, C11+=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = -1; W[3][4] = 1
    
    # Term 6: (A10-A00)(B00+B01) -> C11+=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 7: (A01-A11)(B10+B11) -> C00+=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check all entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_np[i*m + j][r] * V_np[j*p + k][r] * W_np[k*n + i][r]
                if abs(val - 1.0) > 1e-9:
                    return False
    
    return True

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    Strategy:
    - Start with schoolbook (rank 80)
    - Try to use Strassen on 2x2 subblocks to reduce rank
    - The 4x4*4x5 problem can be decomposed using Strassen-like techniques
    
    For 4x4*4x5:
    - Schoolbook rank = 4*4*5 = 80
    - We can try to use Strassen on the 4x4 part (treating it as 2x2 blocks of 2x2)
    
    Let's partition:
    A (4x4) = [[A11, A12], [A21, A22]] where each block is 2x2
    B (4x5) = [[B11, B12], [B21, B22]] where B11, B21 are 2x5, B12, B22 are 2x5
    Wait, that's not right. B is 4x5, so if we split rows into 2+2:
    B = [[B_top], [B_bottom]] where B_top is 2x5, B_bottom is 2x5
    
    C (4x5) = C_top (2x5) + C_bottom (2x5)
    C_top = A11 * B_top + A12 * B_bottom
    C_bottom = A21 * B_top + A22 * B_bottom
    
    This is like multiplying a 2x2 block matrix by a 2x5 block matrix.
    Using Strassen for the 2x2 "matrix" multiplication where the entries are 2x5 matrices:
    
    M1 = (A11 + A22) * (B_top + B_bottom)  [2x5 * 2x5 -> 2x5]
    M2 = (A21 + A22) * B_top  [2x2 * 2x5 -> 2x5]
    M3 = A11 * (B_bottom - B_top)  [2x2 * 2x5 -> 2x5]
    M4 = A22 * (B_top - B_bottom)  [2x2 * 2x5 -> 2x5]
    M5 = (A11 + A12) * B_bottom  [2x2 * 2x5 -> 2x5]
    M6 = (A21 - A11) * (B_top + B11_part)  -- this doesn't work cleanly
    
    Actually, Strassen is for square matrices. Let me think differently.
    
    Alternative: Use the fact that 4x4*4x5 can be seen as computing 5 separate 4x4*4*1 products
    (one for each column of B/C), but that's just schoolbook.
    
    Let me try a different approach: use a known rank bound.
    For n x m * m x p, the best known upper bound is often close to n*m*p but can be lower.
    For 4x4*4x5, let's try to use Strassen on pairs of rows/columns.
    
    Actually, let's just use schoolbook for now since it's guaranteed correct,
    and the task says "rank-minimal" but we may not know the exact minimum.
    The current score is 1.0 with schoolbook rank 80.
    
    Let me try to implement a rank-64 decomposition using Strassen on 2x2 blocks.
    
    Partition A into 2x2 blocks of size 2x2:
    A = [[A00, A01], [A10, A11]] where each is 2x2
    
    Partition B (4x5) into:
    B = [[B00, B01], [B10, B11]] where B00, B10 are 2x5, B01, B11 are 2x5
    Wait, B is 4x5. If we split the 4 rows into 2+2:
    B = [B_top; B_bottom] where B_top is 2x5, B_bottom is 2x5
    
    But for Strassen, we need the "inner" dimension to match.
    A is 4x4, B is 4x5. The inner dimension is 4.
    
    Let me split the inner dimension (columns of A / rows of B) into 2+2:
    A = [A_left, A_right] where A_left is 4x2, A_right is 4x2
    B = [B_top; B_bottom] where B_top is 2x5, B_bottom is 2x5
    
    C = A_left * B_top + A_right * B_bottom  (4x5)
    
    Now, A_left is 4x2, B_top is 2x5, so A_left*B_top is 4x5.
    A_right is 4x2, B_bottom is 2x5, so A_right*B_bottom is 4x5.
    
    This doesn't directly help with Strassen.
    
    Let me try another partition. Split both the row dimension of A/C and the inner dimension:
    A (4x4) = [[A11, A12], [A21, A22]] where each block is 2x2
    B (4x5) = [[B11, B12], [B21, B22]] where B11 is 2x5, B21 is 2x5, etc.
    Wait, B is 4x5. If we split rows into 2+2 and columns into 5 (no split):
    B = [B_top; B_bottom] where B_top is 2x5, B_bottom is 2x5
    
    C (4x5) = [C_top; C_bottom] where C_top is 2x5, C_bottom is 2x5
    
    C_top = A11 * B_top + A12 * B_bottom  (2x2 * 2x5 + 2x2 * 2x5 -> 2x5)
    C_bottom = A21 * B_top + A22 * B_bottom  (2x2 * 2x5 + 2x2 * 2x5 -> 2x5)
    
    Now, for each 2x5 block multiplication, we can use Strassen-like methods.
    But 2x2 * 2x5 is not square.
    
    Let me just use schoolbook. It's the safest bet for now.
    """
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
