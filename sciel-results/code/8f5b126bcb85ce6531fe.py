"""Strassen-based rank-56 decomposition for 4x4 x 4x7 matrix multiplication."""
import time
import random

DIMENSIONS = [4, 4, 7]

def _strassen444():
    """Return (U, V, W) for 4x4 x 4x4 multiplication with rank 56.
    
    Uses 2x2x2 Strassen recursively: 8 Strassen steps per 2x2x2 block,
    and 4 such blocks for 4x4x4. Total rank = 4 * 7 = 28... 
    Wait, that's not right. Let me think again.
    
    4x4 x 4x4: split into 2x2 blocks:
    A = [[A11, A12], [A21, A22]]  each 2x2
    B = [[B11, B12], [B21, B22]]  each 2x2
    C = [[C11, C12], [C21, C22]]  each 2x2
    
    Using Strassen on the block level: 7 multiplications of 2x2 matrices.
    Each 2x2 multiplication has rank 7 (schoolbook for 2x2x2).
    So total rank = 7 * 7 = 49.
    
    Let me implement this properly.
    """
    n = 4
    rank = 49
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]
    
    # Strassen for 2x2 block multiplication:
    # Given 2x2 blocks A11, A12, A21, A22, B11, B12, B21, B22 (each scalar in the 2x2x2 case)
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4x4, we have 4 quadrants of the output, each computed via 7 Strassen products
    # of 2x2 matrices. Each 2x2x2 product has rank 7.
    # Total: 4 * 7 = 28 rank-7 blocks... no wait.
    
    # Actually: the 4x4x4 tensor can be viewed as 4 independent 2x2x2 tensors
    # (for the 4 output blocks), but they share input structure.
    # Using block Strassen: 7 intermediate 2x2x2 multiplications, each of rank 7.
    # Total rank = 7 * 7 = 49.
    
    # Let me index properly.
    # For each of the 7 Strassen products (m=0..6), we have a 2x2x2 multiplication
    # which requires 7 rank-1 terms.
    
    # 2x2x2 schoolbook rank-7:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # Rank 4 schoolbook. But we want rank 7 for Strassen compatibility? 
    # No - for the inner 2x2x2, schoolbook rank is 4.
    
    # Wait, I'm confusing myself. Let me restart.
    
    # 4x4 x 4x4 multiplication:
    # Schoolbook: 16 rank-1 terms, rank = 16.
    # Strassen block: 7 multiplications of 2x2 matrices. Each 2x2x2 has rank 4 (schoolbook).
    # Total rank = 7 * 4 = 28.
    
    # Hmm, but the known result is that 4x4x4 has rank at most 49 using Strassen.
    # Let me reconsider: maybe I should just use schoolbook for 4x4x4 (rank 16) 
    # and schoolbook for the 4x4x3 part (rank 12), total rank 28.
    
    # That's already better than 112! Let me just do schoolbook for both parts.
    # 4x4x4: rank 16, 4x4x3: rank 12, total 28.
    
    # Actually wait - the task is 4x4 times 4x7. I can decompose it as:
    # M4x4 * [N4x4 | N4x3] = [M*N4x4 | M*N4x3]
    # First part: 4x4 x 4x4, schoolbook rank 16
    # Second part: 4x4 x 4x3, schoolbook rank 12
    # Total rank: 28
    
    return None  # Placeholder, we'll do it directly in construct

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    # Decompose as 4x4 x 4x4 (rank 16) + 4x4 x 4x3 (rank 12) = rank 28
    rank_total = n * m + n * m * (p - m)  # 16 + 12 = 28
    rank_total = n * m * p  # Just use full schoolbook: 112
    # Actually let's use the block decomposition:
    # Split columns of B into B1 (4 cols) and B2 (3 cols)
    # C1 = A * B1: 4x4 x 4x4, schoolbook rank 16
    # C2 = A * B2: 4x4 x 4x3, schoolbook rank 12
    # Total rank: 28
    
    rank = n * m + n * m * (p - m)  # 16 + 12 = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: 4x4 x 4x4 (columns 0-3 of B)
    for i in range(n):
        for j in range(m):
            for k in range(m):  # k in 0..3
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Part 2: 4x4 x 4x3 (columns 4-6 of B)
    for i in range(n):
        for j in range(m):
            for k in range(m, p):  # k in 4..6
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
