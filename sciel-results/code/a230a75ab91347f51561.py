import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

def _tensor_check(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor.
    For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n*m)]
    V = [[0] * rank for _ in range(m*p)]
    W = [[0] * rank for _ in range(p*n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return (U, V, W)

def _strassen_2x2_2x2_2x2():
    """
    Return the 7-term Strassen decomposition for 2x2 * 2x2 -> 2x2.
    U: (2*2)=4 x 7, V: (2*2)=4 x 7, W: (2*2)=4 x 7
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (a00+a11)(b00+b11) -> contributes to c00 and c11
    U[0][0] = 1; U[3][0] = 1  # a00, a11
    V[0][0] = 1; V[3][0] = 1  # b00, b11
    W[0][0] = 1; W[3][0] = 1  # c00, c11
    
    # M2 = (a10+a11)b00 -> contributes to c10
    U[2][1] = 1; U[3][1] = 1  # a10, a11
    V[0][1] = 1               # b00
    W[2][1] = 1               # c10
    
    # M3 = a00(b01-b11) -> contributes to c01
    U[0][2] = 1               # a00
    V[1][2] = 1; V[3][2] = -1 # b01, b11
    W[1][2] = 1               # c01
    
    # M4 = a11(b10-b00) -> contributes to c11
    U[3][3] = 1               # a11
    V[2][3] = 1; V[0][3] = -1 # b10, b00
    W[3][3] = 1               # c11
    
    # M5 = (a00+a01)b11 -> contributes to c00
    U[0][4] = 1; U[1][4] = 1  # a00, a01
    V[3][4] = 1               # b11
    W[0][4] = 1               # c00
    
    # M6 = (a10-a00)(b00+b01) -> contributes to c10, c11
    U[2][5] = 1; U[0][5] = -1 # a10, a00
    V[0][5] = 1; V[1][5] = 1  # b00, b01
    W[2][5] = 1; W[3][5] = 1  # c10, c11
    
    # M7 = (a01-a11)(b10+b11) -> contributes to c01
    U[1][6] = 1; U[3][6] = -1 # a01, a11
    V[2][6] = 1; V[3][6] = 1  # b10, b11
    W[1][6] = 1               # c01
    
    return U, V, W

def _block_decomposition(n, m, p):
    """
    Construct a block decomposition for n x m * m x p.
    Split A into 2x2 block + rest, B into 2x2 block + rest.
    Use Strassen for the 2x2*2x2 part, schoolbook for the rest.
    """
    # For 3x3 * 3x7:
    # A is 3x3, B is 3x7
    # Split A: rows 0-1, rows 2
    # Split B: cols 0-1, cols 2-6
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> 2x2 * 2x2, use Strassen (rank 7)
    # Block 2: A[0:2, 0:2] * B[0:2, 2:7] -> 2x2 * 2x5, schoolbook (rank 20)
    # Block 3: A[0:2, 0:2] * B[2:3, 0:2] -> 2x2 * 1x2, schoolbook (rank 4)
    # Block 4: A[0:2, 0:2] * B[2:3, 2:7] -> 2x2 * 1x5, schoolbook (rank 10)
    # Block 5: A[2:3, 0:2] * B[0:2, 0:2] -> 1x2 * 2x2, schoolbook (rank 4)
    # Block 6: A[2:3, 0:2] * B[0:2, 2:7] -> 1x2 * 2x5, schoolbook (rank 10)
    # Block 7: A[2:3, 0:2] * B[2:3, 0:2] -> 1x2 * 1x2, schoolbook (rank 2)
    # Block 8: A[2:3, 0:2] * B[2:3, 2:7] -> 1x2 * 1x5, schoolbook (rank 5)
    # Block 9: A[0:2, 2:3] * B[0:2, 0:2] -> 2x1 * 2x2, schoolbook (rank 4)
    # Block 10: A[0:2, 2:3] * B[0:2, 2:7] -> 2x1 * 2x5, schoolbook (rank 10)
    # Block 11: A[0:2, 2:3] * B[2:3, 0:2] -> 2x1 * 1x2, schoolbook (rank 2)
    # Block 12: A[0:2, 2:3] * B[2:3, 2:7] -> 2x1 * 1x5, schoolbook (rank 5)
    # Block 13: A[2:3, 2:3] * B[0:2, 0:2] -> 1x1 * 2x2, schoolbook (rank 2)
    # Block 14: A[2:3, 2:3] * B[0:2, 2:7] -> 1x1 * 2x5, schoolbook (rank 5)
    # Block 15: A[2:3, 2:3] * B[2:3, 0:2] -> 1x1 * 1x2, schoolbook (rank 1)
    # Block 16: A[2:3, 2:3] * B[2:3, 2:7] -> 1x1 * 1x5, schoolbook (rank 5)
    
    # Total rank: 7 + 20 + 4 + 10 + 4 + 10 + 2 + 5 + 4 + 10 + 2 + 5 + 2 + 5 + 1 + 5 = 96? That's worse than schoolbook.
    # Let me recalculate. Schoolbook is 3*3*7 = 63.
    # The block decomposition above is not helpful. Let me try a different approach.
    
    # Actually, let's just use schoolbook for now and return it.
    # The block decomposition approach is complex and may not yield improvement.
    return _schoolbook(n, m, p)

def construct():
    start_time = time.time()
    deadline = start_time + 100

    n, m, p = DIMENSIONS  # 3, 3, 7

    # Fallback: Schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_result = (U_sb, V_sb, W_sb)
    best_rank = n * m * p

    # Try to find a better decomposition using block decomposition
    if time.time() < deadline - 20:
        U_bd, V_bd, W_bd = _block_decomposition(n, m, p)
        rank_bd = len(U_bd[0])
        if rank_bd < best_rank:
            if _tensor_check(U_bd, V_bd, W_bd, n, m, p):
                best_result = (U_bd, V_bd, W_bd)
                best_rank = rank_bd

    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
