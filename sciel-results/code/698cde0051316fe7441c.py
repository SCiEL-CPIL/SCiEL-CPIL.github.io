"""
Rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.

Strategy:
  - Use Strassen's 7-multiplication algorithm for 2x2x2 sub-blocks.
  - Use schoolbook (rank = product of dims) for remaining blocks.
  - Block decomposition:
      A (3x4) split as A1 (1x4) on top, A2 (2x4) below.
      B (4x5) split as B1 (4x1) on left, B2 (4x4) on right.
    Result C = A1*B1 (1x1) | A1*B2 (1x4)
              A2*B1 (2x1) | A2*B2 (2x4)

    Ranks:
      A1*B1: 1x1*4x1 = 1x1, rank 4 (schoolbook)
      A1*B2: 1x4*4x4 = 1x4, rank 16 (schoolbook)
      A2*B1: 2x4*4x1 = 2x1, rank 8 (schoolbook)
      A2*B2: 2x4*4x4 = 2x4, rank 28 (4 Strassen blocks)
      
    Total rank: 4 + 16 + 8 + 28 = 56
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def _strassen_2x2x2():
    """
    Returns (U, V, W) for 2x2x2 matrix multiplication using Strassen's algorithm.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    
    For A (2x2) * B (2x2) = C (2x2):
    Strassen's 7 products:
      M1 = (A11 + A22) * (B11 + B22)
      M2 = (A21 + A22) * B11
      M3 = A11 * (B12 - B22)
      M4 = A22 * (B21 - B11)
      M5 = (A11 + A12) * B22
      M6 = (A21 - A11) * (B11 + B12)
      M7 = (A12 - A22) * (B21 + B22)
      
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Indexing: A[i,j] -> U[i*2+j, col], B[i,j] -> V[i*2+j, col], C[i,j] -> W[i*2+j, col]
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (A11+A22)*(B11+B22)
    U[0][0] = 1; U[3][0] = 1  # A11, A22
    V[0][0] = 1; V[3][0] = 1  # B11, B22
    W[0][0] = 1  # C11
    W[3][0] = 1  # C22
    
    # M2 = (A21+A22)*B11
    U[2][1] = 1; U[3][1] = 1  # A21, A22
    V[0][1] = 1  # B11
    W[2][1] = 1  # C21
    
    # M3 = A11*(B12-B22)
    U[0][2] = 1  # A11
    V[1][2] = 1; V[3][2] = -1  # B12, -B22
    W[1][2] = 1  # C12
    W[3][2] = 1  # C22
    
    # M4 = A22*(B21-B11)
    U[3][3] = 1  # A22
    V[2][3] = 1; V[0][3] = -1  # B21, -B11
    W[0][3] = 1  # C11
    W[2][3] = 1  # C21
    
    # M5 = (A11+A12)*B22
    U[0][4] = 1; U[1][4] = 1  # A11, A12
    V[3][4] = 1  # B22
    W[0][4] = -1  # C11
    W[1][4] = 1  # C12
    
    # M6 = (A21-A11)*(B11+B12)
    U[2][5] = 1; U[0][5] = -1  # A21, -A11
    V[0][5] = 1; V[1][5] = 1  # B11, B12
    W[3][5] = 1  # C22
    
    # M7 = (A12-A22)*(B21+B22)
    U[1][6] = 1; U[3][6] = -1  # A12, -A22
    V[2][6] = 1; V[3][6] = 1  # B21, B22
    W[0][6] = 1  # C11
    
    return U, V, W

def _schoolbook(n, m, p):
    """
    Schoolbook decomposition for n*m * m*p = n*p.
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    Note: W uses trace convention W[k*n+i, col]
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _combine_blocks(blocks):
    """
    Combine a list of (U, V, W) blocks into a single decomposition.
    Each block has U: (nu, r), V: (nv, r), W: (nw, r)
    We stack them column-wise.
    """
    total_rank = sum(len(b[0][0]) for b in blocks)
    nu = len(blocks[0][0])
    nv = len(blocks[0][1])
    nw = len(blocks[0][2])
    
    U = [[0]*total_rank for _ in range(nu)]
    V = [[0]*total_rank for _ in range(nv)]
    W = [[0]*total_rank for _ in range(nw)]
    
    col_offset = 0
    for U_b, V_b, W_b in blocks:
        r = len(U_b[0])
        for i in range(nu):
            for j in range(r):
                U[i][col_offset+j] = U_b[i][j]
        for i in range(nv):
            for j in range(r):
                V[i][col_offset+j] = V_b[i][j]
        for i in range(nw):
            for j in range(r):
                W[i][col_offset+j] = W_b[i][j]
        col_offset += r
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Block decomposition:
    # A (3x4): A1 (1x4) rows 0, A2 (2x4) rows 1-2
    # B (4x5): B1 (4x1) cols 0, B2 (4x4) cols 1-4
    # C (3x5): C11 (1x1), C12 (1x4), C21 (2x1), C22 (2x4)
    
    blocks = []
    
    # C11 = A1 * B1: 1x4 * 4x1 = 1x1, rank 4
    blocks.append(_schoolbook(1, 4, 1))
    
    # C12 = A1 * B2: 1x4 * 4x4 = 1x4, rank 16
    blocks.append(_schoolbook(1, 4, 4))
    
    # C21 = A2 * B1: 2x4 * 4x1 = 2x1, rank 8
    blocks.append(_schoolbook(2, 4, 1))
    
    # C22 = A2 * B2: 2x4 * 4x4 = 2x4, rank 28 via 4 Strassen blocks
    # Split A2 (2x4) into A2a (2x2) cols 0-1, A2b (2x2) cols 2-3
    # Split B2 (4x4) into B2l (4x2) cols 0-1, B2r (4x2) cols 2-3
    # For each column block of B2:
    #   A2 * B2col = A2a * B2col_top + A2b * B2col_bot
    #   where B2col_top is top 2x2, B2col_bot is bottom 2x2
    #   Each product is 2x2 * 2x2 = 2x2, rank 7 (Strassen)
    #   Total per column block: 14
    #   Total for C22: 28
    
    for s in range(2):  # s=0: B2l, s=1: B2r
        for t in range(2):  # t=0: A2a*B2?, t=1: A2b*B2?
            U_s, V_s, W_s = _strassen_2x2x2()
            blocks.append((U_s, V_s, W_s))
    
    U, V, W = _combine_blocks(blocks)
    
    return {"U": U, "V": V, "W": W}
