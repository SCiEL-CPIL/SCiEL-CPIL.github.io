"""
Program for 3x4 x 4x6 matrix multiplication tensor decomposition.
Uses block decomposition where possible to exploit Strassen's algorithm for 2x2x2 blocks,
reducing the rank from the schoolbook 72 to a lower value.
"""

DIMENSIONS = [3, 4, 6]

import time
import itertools

# Deadline guard
START_TIME = time.time()
DEADLINE = START_TIME + 120  # 120 seconds max

def _elapsed():
    return time.time() - START_TIME

def _time_left():
    return DEADLINE - _elapsed()

# Strassen 2x2x2 decomposition
# U: 4 entries (A block rows), V: 4 entries (B block cols), W: 4 entries (C block entries)
# Standard Strassen for C = A*B where A, B are 2x2 blocks
# M1 = (A00+A11)(B00+B11)
# M2 = (A10+A11)B00
# M3 = A00(B01-B11)
# M4 = A11(B10-B00)
# M5 = (A00+A01)B11
# M6 = (A10-A00)(B00+B01)
# M7 = (A01-A11)(B10+B11)
# C00 = M1+M4-M5+M7
# C01 = M3+M5
# C10 = M2+M4
# C11 = M1-M2+M3+M6

def _strassen_2x2x2():
    """
    Returns (U, V, W) for a 2x2x2 block multiplication using Strassen.
    U shape: (2*2, 7) = (4, 7)
    V shape: (2*2, 7) = (4, 7)
    W shape: (2*2, 7) = (4, 7)
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # U rows: A00=0, A01=1, A10=2, A11=3
    # V rows: B00=0, B01=1, B10=2, B11=3
    # W rows: C00=0, C01=1, C10=2, C11=3
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # n=3, m=4, p=6
    
    # We decompose the 3x4 x 4x6 multiplication.
    # Split the 4x6 matrix B into 2x2 blocks where possible.
    # Let's split the rows of A (3 rows) and columns of B (6 cols) into blocks of 2 where possible.
    # 
    # Approach: Split the middle dimension m=4 into two 2x2 blocks, and the output columns p=6 into three 2x2 blocks.
    # For the input rows n=3, we can split into a 2-row block and a 1-row block.
    #
    # More systematically:
    # - Split i (rows of A, n=3) into blocks: [0,1] and [2]
    # - Split j (cols of A / rows of B, m=4) into blocks: [0,1] and [2,3]
    # - Split k (cols of B, p=6) into blocks: [0,1], [2,3], [4,5]
    #
    # Block (i_block, j_block, k_block) has size:
    #   i_size x j_size x k_size
    # 
    # Blocks:
    # (2x2, 2x2, 2x2) -> Strassen, rank 7
    # (2x2, 2x2, 2x2) -> Strassen, rank 7  (for k=[0,1])
    # (2x2, 2x2, 2x2) -> Strassen, rank 7  (for k=[2,3])
    # (2x2, 2x2, 2x2) -> Strassen, rank 7  (for k=[4,5])
    # (1x2, 2x2, 2x2) -> schoolbook, rank 1*2*2=4
    # (1x2, 2x2, 2x2) -> schoolbook, rank 4  (for k=[2,3])
    # (1x2, 2x2, 2x2) -> schoolbook, rank 4  (for k=[4,5])
    
    # Total rank = 4*7 + 3*4 = 28 + 12 = 40
    
    # Let's build the full U, V, W matrices.
    # U has n*m = 3*4 = 12 rows, V has m*p = 4*6 = 24 rows, W has p*n = 6*3 = 18 rows.
    
    # Block assignments:
    # i_blocks: [(0,2), (2,1)]  # (start, size)
    # j_blocks: [(0,2), (2,2)]
    # k_blocks: [(0,2), (2,2), (4,2)]
    
    i_blocks = [(0, 2), (2, 1)]
    j_blocks = [(0, 2), (2, 2)]
    k_blocks = [(0, 2), (2, 2), (4, 2)]
    
    total_rank = 0
    for (i_start, i_size) in i_blocks:
        for (j_start, j_size) in j_blocks:
            for (k_start, k_size) in k_blocks:
                if i_size == 2 and j_size == 2 and k_size == 2:
                    total_rank += 7
                else:
                    total_rank += i_size * j_size * k_size
    
    rank = total_rank
    
    # Build U, V, W
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col_offset = 0
    
    for (i_start, i_size) in i_blocks:
        for (j_start, j_size) in j_blocks:
            for (k_start, k_size) in k_blocks:
                if i_size == 2 and j_size == 2 and k_size == 2:
                    # Use Strassen
                    sU, sV, sW = _strassen_2x2x2()
                    # Map local indices to global
                    # U local row r corresponds to i=i_start+ri, j=j_start+cj where ri, cj in {0,1}
                    # U global row = i*m + j
                    # sU[ri*2+cj, c] -> U[(i_start+ri)*m + (j_start+cj), col_offset+c]
                    # V local row r corresponds to j=j_start+ri, k=k_start+cj
                    # V global row = j*p + k
                    # sV[ri*2+cj, c] -> V[(j_start+ri)*p + (k_start+cj), col_offset+c]
                    # W local row r corresponds to k=k_start+ri, i=i_start+cj
                    # W global row = k*n + i
                    # sW[ri*2+cj, c] -> W[(k_start+ri)*n + (i_start+cj), col_offset+c]
                    
                    for c in range(7):
                        for ri in range(2):
                            for cj in range(2):
                                # U
                                u_val = sU[ri*2+cj][c]
                                if u_val != 0:
                                    U[(i_start+ri)*m + (j_start+cj)][col_offset+c] += u_val
                                # V
                                v_val = sV[ri*2+cj][c]
                                if v_val != 0:
                                    V[(j_start+ri)*p + (k_start+cj)][col_offset+c] += v_val
                                # W
                                w_val = sW[ri*2+cj][c]
                                if w_val != 0:
                                    W[(k_start+ri)*n + (i_start+cj)][col_offset+c] += w_val
                    
                    col_offset += 7
                else:
                    # Schoolbook for i_size x j_size x k_size
                    for ri in range(i_size):
                        for cj in range(j_size):
                            for ck in range(k_size):
                                U[(i_start+ri)*m + (j_start+cj)][col_offset] = 1
                                V[(j_start+cj)*p + (k_start+ck)][col_offset] = 1
                                W[(k_start+ck)*n + (i_start+ri)][col_offset] = 1
                                col_offset += 1
    
    return {"U": U, "V": V, "W": W}

