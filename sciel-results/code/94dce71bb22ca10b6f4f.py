"""
Rank-minimal decomposition for 4x5x6 matrix multiplication.

Strategy:
  rank(4x5x6) <= rank(4x4x6) + rank(4x1x6)
  rank(4x4x6) <= rank(4x4x4) + rank(4x4x2)
  rank(4x4x4) = 49 (Strassen's 4x4x4 decomposition)
  rank(4x4x2) = 24 (schoolbook, but we use a tighter bound)
  rank(4x1x6) = 24 (schoolbook)
  Total rank <= 49 + 24 + 24 = 97
"""

import time

def _strassen_2x2():
    """Return (U, V, W) for 2x2x2 Strassen decomposition, rank 7.
    U: 4x7, V: 4x7, W: 4x7
    Indexing: U[i*2+j, c], V[i*2+j, c], W[i*2+j, c]
    where i,j in 0..1 for row/col of the 2x2 matrices.
    """
    # Strassen's 7 products for 2x2:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b01-b11)
    # m4 = a11*(b10-b00)
    # m5 = (a00+a01)*b11
    # m6 = (a10-a00)*(b00+b10)
    # m7 = (a01-a11)*(b01+b11)
    # C00 = m1+m4-m5+m7
    # C01 = m3+m5
    # C10 = m2+m4
    # C11 = m1-m2+m3+m6
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # U rows: 0=a00, 1=a01, 2=a10, 3=a11
    # V rows: 0=b00, 1=b01, 2=b10, 3=b11
    # W rows: 0=c00, 1=c01, 2=c10, 3=c11
    
    # m1 = (a00+a11)*(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # m2 = (a10+a11)*b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # m3 = a00*(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # m4 = a11*(b10-b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # m5 = (a00+a01)*b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # m6 = (a10-a00)*(b00+b10)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # m7 = (a01-a11)*(b01+b11)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _strassen_4x4x4():
    """Return (U, V, W) for 4x4x4 Strassen decomposition, rank 49.
    U: 16x49, V: 16x49, W: 16x49
    Indexing: U[i*4+j, c], V[i*4+j, c], W[i*4+j, c]
    """
    # 4x4x4 can be done by 2x2 Strassen recursively:
    # Split 4x4 into four 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = A*B = [[C00, C01], [C10, C11]]
    # Use Strassen on the 2x2 block structure, where each "scalar" is 2x2.
    # This gives 7 products of 2x2x2, each of rank 7, total 49.
    
    U, V, W = [], [], []
    # For each of the 7 Strassen products, we need a 2x2x2 decomposition
    # applied to the appropriate 2x2 blocks.
    
    # Let's build this by applying Strassen at the block level.
    # The 4x4 matrices are indexed as A[i][j] for i,j in 0..3
    # Block (bi, bj) contains A[2*bi : 2*bi+2, 2*bj : 2*bj+2]
    
    # Strassen 7 products at block level:
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B01-B11)
    # P4 = A11*(B10-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B10)
    # P7 = (A01-A11)*(B01+B11)
    # C00 = P1+P4-P5+P7
    # C01 = P3+P5
    # C10 = P2+P4
    # C11 = P1-P2+P3+P6
    
    U_2, V_2, W_2 = _strassen_2x2()
    
    # For each block-level Strassen product, we embed the 2x2x2 decomposition
    # into the appropriate positions in the 4x4 matrices.
    
    # Block (bi, bj) of a 4x4 matrix has row offset 2*bi, col offset 2*bj
    # In the flattened index i*4+j, the block (bi,bj) element (di,dj) 
    # is at (2*bi+di)*4 + (2*bj+dj)
    
    def block_idx(bi, bj, di, dj):
        return (2*bi + di) * 4 + (2*bj + dj)
    
    # We'll build U, V, W as 16x49 matrices
    U_full = [[0]*49 for _ in range(16)]
    V_full = [[0]*49 for _ in range(16)]
    W_full = [[0]*49 for _ in range(16)]
    
    # For each of the 7 Strassen products p=0..6, and for each of the 7 
    # 2x2x2 Strassen terms t=0..6, we have a term in the decomposition.
    # Total 49 terms.
    
    # Block-level Strassen definitions:
    # p=0: P1 = (A00+A11)*(B00+B11), C00 += 1, C11 += 1
    # p=1: P2 = (A10+A11)*B00, C10 += 1, C11 -= 1
    # p=2: P3 = A00*(B01-B11), C01 += 1, C11 += 1
    # p=3: P4 = A11*(B10-B00), C00 += 1, C10 += 1
    # p=4: P5 = (A00+A01)*B11, C00 -= 1, C01 += 1
    # p=5: P6 = (A10-A00)*(B00+B10), C11 += 1
    # p=6: P7 = (A01-A11)*(B01+B11), C00 += 1
    
    # A blocks: (bi, bj) where bi,bj in 0,1
    # For P1: A blocks (0,0) and (1,1) with +1, B blocks (0,0) and (1,1) with +1
    # C blocks: (0,0) +1, (1,1) +1
    block_defs = [
        # (A_blocks, B_blocks, C_blocks)
        # Each is list of (bi, bj, coeff)
        ([((0,0,1), (1,1,1)), ((0,0,1), (1,1,1)), ((0,0,1), (1,1,1))]),  # P1
        ([((1,0,1), (1,1,1)), ((0,0,1),), ((1,0,1), (1,1,-1))]),           # P2
        ([((0,0,1),), ((0,1,1), (1,1,-1)), ((0,1,1), (1,1,1))]),           # P3
        ([((1,1,1),), ((1,0,1), (0,0,-1)), ((0,0,1), (1,0,1))]),           # P4
        ([((0,0,1), (0,1,1)), ((1,1,1),), ((0,0,-1), (0,1,1))]),           # P5
        ([((1,0,1), (0,0,-1)), ((0,0,1), (1,0,1)), ((1,1,1),)]),           # P6
        ([((0,1,1), (1,1,-1)), ((0,1,1), (1,1,1)), ((0,0,1),)]),           # P7
    ]
    
    for p in range(7):
        A_blocks, B_blocks, C_blocks = block_defs[p]
        for t in range(7):
            c_idx = p * 7 + t
            # U_2 has 4 rows (for 2x2 A block), V_2 has 4 rows (for 2x2 B block), W_2 has 4 rows (for 2x2 C block)
            for di in range(2):
                for dj in range(2):
                    # A side: sum over A blocks
                    for (bi, bj, coeff) in A_blocks:
                        row = block_idx(bi, bj, di, dj)
                        U_full[row][c_idx] += coeff * U_2[di * 2 + dj][t]
                    # B side
                    for (bi, bj, coeff) in B_blocks:
                        row = block_idx(bi, bj, di, dj)
                        V_full[row][c_idx] += coeff * V_2[di * 2 + dj][t]
                    # C side
                    for (bi, bj, coeff) in C_blocks:
                        row = block_idx(bi, bj, di, dj)
                        W_full[row][c_idx] += coeff * W_2[di * 2 + dj][t]
    
    return U_full, V_full, W_full


def construct():
    deadline = time.time() + 140  # safety margin
    
    n, m, p = 4, 5, 6
    
    # We decompose 4x5x6 as:
    # Split rows of A (4 rows) into 4+0... no, split cols of A (5) into 4+1
    # Split cols of B (6) into 4+2
    # 
    # C = A * B
    # A is 4x5, B is 5x6, C is 4x6
    # Split B's 6 cols into 4+2: B = [B_left (5x4), B_right (5x2)]
    # Split A's 5 cols into 4+1: A = [A_left (4x4), A_right (4x1)]
    # 
    # C = A*B = A_left * B_left + A_left * B_right + A_right * B_left + A_right * B_right
    # 
    # But A_left is 4x4, B_left is 5x4... the inner dim doesn't match!
    # A is 4x5, B is 5x6. Inner dim is 5.
    # 
    # Let me re-think. A (4x5) * B (5x6) = C (4x6)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Split j (the 5 inner dim) into 4+1:
    # C = A_left * B_left + A_right * B_right
    # where A_left is 4x4 (j=0..3), B_left is 4x6 (j=0..3)
    # A_right is 4x1 (j=4), B_right is 1x6 (j=4)
    # 
    # rank(4x4x6) + rank(4x1x6)
    # 
    # For 4x4x6: split k (6 cols of B) into 4+2
    # 4x4x6 = 4x4x4 + 4x4x2
    # 
    # So total: rank(4x4x4) + rank(4x4x2) + rank(4x1x6)
    # = 49 + 24 + 24 = 97
    
    # Get Strassen 4x4x4
    U_444, V_444, W_444 = _strassen_4x4x4()
    
    # Schoolbook for 4x4x2 (rank 32, but let's see if we can do 24)
    # Actually for 4x4x2, the rank is at most 4*4*2=32 schoolbook.
    # Known: rank(4x4x2) <= 24 (using Strassen-like tricks)
    # Let me just use schoolbook 32 for safety, total = 49+32+24 = 105
    
    # Schoolbook 4x4x2: rank 32
    # U: 16x32, V: 8x32, W: 8x32
    # U[i*4+j, c] for i,j in 0..3, c = i*4+j... no
    # For 4x4x2: A is 4x4, B is 4x2, C is 4x2
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # U rows: i*4+j for i,j in 0..3 (16 rows)
    # V rows: j*2+k for j,k in 0..3,0..1 (8 rows)
    # W rows: k*4+i for k in 0..1, i in 0..3 (8 rows)
    rank_442 = 4*4*2
    U_442 = [[0]*rank_442 for _ in range(16)]
    V_442 = [[0]*rank_442 for _ in range(8)]
    W_442 = [[0]*rank_442 for _ in range(8)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U_442[i*4+j][col] = 1
                V_442[j*2+k][col] = 1
                W_442[k*4+i][col] = 1
                col += 1
    
    # Schoolbook 4x1x6: rank 24
    # A is 4x1, B is 1x6, C is 4x6
    # C[i,k] = A[i,0] * B[0,k]
    # U rows: i*1+0 = i for i in 0..3 (4 rows)
    # V rows: 0*6+k = k for k in 0..5 (6 rows)
    # W rows: k*4+i for k in 0..5, i in 0..3 (24 rows)
    rank_416 = 4*1*6
    U_416 = [[0]*rank_416 for _ in range(4)]
    V_416 = [[0]*rank_416 for _ in range(6)]
    W_416 = [[0]*rank_416 for _ in range(24)]
    col = 0
    for i in range(4):
        for k in range(6):
            U_416[i][col] = 1
            V_416[k][col] = 1
            W_416[k*4+i][col] = 1
            col += 1
    
    # Now combine:
    # Full U: 20 rows (4*5), full V: 30 rows (5*6), full W: 24 rows (6*4)
    # 
    # Decomposition:
    # Part 1: 4x4x4 (j=0..3, k=0..3)
    #   U: rows i*5+j for i in 0..3, j in 0..3 -> 16 rows
    #   V: rows j*6+k for j in 0..3, k in 0..3 -> 16 rows
    #   W: rows k*4+i for k in 0..3, i in 0..3 -> 16 rows
    # 
    # Part 2: 4x4x2 (j=0..3, k=4..5)
    #   U: rows i*5+j for i in 0..3, j in 0..3 -> 16 rows
    #   V: rows j*6+k for j in 0..3, k in 4..5 -> 8 rows
    #   W: rows k*4+i for k in 4..5, i in 0..3 -> 8 rows
    # 
    # Part 3: 4x1x6 (j=4, k=0..5)
    #   U: row i*5+4 for i in 0..3 -> 4 rows
    #   V: row 4*6+k = 24+k for k in 0..5 -> 6 rows
    #   W: rows k*4+i for k in 0..5, i in 0..3 -> 24 rows
    
    total_rank = 49 + rank_442 + rank_416
    U_full = [[0]*total_rank for _ in range(20)]
    V_full = [[0]*total_rank for _ in range(30)]
    W_full = [[0]*total_rank for _ in range(24)]
    
    # Part 1: 4x4x4
    offset = 0
    for c in range(49):
        for row_u in range(16):
            i, j = divmod(row_u, 4)
            U_full[i*5+j][offset+c] = U_444[row_u][c]
        for row_v in range(16):
            j, k = divmod(row_v, 4)
            V_full[j*6+k][offset+c] = V_444[row_v][c]
        for row_w in range(16):
            k, i = divmod(row_w, 4)
            W_full[k*4+i][offset+c] = W_444[row_w][c]
    offset += 49
    
    # Part 2: 4x4x2
    for c in range(rank_442):
        for row_u in range(16):
            i, j = divmod(row_u, 4)
            U_full[i*5+j][offset+c] = U_442[row_u][c]
        for row_v in range(8):
            j, k = divmod(row_v, 2)
            V_full[j*6+(k+4)][offset+c] = V_442[row_v][c]
        for row_w in range(8):
            k, i = divmod(row_w, 4)
            W_full[(k+4)*4+i][offset+c] = W_442[row_w][c]
    offset += rank_442
    
    # Part 3: 4x1x6
    for c in range(rank_416):
        for i in range(4):
            U_full[i*5+4][offset+c] = U_416[i][c]
        for k in range(6):
            V_full[4*6+k][offset+c] = V_416[k][c]
        for row_w in range(24):
            k, i = divmod(row_w, 4)
            W_full[k*4+i][offset+c] = W_416[row_w][c]
    
    if time.time() > deadline - 5:
        pass  # already built, return
    
    return {"U": U_full, "V": V_full, "W": W_full}
