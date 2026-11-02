"""Matrix multiplication 2x4x5 tensor decomposition using Strassen blocks."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition is exact using the tensor check."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    # Also verify that non-matching triples sum to 0
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                if j_a == j_b and k_b == k_c and i_a == i_c:
                    continue
                s = 0
                for col in range(rank):
                    s += U[a][col] * V[b][col] * W[c][col]
                if abs(s) > 1e-9:
                    return False
    return True

def _strassen_2x2x2(offset_u, offset_v, offset_w, U, V, W, m, p, n, col_start):
    """
    Add a Strassen 2x2x2 block at given offsets.
    U rows: offset_u to offset_u+3 (i=0,1; j=0,1)
    V rows: offset_v to offset_v+3 (j=0,1; k=0,1)
    W rows: offset_w to offset_w+3 (k=0,1; i=0,1)
    Returns number of columns used (7).
    """
    # Strassen's 7 products for 2x2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    col = col_start
    
    # M1 = (A00+A11)*(B00+B11)
    U[offset_u + 0][col] = 1  # A00
    U[offset_u + 3][col] = 1  # A11
    V[offset_v + 0][col] = 1  # B00
    V[offset_v + 3][col] = 1  # B11
    W[offset_w + 0][col] = 1  # C00 += M1
    W[offset_w + 3][col] = 1  # C11 += M1
    col += 1
    
    # M2 = (A10+A11)*B00
    U[offset_u + 2][col] = 1  # A10
    U[offset_u + 3][col] = 1  # A11
    V[offset_v + 0][col] = 1  # B00
    W[offset_w + 2][col] = 1  # C10 += M2
    W[offset_w + 3][col] = -1 # C11 -= M2
    col += 1
    
    # M3 = A00*(B01-B11)
    U[offset_u + 0][col] = 1  # A00
    V[offset_v + 1][col] = 1  # B01
    V[offset_v + 3][col] = -1 # B11
    W[offset_w + 0][col] = 1  # C00 += M3
    W[offset_w + 3][col] = 1  # C11 += M3
    col += 1
    
    # M4 = A11*(B10-B00)
    U[offset_u + 3][col] = 1  # A11
    V[offset_v + 2][col] = 1  # B10
    V[offset_v + 0][col] = -1 # B00
    W[offset_w + 0][col] = 1  # C00 += M4
    W[offset_w + 2][col] = 1  # C10 += M4
    col += 1
    
    # M5 = (A00+A01)*B11
    U[offset_u + 0][col] = 1  # A00
    U[offset_u + 1][col] = 1  # A01
    V[offset_v + 3][col] = 1  # B11
    W[offset_w + 0][col] = -1 # C00 -= M5
    W[offset_w + 1][col] = 1  # C01 += M5
    col += 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[offset_u + 2][col] = 1  # A10
    U[offset_u + 0][col] = -1 # A00
    V[offset_v + 0][col] = 1  # B00
    V[offset_v + 1][col] = 1  # B01
    W[offset_w + 3][col] = 1  # C11 += M6
    col += 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[offset_u + 1][col] = 1  # A01
    U[offset_u + 3][col] = -1 # A11
    V[offset_v + 2][col] = 1  # B10
    V[offset_v + 3][col] = 1  # B11
    W[offset_w + 0][col] = 1  # C00 += M7
    col += 1
    
    return col

def _construct_hybrid():
    """
    Construct a hybrid Strassen+schoolbook decomposition for 2x4x5.
    
    Strategy:
    - Split A into two 2x2 blocks: A = [A1 | A2] where A1 is cols 0-1, A2 is cols 2-3
    - Split B into two 2x5 blocks: B = [B1; B2] where B1 is rows 0-1, B2 is rows 2-3
    - C = A1*B1 + A2*B2
    
    For each 2x2 * 2x5 product:
    - Split the 5 columns into 2+2+1
    - Use Strassen (rank 7) for the 2x2x2 parts
    - Use schoolbook (rank 4) for the 2x2x1 part
    
    Total per 2x2*2x5: 7 + 7 + 4 = 18
    Total for both: 36
    """
    n, m, p = 2, 4, 5
    total_rank = 36
    U = [[0.0] * total_rank for _ in range(n * m)]  # 8 rows
    V = [[0.0] * total_rank for _ in range(m * p)]  # 20 rows
    W = [[0.0] * total_rank for _ in range(p * n)]  # 10 rows
    
    col_offset = 0
    
    # Block 1: A1 (rows 0-1, cols 0-1 of A) * B1 (rows 0-1 of B, all 5 cols)
    # A1 is at U rows 0-3 (i=0,1; j=0,1)
    # B1 is at V rows 0-9 (j=0,1; k=0..4)
    
    # Sub-block 1a: A1 * B1a (B1a is rows 0-1, cols 0-1)
    # Strassen 2x2x2
    # U offset 0 (i=0,1; j=0,1), V offset 0 (j=0,1; k=0,1), W offset 0 (k=0,1; i=0,1)
    col_offset = _strassen_2x2x2(0, 0, 0, U, V, W, m, p, n, col_offset)
    
    # Sub-block 1b: A1 * B1b (B1b is rows 0-1, cols 2-3)
    # Strassen 2x2x2
    # U offset 0 (i=0,1; j=0,1), V offset 4 (j=0,1; k=2,3), W offset 4 (k=2,3; i=0,1)
    col_offset = _strassen_2x2x2(0, 4, 4, U, V, W, m, p, n, col_offset)
    
    # Sub-block 1c: A1 * B1c (B1c is rows 0-1, col 4)
    # Schoolbook 2x2x1 (one column of B)
    # C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4]
    # C[1,4] = A[1,0]*B[0,4] + A[1,1]*B[1,4]
    # U rows: 0-3 (i=0,1; j=0,1)
    # V rows: 8-9 (j=0,1; k=4)
    # W rows: 8-9 (k=4; i=0,1)
    
    # A[0,0]*B[0,4] -> C[0,4]
    U[0][col_offset] = 1; V[8][col_offset] = 1; W[8][col_offset] = 1
    col_offset += 1
    # A[0,1]*B[1,4] -> C[0,4]
    U[1][col_offset] = 1; V[9][col_offset] = 1; W[8][col_offset] = 1
    col_offset += 1
    # A[1,0]*B[0,4] -> C[1,4]
    U[2][col_offset] = 1; V[8][col_offset] = 1; W[9][col_offset] = 1
    col_offset += 1
    # A[1,1]*B[1,4] -> C[1,4]
    U[3][col_offset] = 1; V[9][col_offset] = 1; W[9][col_offset] = 1
    col_offset += 1
    
    # Block 2: A2 (rows 0-1, cols 2-3 of A) * B2 (rows 2-3 of B, all 5 cols)
    # A2 is at U rows 4-7 (i=0,1; j=2,3)
    # B2 is at V rows 10-19 (j=2,3; k=0..4)
    
    # Sub-block 2a: A2 * B2a (B2a is rows 2-3, cols 0-1)
    # Strassen 2x2x2
    # U offset 4 (i=0,1; j=2,3), V offset 10 (j=2,3; k=0,1), W offset 0 (k=0,1; i=0,1)
    col_offset = _strassen_2x2x2(4, 10, 0, U, V, W, m, p, n, col_offset)
    
    # Sub-block 2b: A2 * B2b (B2b is rows 2-3, cols 2-3)
    # Strassen 2x2x2
    # U offset 4 (i=0,1; j=2,3), V offset 14 (j=2,3; k=2,3), W offset 4 (k=2,3; i=0,1)
    col_offset = _strassen_2x2x2(4, 14, 4, U, V, W, m, p, n, col_offset)
    
    # Sub-block 2c: A2 * B2c (B2c is rows 2-3, col 4)
    # Schoolbook 2x2x1 (one column of B)
    # C[0,4] += A[0,2]*B[2,4] + A[0,3]*B[3,4]
    # C[1,4] += A[1,2]*B[2,4] + A[1,3]*B[3,4]
    # U rows: 4-7 (i=0,1; j=2,3)
    # V rows: 18-19 (j=2,3; k=4)
    # W rows: 8-9 (k=4; i=0,1)
    
    # A[0,2]*B[2,4] -> C[0,4]
    U[4][col_offset] = 1; V[18][col_offset] = 1; W[8][col_offset] = 1
    col_offset += 1
    # A[0,3]*B[3,4] -> C[0,4]
    U[5][col_offset] = 1; V[19][col_offset] = 1; W[8][col_offset] = 1
    col_offset += 1
    # A[1,2]*B[2,4] -> C[1,4]
    U[6][col_offset] = 1; V[18][col_offset] = 1; W[9][col_offset] = 1
    col_offset += 1
    # A[1,3]*B[3,4] -> C[1,4]
    U[7][col_offset] = 1; V[19][col_offset] = 1; W[9][col_offset] = 1
    col_offset += 1
    
    assert col_offset == total_rank, f"Expected {total_rank} columns, got {col_offset}"
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Use the constructive hybrid decomposition
    result = _construct_hybrid()
    
    # Verify
    U = result["U"]
    V = result["V"]
    W = result["W"]
    rank = len(U[0])
    
    if _verify(U, V, W, n, m, p, rank):
        return result
    
    # Fallback to schoolbook if verification fails
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
# EVOLVE-BLOCK-END
