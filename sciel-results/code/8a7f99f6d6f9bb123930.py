"""Matrix multiplication 3,3,8 with block decomposition rank 68."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 3, 3, 8
    
    # Block decomposition:
    # A is 3x3, split as:
    #   A11 (rows 0-1, cols 0-1): 2x2
    #   a12 (rows 0-1, col 2):    2x1
    #   a21 (row 2, cols 0-1):    1x2
    #   a22 (row 2, col 2):       1x1
    # B is 3x8, split as:
    #   B11 (rows 0-1, cols 0-1): 2x2
    #   B12 (rows 0-1, cols 2-7): 2x6
    #   B21 (row 2, cols 0-1):    1x2
    #   B22 (row 2, cols 2-7):    1x6
    # C is 3x8, output indexed as C[i][j] where i in 0..2, j in 0..7
    # W index: k*n + i where k is column of C (0..7), i is row of C (0..2)
    
    rank = 68
    U = np.zeros((n * m, rank), dtype=float)  # 9 x 68
    V = np.zeros((m * p, rank), dtype=float)  # 24 x 68
    W = np.zeros((p * n, rank), dtype=float)  # 24 x 68
    
    col = 0
    
    # Term 1: A11 * B11 (2x2 * 2x2 -> 2x2), rank 4
    # A11 elements: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices 0,1,3,4
    # B11 elements: B[0,0], B[0,1], B[1,0], B[1,1] -> V indices 0,1,8,9
    # C output: C[0,0], C[0,1], C[1,0], C[1,1] -> W indices 0*3+0, 1*3+0, 0*3+1, 1*3+1
    # i.e., W[k*n+i] where k=col of C, i=row of C
    # C[0,0]: k=0,i=0 -> W[0]
    # C[0,1]: k=1,i=0 -> W[3]
    # C[1,0]: k=0,i=1 -> W[1]
    # C[1,1]: k=1,i=1 -> W[4]
    
    # A11*B11: C[i,j] = sum_lk A[i,l]*B[l,k] for i in {0,1}, j in {0,1}
    # Trivial: for each (i,l,k) triple
    for i in range(2):
        for l in range(2):
            for k in range(2):
                U[i * m + l, col] = 1  # A[i,l]
                V[l * p + k, col] = 1  # B[l,k]
                W[k * n + i, col] = 1  # C[i,k]
                col += 1
    # col = 4
    
    # Term 2: a12 * B21 (2x1 * 1x2 -> 2x2), rank 4
    # a12: A[0,2], A[1,2] -> U indices 2, 5
    # B21: B[2,0], B[2,1] -> V indices 16, 17
    # C output: C[0,0], C[0,1], C[1,0], C[1,1] -> W[0], W[3], W[1], W[4]
    # C[i,j] = A[i,2] * B[2,j] for i in {0,1}, j in {0,1}
    for i in range(2):
        for k in range(2):
            U[i * m + 2, col] = 1  # A[i,2]
            V[2 * p + k, col] = 1  # B[2,k]
            W[k * n + i, col] = 1  # C[i,k]
            col += 1
    # col = 8
    
    # Term 3: a12 * B22 (2x1 * 1x6 -> 2x6), rank 12
    # a12: A[0,2], A[1,2] -> U indices 2, 5
    # B22: B[2,2], B[2,3], ..., B[2,7] -> V indices 18,19,20,21,22,23
    # C output: C[i,k] for i in {0,1}, k in {2,...,7} -> W[k*n+i]
    for i in range(2):
        for k in range(2, 8):
            U[i * m + 2, col] = 1  # A[i,2]
            V[2 * p + k, col] = 1  # B[2,k]
            W[k * n + i, col] = 1  # C[i,k]
            col += 1
    # col = 20
    
    # Term 4: A11 * B12 (2x2 * 2x6 -> 2x6), rank 24
    # A11: A[0,0],A[0,1],A[1,0],A[1,1] -> U indices 0,1,3,4
    # B12: B[0,2..7], B[1,2..7] -> V indices 2,3,4,5,6,7, 10,11,12,13,14,15
    # C output: C[i,k] for i in {0,1}, k in {2,...,7} -> W[k*n+i]
    # C[i,k] = sum_l A[i,l]*B[l,k] for l in {0,1}
    for i in range(2):
        for l in range(2):
            for k in range(2, 8):
                U[i * m + l, col] = 1  # A[i,l]
                V[l * p + k, col] = 1  # B[l,k]
                W[k * n + i, col] = 1  # C[i,k]
                col += 1
    # col = 44
    
    # Term 5: a21 * B11 (1x2 * 2x2 -> 1x2), rank 4
    # a21: A[2,0], A[2,1] -> U indices 6, 7
    # B11: B[0,0], B[0,1], B[1,0], B[1,1] -> V indices 0,1,8,9
    # C output: C[2,0], C[2,1] -> W[0*3+2], W[1*3+2] = W[2], W[5]
    # C[2,k] = sum_l A[2,l]*B[l,k] for l in {0,1}, k in {0,1}
    for l in range(2):
        for k in range(2):
            U[2 * m + l, col] = 1  # A[2,l]
            V[l * p + k, col] = 1  # B[l,k]
            W[k * n + 2, col] = 1  # C[2,k]
            col += 1
    # col = 48
    
    # Term 6: a21 * B12 (1x2 * 2x6 -> 1x6), rank 12
    # a21: A[2,0], A[2,1] -> U indices 6, 7
    # B12: B[0,2..7], B[1,2..7] -> V indices 2..7, 10..15
    # C output: C[2,k] for k in {2,...,7} -> W[k*3+2]
    for l in range(2):
        for k in range(2, 8):
            U[2 * m + l, col] = 1  # A[2,l]
            V[l * p + k, col] = 1  # B[l,k]
            W[k * n + 2, col] = 1  # C[2,k]
            col += 1
    # col = 60
    
    # Term 7: a22 * B21 (1x1 * 1x2 -> 1x2), rank 2
    # a22: A[2,2] -> U index 8
    # B21: B[2,0], B[2,1] -> V indices 16, 17
    # C output: C[2,0], C[2,1] -> W[2], W[5]
    for k in range(2):
        U[2 * m + 2, col] = 1  # A[2,2]
        V[2 * p + k, col] = 1  # B[2,k]
        W[k * n + 2, col] = 1  # C[2,k]
        col += 1
    # col = 62
    
    # Term 8: a22 * B22 (1x1 * 1x6 -> 1x6), rank 6
    # a22: A[2,2] -> U index 8
    # B22: B[2,2..7] -> V indices 18..23
    # C output: C[2,k] for k in {2,...,7} -> W[k*3+2]
    for k in range(2, 8):
        U[2 * m + 2, col] = 1  # A[2,2]
        V[2 * p + k, col] = 1  # B[2,k]
        W[k * n + 2, col] = 1  # C[2,k]
        col += 1
    # col = 68
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
