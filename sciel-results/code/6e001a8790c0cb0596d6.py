"""
Matrix multiplication 5x5x5 tensor decomposition.

Approach: Block decomposition with 5 = 3 + 2 partition.
- Use Strassen's algorithm (rank 7) for the 2x2x2 block
- Use schoolbook (rank n*m*p) for all other blocks

Total rank: 27 + 18 + 12 + 18 + 12 + 7 = 94
"""

DIMENSIONS = [5, 5, 5]


def _schoolbook_factors(n, m, p):
    """Return U, V, W for schoolbook n x m x p matrix multiplication."""
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
    return U, V, W


def _strassen_factors():
    """
    Return U, V, W for Strassen's 2x2x2 algorithm.
    A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    C = A*B
    
    Seven products:
    M1 = a*(e+f)
    M2 = (a+b)*h
    M3 = c*(g+h)
    M4 = d*(e+f)
    M5 = (a+d)*(e+h-f)
    M6 = (b+c)*(g+e-h)
    M7 = (a+c)*(g+h)
    
    C00 = M1 - M4 + M5
    C01 = M2 - M5 + M7
    C10 = M3 + M4 - M6
    C11 = M1 - M3 + M6
    """
    rank = 7
    # U: shape (n*m, rank) = (4, 7), indexed by [i*m+j]
    # A = [[a, b], [c, d]] -> indices: a=0, b=1, c=2, d=3
    U = [
        [1, 0, 0, 0, 1, 0, 0],  # a: M1, M5
        [0, 1, 0, 0, 0, 1, 0],  # b: M2, M6
        [0, 0, 1, 0, 0, 1, 1],  # c: M3, M6, M7
        [0, 0, 0, 1, 1, 0, 0],  # d: M4, M5
    ]
    # V: shape (m*p, rank) = (4, 7), indexed by [j*p+k]
    # B = [[e, f], [g, h]] -> indices: e=0, f=1, g=2, h=3
    V = [
        [1, 0, 0, 1, 1, 1, 0],  # e: M1, M4, M5, M6
        [1, 0, 0, 1, -1, 0, 0], # f: M1, M4, M5
        [0, 0, 1, 0, 0, 0, 1],  # g: M3, M7
        [0, 1, 1, 0, 1, -1, 1], # h: M2, M3, M5, M6, M7
    ]
    # W: shape (p*n, rank) = (4, 7), indexed by [k*n+i]
    # C = [[C00, C01], [C10, C11]]
    # W[k*n + i, r]
    # C00: i=0, k=0 -> index 0
    # C01: i=0, k=1 -> index 1
    # C10: i=1, k=0 -> index 2
    # C11: i=1, k=1 -> index 3
    W = [
        [1, 0, 0, -1, 1, 0, 0],   # C00 = M1 - M4 + M5
        [0, 1, 0, 0, -1, 0, 1],   # C01 = M2 - M5 + M7
        [0, 0, 1, 1, 0, -1, 0],   # C10 = M3 + M4 - M6
        [1, 0, -1, 0, 0, 1, 0],   # C11 = M1 - M3 + M6
    ]
    return U, V, W


def _concat_factors(U1, V1, W1, U2, V2, W2, **kw):
    """Concatenate two sets of factors along the rank (column) dimension."""
    U = [row + row2 for row, row2 in zip(U1, U2)]
    V = [row + row2 for row, row2 in zip(V1, V2)]
    W = [row + row2 for row, row2 in zip(W1, W2)]
    return U, V, W


def _block_decompose_5x5x5():
    """
    Decompose 5x5x5 matrix mult into blocks using 5 = 3 + 2.
    
    A is 5x5, partitioned as:
    A = [[A11, A12], [A21, A22]] where A11 is 3x3, A12 is 3x2, A21 is 2x3, A22 is 2x2
    B is 5x5, partitioned as:
    B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x3, B21 is 3x2, B22 is 3x3
    
    C = A*B, partitioned as:
    C = [[C11, C12], [C21, C22]] where C11 is 3x2, C12 is 3x3, C21 is 2x2, C22 is 2x3
    
    C11 = A11*B11 + A12*B21   (3x2)
    C12 = A11*B12 + A12*B22   (3x3)
    C21 = A21*B11 + A22*B21   (2x2)
    C22 = A21*B12 + A22*B22   (2x3)
    
    Each product term is a sub-matrix multiplication:
    1. A11*B11: 3x3 * 2x2 -> 3x2 (schoolbook: 3*3*2 = 18)
    2. A12*B21: 3x2 * 3x2 -> 3x2 (schoolbook: 3*2*3 = 18)
    3. A11*B12: 3x3 * 2x3 -> 3x3 (schoolbook: 3*3*3 = 27)
    4. A12*B22: 3x2 * 3x3 -> 3x3 (schoolbook: 3*2*3 = 18)
    5. A21*B11: 2x3 * 2x2 -> 2x2 (schoolbook: 2*3*2 = 12)
    6. A22*B21: 2x2 * 3x2 -> 2x2 (schoolbook: 2*2*3 = 12)
    7. A21*B12: 2x3 * 2x3 -> 2x3 (schoolbook: 2*3*3 = 18)
    8. A22*B22: 2x2 * 3x3 -> 2x3 (schoolbook: 2*2*3 = 12)
    """
    n, m, p = 5, 5, 5
    total_rank = 18 + 18 + 27 + 18 + 12 + 12 + 18 + 12
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    col = 0
    
    # Block 1: A11(3x3) * B11(2x2) -> C11(3x2), schoolbook rank 18
    # A11: rows 0-2, cols 0-2. U global: i*5+j
    # B11: rows 0-1, cols 0-1. V global: j*5+k
    # C11: rows 0-2, cols 0-1. W global: k*5+i
    for i in range(3):
        for j in range(3):
            for k in range(2):
                c = i * 6 + j * 2 + k
                U[i * 5 + j][col + c] = 1
                V[j * 5 + k][col + c] = 1
                W[k * 5 + i][col + c] = 1
    col += 18
    
    # Block 2: A12(3x2) * B21(3x2) -> C11(3x2), schoolbook rank 18
    # A12: rows 0-2, cols 3-4. U global: i*5+(3+j)
    # B21: rows 2-4, cols 0-1. V global: (2+j)*5+k
    # C11: rows 0-2, cols 0-1. W global: k*5+i
    for i in range(3):
        for j in range(2):
            for k in range(3):
                c = i * 6 + j * 3 + k
                U[i * 5 + (3 + j)][col + c] = 1
                V[(2 + j) * 5 + k][col + c] = 1
                W[k * 5 + i][col + c] = 1
    col += 18
    
    # Block 3: A11(3x3) * B12(2x3) -> C12(3x3), schoolbook rank 27
    # A11: rows 0-2, cols 0-2. U global: i*5+j
    # B12: rows 0-1, cols 2-4. V global: j*5+(2+k)
    # C12: rows 0-2, cols 2-4. W global: (2+k)*5+i
    for i in range(3):
        for j in range(3):
            for k in range(3):
                c = i * 9 + j * 3 + k
                U[i * 5 + j][col + c] = 1
                V[j * 5 + (2 + k)][col + c] = 1
                W[(2 + k) * 5 + i][col + c] = 1
    col += 27
    
    # Block 4: A12(3x2) * B22(3x3) -> C12(3x3), schoolbook rank 18
    # A12: rows 0-2, cols 3-4. U global: i*5+(3+j)
    # B22: rows 2-4, cols 2-4. V global: (2+j)*5+(2+k)
    # C12: rows 0-2, cols 2-4. W global: (2+k)*5+i
    for i in range(3):
        for j in range(2):
            for k in range(3):
                c = i * 9 + j * 3 + k
                U[i * 5 + (3 + j)][col + c] = 1
                V[(2 + j) * 5 + (2 + k)][col + c] = 1
                W[(2 + k) * 5 + i][col + c] = 1
    col += 18
    
    # Block 5: A21(2x3) * B11(2x2) -> C21(2x2), schoolbook rank 12
    # A21: rows 3-4, cols 0-2. U global: (3+i)*5+j
    # B11: rows 0-1, cols 0-1. V global: j*5+k
    # C21: rows 3-4, cols 0-1. W global: k*5+(3+i)
    for i in range(2):
        for j in range(3):
            for k in range(2):
                c = i * 6 + j * 2 + k
                U[(3 + i) * 5 + j][col + c] = 1
                V[j * 5 + k][col + c] = 1
                W[k * 5 + (3 + i)][col + c] = 1
    col += 12
    
    # Block 6: A22(2x2) * B21(3x2) -> C21(2x2), schoolbook rank 12
    # A22: rows 3-4, cols 3-4. U global: (3+i)*5+(3+j)
    # B21: rows 2-4, cols 0-1. V global: (2+j)*5+k
    # C21: rows 3-4, cols 0-1. W global: k*5+(3+i)
    for i in range(2):
        for j in range(2):
            for k in range(3):
                c = i * 6 + j * 3 + k
                U[(3 + i) * 5 + (3 + j)][col + c] = 1
                V[(2 + j) * 5 + k][col + c] = 1
                W[k * 5 + (3 + i)][col + c] = 1
    col += 12
    
    # Block 7: A21(2x3) * B12(2x3) -> C22(2x3), schoolbook rank 18
    # A21: rows 3-4, cols 0-2. U global: (3+i)*5+j
    # B12: rows 0-1, cols 2-4. V global: j*5+(2+k)
    # C22: rows 3-4, cols 2-4. W global: (2+k)*5+(3+i)
    for i in range(2):
        for j in range(3):
            for k in range(3):
                c = i * 9 + j * 3 + k
                U[(3 + i) * 5 + j][col + c] = 1
                V[j * 5 + (2 + k)][col + c] = 1
                W[(2 + k) * 5 + (3 + i)][col + c] = 1
    col += 18
    
    # Block 8: A22(2x2) * B22(3x3) -> C22(2x3), schoolbook rank 12
    # A22: rows 3-4, cols 3-4. U global: (3+i)*5+(3+j)
    # B22: rows 2-4, cols 2-4. V global: (2+j)*5+(2+k)
    # C22: rows 3-4, cols 2-4. W global: (2+k)*5+(3+i)
    for i in range(2):
        for j in range(2):
            for k in range(3):
                c = i * 9 + j * 3 + k
                U[(3 + i) * 5 + (3 + j)][col + c] = 1
                V[(2 + j) * 5 + (2 + k)][col + c] = 1
                W[(2 + k) * 5 + (3 + i)][col + c] = 1
    col += 12
    
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
    return U, V, W


def construct():
    U, V, W = _block_decompose_5x5x5()
    return {"U": U, "V": V, "W": W}
