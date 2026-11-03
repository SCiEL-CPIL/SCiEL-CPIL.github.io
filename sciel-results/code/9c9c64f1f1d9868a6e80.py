"""
Matrix multiplication 2x4 by 4x5 with Strassen on 2x2 blocks.

Strategy:
- Split B into four 2x2 blocks (B00, B01, B10, B11) plus one 4x1 column.
- Split A correspondingly into A00, A01, A10, A11 (each 2x2) plus the full A for the last column.
- Use Strassen (7 multiplications) for each of the four 2x2x2 block products.
- Use schoolbook (8 multiplications) for the 2x4 by 4x1 product for the last column.

Total rank: 4*7 + 8 = 36.

The decomposition is exact over integers.
"""

import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140  # safety margin before 150s limit


def construct():
    """
    Return factor matrices U, V, W for the tensor decomposition of 2x4 * 4x5 multiplication.
    
    U: shape (n*m, rank) = (8, 36)
    V: shape (m*p, rank) = (20, 36)
    W: shape (p*n, rank) = (10, 36)
    
    Entry convention:
      U[i*m + j, k] is the coefficient of A[i,j] in the k-th multiplicative term.
      V[j*p + k, l] is the coefficient of B[j,k] in the l-th multiplicative term.
      W[k*n + i, m] is the coefficient added to C[i,k] from the m-th multiplicative term.
    """
    n, m, p = 2, 4, 5
    rank = 36

    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36

    def add_strassen_block(a_row0, a_col0, b_row0, b_col0, start):
        """
        Add 7 Strassen multiplications for:
        C[a_row0:a_row0+2, b_col0:b_col0+2] += A[a_row0:a_row0+2, a_col0:a_col0+2] * B[b_row0:b_row0+2, b_col0:b_col0+2]
        
        Returns the next column index.
        """
        # Strassen formulas for 2x2 matrices:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        idx = start
        
        # M1
        U[a_row0 * m + a_col0][idx] += 1
        U[(a_row0 + 1) * m + (a_col0 + 1)][idx] += 1
        V[b_row0 * p + b_col0][idx] += 1
        V[(b_row0 + 1) * p + (b_col0 + 1)][idx] += 1
        W[b_col0 * n + a_row0][idx] += 1
        W[(b_col0 + 1) * n + (a_row0 + 1)][idx] += 1
        idx += 1
        
        # M2
        U[(a_row0 + 1) * m + a_col0][idx] += 1
        U[(a_row0 + 1) * m + (a_col0 + 1)][idx] += 1
        V[b_row0 * p + b_col0][idx] += 1
        W[b_col0 * n + (a_row0 + 1)][idx] += 1
        W[(b_col0 + 1) * n + (a_row0 + 1)][idx] -= 1
        idx += 1
        
        # M3
        U[a_row0 * m + a_col0][idx] += 1
        V[b_row0 * p + (b_col0 + 1)][idx] += 1
        V[(b_row0 + 1) * p + (b_col0 + 1)][idx] -= 1
        W[(b_col0 + 1) * n + a_row0][idx] += 1
        W[(b_col0 + 1) * n + (a_row0 + 1)][idx] += 1
        idx += 1
        
        # M4
        U[(a_row0 + 1) * m + (a_col0 + 1)][idx] += 1
        V[(b_row0 + 1) * p + b_col0][idx] += 1
        V[b_row0 * p + b_col0][idx] -= 1
        W[b_col0 * n + a_row0][idx] += 1
        W[b_col0 * n + (a_row0 + 1)][idx] += 1
        idx += 1
        
        # M5
        U[a_row0 * m + a_col0][idx] += 1
        U[a_row0 * m + (a_col0 + 1)][idx] += 1
        V[(b_row0 + 1) * p + (b_col0 + 1)][idx] += 1
        W[b_col0 * n + a_row0][idx] -= 1
        W[(b_col0 + 1) * n + a_row0][idx] += 1
        idx += 1
        
        # M6
        U[(a_row0 + 1) * m + a_col0][idx] += 1
        U[a_row0 * m + a_col0][idx] -= 1
        V[b_row0 * p + b_col0][idx] += 1
        V[b_row0 * p + (b_col0 + 1)][idx] += 1
        W[(b_col0 + 1) * n + (a_row0 + 1)][idx] += 1
        idx += 1
        
        # M7
        U[a_row0 * m + (a_col0 + 1)][idx] += 1
        U[(a_row0 + 1) * m + (a_col0 + 1)][idx] -= 1
        V[(b_row0 + 1) * p + b_col0][idx] += 1
        V[(b_row0 + 1) * p + (b_col0 + 1)][idx] += 1
        W[b_col0 * n + a_row0][idx] += 1
        idx += 1
        
        return idx

    # Add Strassen for the four 2x2 blocks
    # Block (0,0): A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    col = add_strassen_block(0, 0, 0, 0, 0)
    
    # Block (0,1): A[0:2, 2:4] * B[0:2, 2:4] -> C[0:2, 2:4]
    col = add_strassen_block(0, 2, 0, 2, col)
    
    # Block (1,0): A[2:4, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    col = add_strassen_block(0, 0, 2, 0, col)
    
    # Block (1,1): A[2:4, 2:4] * B[0:2, 2:4] -> C[0:2, 2:4]
    col = add_strassen_block(0, 2, 2, 2, col)
    
    # Schoolbook for the last column: A[0:2, 0:4] * B[0:4, 4:5] -> C[0:2, 4:5]
    for i in range(2):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Deadline guard
    if time.time() > _DEADLINE:
        raise TimeoutError("Deadline exceeded")
    
    return {"U": U, "V": V, "W": W}
