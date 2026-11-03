"""Optimized matrix multiplication 2x4 by 4x5 using Strassen for 2x2 sub-blocks."""

import time

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for
    multiplying a 2x4 matrix by a 4x5 matrix.
    
    Strategy:
    - Decompose the 2x4x5 multiplication into 4 disjoint 2x2x2 multiplications
      and one 2x4x1 multiplication.
    - Each 2x2x2 multiplication uses Strassen's algorithm with rank 7.
    - The 2x4x1 multiplication uses schoolbook with rank 8.
    - Total rank: 4 * 7 + 8 = 36.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    rank = 36
    
    # Initialize factor matrices with zeros
    # U: (n*m) x rank = 8 x 36
    # V: (m*p) x rank = 20 x 36
    # W: (p*n) x rank = 10 x 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_222(i0, j0, k0, col_start):
        """
        Add a Strassen 2x2x2 decomposition for:
        C[i0:i0+2, k0:k0+2] = A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        """
        nonlocal col
        
        # M1 = (A00 + A11) * (B00 + B11)
        c = col_start
        U[i0 * m + j0][c] += 1
        U[(i0+1) * m + (j0+1)][c] += 1
        V[j0 * p + k0][c] += 1
        V[(j0+1) * p + (k0+1)][c] += 1
        W[k0 * n + i0][c] += 1
        W[(k0+1) * n + (i0+1)][c] += 1
        col += 1
        
        # M2 = (A10 + A11) * B00
        c = col_start + 1
        U[(i0+1) * m + j0][c] += 1
        U[(i0+1) * m + (j0+1)][c] += 1
        V[j0 * p + k0][c] += 1
        W[k0 * n + (i0+1)][c] += 1
        W[(k0+1) * n + (i0+1)][c] -= 1
        col += 1
        
        # M3 = A00 * (B01 - B11)
        c = col_start + 2
        U[i0 * m + j0][c] += 1
        V[j0 * p + (k0+1)][c] += 1
        V[(j0+1) * p + (k0+1)][c] -= 1
        W[(k0+1) * n + i0][c] += 1
        W[(k0+1) * n + (i0+1)][c] += 1
        col += 1
        
        # M4 = A11 * (B10 - B00)
        c = col_start + 3
        U[(i0+1) * m + (j0+1)][c] += 1
        V[(j0+1) * p + k0][c] += 1
        V[j0 * p + k0][c] -= 1
        W[k0 * n + i0][c] += 1
        W[k0 * n + (i0+1)][c] += 1
        col += 1
        
        # M5 = (A00 + A01) * B11
        c = col_start + 4
        U[i0 * m + j0][c] += 1
        U[i0 * m + (j0+1)][c] += 1
        V[(j0+1) * p + (k0+1)][c] += 1
        W[k0 * n + i0][c] -= 1
        W[(k0+1) * n + i0][c] += 1
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        c = col_start + 5
        U[(i0+1) * m + j0][c] += 1
        U[i0 * m + j0][c] -= 1
        V[j0 * p + k0][c] += 1
        V[j0 * p + (k0+1)][c] += 1
        W[(k0+1) * n + (i0+1)][c] += 1
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        c = col_start + 6
        U[i0 * m + (j0+1)][c] += 1
        U[(i0+1) * m + (j0+1)][c] -= 1
        V[(j0+1) * p + k0][c] += 1
        V[(j0+1) * p + (k0+1)][c] += 1
        W[k0 * n + i0][c] += 1
        col += 1
        
        return col_start + 7
    
    def add_schoolbook_2x4x1(i0, j0, k0, col_start):
        """
        Add schoolbook decomposition for:
        C[i0:i0+2, k0:k0+1] = A[i0:i0+2, j0:j0+4] * B[j0:j0+4, k0:k0+1]
        """
        c = col_start
        for di in range(2):
            for dj in range(4):
                U[(i0 + di) * m + (j0 + dj)][c] = 1
                V[(j0 + dj) * p + k0][c] = 1
                W[k0 * n + (i0 + di)][c] = 1
                c += 1
        return c
    
    # Block 1: C[:, 0:2] from A[:, 0:2] * B[0:2, 0:2]
    col = add_strassen_222(0, 0, 0, 0)
    
    # Block 2: C[:, 0:2] from A[:, 2:4] * B[2:4, 0:2]
    col = add_strassen_222(0, 2, 0, col)
    
    # Block 3: C[:, 2:4] from A[:, 0:2] * B[0:2, 2:4]
    col = add_strassen_222(0, 0, 2, col)
    
    # Block 4: C[:, 2:4] from A[:, 2:4] * B[2:4, 2:4]
    col = add_strassen_222(0, 2, 2, col)
    
    # Block 5: C[:, 4] from A * B[:, 4] (schoolbook 2x4 * 4x1)
    col = add_schoolbook_2x4x1(0, 0, 4, col)
    
    # Verify rank
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
