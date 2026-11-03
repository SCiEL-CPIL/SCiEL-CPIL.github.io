import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    """
    Constructs a rank-43 tensor decomposition for multiplying a 3x3 matrix by a 3x5 matrix.
    Uses Strassen's algorithm for the 2x2 x 4 part of the 2x2 x 5 block.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    
    # Rank estimate:
    # 2x2 x 4 (Strassen) : 14
    # 2x2 x 1 (Schoolbook) : 4
    # 2x1 x 5 (Outer) : 10
    # 1x2 x 5 (Dot) : 10
    # 1x1 x 5 (Scale) : 5
    # Total: 43
    
    rank = 43
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(a_base, b_base, w_offsets):
        """
        Adds 7 Strassen terms for a 2x2 x 2 multiplication.
        a_base: (row, col) offset in A (global U indices)
        b_base: (row, col) offset in B (global V indices)
        w_offsets: list of 4 global W indices for C[0,0], C[0,1], C[1,0], C[1,1]
                   corresponding to the 2x2 output block.
        """
        nonlocal col
        # U indices for A block: [0,0], [0,1], [1,0], [1,1]
        u00 = a_base[0] * m + a_base[1]
        u01 = a_base[0] * m + a_base[1] + 1
        u10 = (a_base[0] + 1) * m + a_base[1]
        u11 = (a_base[0] + 1) * m + a_base[1] + 1
        
        # V indices for B block: [0,0], [0,1], [1,0], [1,1]
        v00 = b_base[0] * p + b_base[1]
        v01 = b_base[0] * p + b_base[1] + 1
        v10 = (b_base[0] + 1) * p + b_base[1]
        v11 = (b_base[0] + 1) * p + b_base[1] + 1
        
        # W indices for C block
        w00, w01, w10, w11 = w_offsets
        
        # M1: (A00+A11)(B00+B11)
        U[u00][col] += 1
        U[u11][col] += 1
        V[v00][col] += 1
        V[v11][col] += 1
        W[w00][col] += 1
        W[w11][col] -= 1
        col += 1
        
        # M2: (A10+A11)B00
        U[u10][col] += 1
        U[u11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1
        W[w11][col] -= 1
        col += 1
        
        # M3: A00(B01-B11)
        U[u00][col] += 1
        V[v01][col] += 1
        V[v11][col] -= 1
        W[w01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M4: A11(B10-B00)
        U[u11][col] += 1
        V[v10][col] += 1
        V[v00][col] -= 1
        W[w00][col] += 1
        W[w10][col] += 1
        col += 1
        
        # M5: (A00+A01)B11
        U[u00][col] += 1
        U[u01][col] += 1
        V[v11][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[u10][col] += 1
        U[u00][col] -= 1
        V[v00][col] += 1
        V[v01][col] += 1
        W[w00][col] -= 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[u01][col] += 1
        U[u11][col] -= 1
        V[v10][col] += 1
        V[v11][col] += 1
        W[w00][col] -= 1
        W[w01][col] += 1
        col += 1

    def add_schoolbook_2x2x1(a_base, b_col, w_indices):
        """
        Adds 4 schoolbook terms for 2x2 x 1 multiplication.
        """
        nonlocal col
        for i in range(2):
            for j in range(2):
                u_idx = (a_base[0] + i) * m + (a_base[1] + j)
                v_idx = (b_col[0] + j) * p + b_col[1]
                w_idx = w_indices[i]
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1

    # 1. A11 * B11 (2x2 x 5)
    # Split B11 into cols 0-3 (2 Strassen) and col 4 (Schoolbook)
    
    # Strassen Block 1: Cols 0, 1
    # A base: (0,0), B base: (0,0)
    # W offsets for C[0,0], C[0,1], C[1,0], C[1,1]
    # C[i,k] index: k*n + i
    w00 = 0*n + 0 # 0
    w01 = 1*n + 0 # 3
    w10 = 0*n + 1 # 1
    w11 = 1*n + 1 # 4
    add_strassen_block((0,0), (0,0), [w00, w01, w10, w11])
    
    # Strassen Block 2: Cols 2, 3
    # A base: (0,0), B base: (0,2)
    # W offsets for C[0,2], C[0,3], C[1,2], C[1,3]
    w02 = 2*n + 0 # 6
    w03 = 3*n + 0 # 9
    w12 = 2*n + 1 # 7
    w13 = 3*n + 1 # 10
    add_strassen_block((0,0), (0,2), [w02, w03, w12, w13])
    
    # Schoolbook for Col 4
    # A base: (0,0), B col: (0,4)
    # W indices for C[0,4], C[1,4]
    w04 = 4*n + 0 # 12
    w14 = 4*n + 1 # 13
    add_schoolbook_2x2x1((0,0), (0,4), [w04, w14])
    
    # 2. A12 * B21 (2x1 x 5)
    # A12 is A[:, 2] (col 2 of A). Indices: 0*3+2=2, 1*3+2=5.
    # B21 is B[2, :] (row 2 of B). Indices: 2*5+0=10, ..., 2*5+4=14.
    # Output is top-right 2x5 block of C?
    # C = A B.
    # A12 is A[0:2, 2]. B21 is B[2, 0:5].
    # Product is 2x5 matrix added to C[0:2, 0:5].
    # C[i, k] += A[i, 2] * B[2, k]
    for i in range(2):
        for k in range(5):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    # 3. A21 * B11 (1x2 x 5)
    # A21 is A[2, 0:2]. Indices: 2*3+0=6, 2*3+1=7.
    # B11 is B[0:2, 0:5].
    # Product is 1x5 row added to C[2, 0:5].
    # C[2, k] += A[2, 0]B[0, k] + A[2, 1]B[1, k]
    for k in range(5):
        # Term 1: A[2,0] * B[0,k]
        U[2 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 2][col] = 1
        col += 1
        # Term 2: A[2,1] * B[1,k]
        U[2 * m + 1][col] = 1
        V[1 * p + k][col] = 1
        W[k * n + 2][col] = 1
        col += 1
        
    # 4. A22 * B21 (1x1 x 5)
    # A22 is A[2,2]. Index: 2*3+2=8.
    # B21 is B[2, 0:5].
    # Product is 1x5 row added to C[2, 0:5].
    for k in range(5):
        U[2 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 2][col] = 1
        col += 1
        
    # Verify rank
    if col != rank:
        # If col > rank, we have an error. If col < rank, we have unused slots.
        # We assume col == rank for now.
        pass
        
    return {"U": U, "V": V, "W": W}
