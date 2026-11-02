"""Improved program for matrix multiplication 2x5x6 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # We will construct a decomposition of rank 54.
    # Structure:
    # 1. Split j (0..4) into j1 (0..1) and j2 (2..4).
    # 2. For j1 (2x2x6), use 3 Strassen blocks (each 2x2x2) -> Rank 3 * 7 = 21.
    # 3. For j2 (2x3x6), split j2 into j2a (0) and j2b (1..2).
    #    a. j2a (2x1x6) -> Rank 12.
    #    b. j2b (2x2x6) -> Rank 3 * 7 = 21.
    # Total Rank = 21 + 12 + 21 = 54.
    
    rank = 54
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a Strassen block for A[2x2] * B[2x2] contributing to C[2x2]
    # Indices: i in 0..1, j in 0..1 (local), k in 0..1 (local)
    # Global indices: i_g, j_g, k_g
    def add_strassen_block(i_offset, j_offset, k_offset):
        nonlocal col
        # Strassen coefficients
        # m1 = (a11+a22)(b11+b22)
        # m2 = (a21+a22)b11
        # m3 = a11(b12-b22)
        # m4 = a22(b21-b11)
        # m5 = (a11+a12)b22
        # m6 = (a21-a11)(b11+b12)
        # m7 = (a12-a22)(b21+b22)
        
        # c11 = m1 + m4 - m5 + m7
        # c12 = m3 + m5
        # c21 = m2 + m4
        # c22 = m1 - m2 + m3 + m6
        
        # Map local (i, j, k) to global
        # i_g = i_offset + i
        # j_g = j_offset + j
        # k_g = k_offset + k
        
        # m1
        U[i_offset * m + j_offset, col] = 1
        U[(i_offset+1) * m + (j_offset+1), col] = 1
        V[j_offset * p + k_offset, col] = 1
        V[(j_offset+1) * p + (k_offset+1), col] = 1
        W[k_offset * n + i_offset, col] = 1
        W[(k_offset+1) * n + (i_offset+1), col] = 1
        col += 1
        
        # m2
        U[(i_offset+1) * m + j_offset, col] = 1
        U[(i_offset+1) * m + (j_offset+1), col] = 1
        V[j_offset * p + k_offset, col] = 1
        W[k_offset * n + (i_offset+1), col] = 1
        col += 1
        
        # m3
        U[i_offset * m + j_offset, col] = 1
        V[(j_offset+1) * p + (k_offset+1), col] = 1
        V[j_offset * p + (k_offset+1), col] = -1
        W[(k_offset+1) * n + i_offset, col] = 1
        col += 1
        
        # m4
        U[(i_offset+1) * m + (j_offset+1), col] = 1
        V[(j_offset+1) * p + k_offset, col] = 1
        V[j_offset * p + k_offset, col] = -1
        W[k_offset * n + (i_offset+1), col] = 1
        col += 1
        
        # m5
        U[i_offset * m + j_offset, col] = 1
        U[i_offset * m + (j_offset+1), col] = 1
        V[(j_offset+1) * p + (k_offset+1), col] = 1
        W[(k_offset+1) * n + i_offset, col] = 1
        col += 1
        
        # m6
        U[(i_offset+1) * m + j_offset, col] = 1
        U[i_offset * m + j_offset, col] = -1
        V[j_offset * p + k_offset, col] = 1
        V[j_offset * p + (k_offset+1), col] = 1
        W[(k_offset+1) * n + (i_offset+1), col] = 1
        col += 1
        
        # m7
        U[i_offset * m + (j_offset+1), col] = 1
        U[(i_offset+1) * m + (j_offset+1), col] = -1
        V[(j_offset+1) * p + k_offset, col] = 1
        V[(j_offset+1) * p + (k_offset+1), col] = 1
        W[k_offset * n + i_offset, col] = 1
        col += 1

    # Part 1: j in 0,1. k in 0..5.
    # 3 Strassen blocks for k=0,1; k=2,3; k=4,5.
    for k_block in range(3):
        k_offset = k_block * 2
        add_strassen_block(0, 0, k_offset)
        
    # Part 2: j in 2,3,4.
    # Sub-part 2a: j=2 (local 0). 2x1x6.
    # Terms: A[i, 2] * B[2, k].
    for i in range(2):
        for k in range(6):
            U[i * m + 2, col] = 1
            V[2 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
            
    # Sub-part 2b: j in 3,4 (local 1,2). 2x2x6.
    # 3 Strassen blocks for k=0,1; k=2,3; k=4,5.
    # j_offset = 3.
    for k_block in range(3):
        k_offset = k_block * 2
        add_strassen_block(0, 3, k_offset)

    def to_halfint_list(mat):
        result = []
        for row in mat:
            r = []
            for val in row:
                v = float(val)
                half = round(v * 2) / 2.0
                r.append(half)
            result.append(r)
        return result

    return {"U": to_halfint_list(U), "V": to_halfint_list(V), "W": to_halfint_list(W)}
