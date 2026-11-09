import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    term_idx = 0
    
    def add_strassen_222(i0, j0, k0):
        nonlocal term_idx
        # 7 terms
        # M1
        U[i0*5 + j0][term_idx] = 1
        U[i0*5 + j0+1][term_idx] = 1
        V[(j0)*6 + k0][term_idx] = 1
        V[(j0+1)*6 + k0+1][term_idx] = 1
        W[(k0)*2 + i0][term_idx] = 1
        W[(k0+1)*2 + i0+1][term_idx] = 1
        term_idx += 1
        # M2
        U[(i0+1)*5 + j0][term_idx] = 1
        U[(i0+1)*5 + j0+1][term_idx] = 1
        V[(j0)*6 + k0][term_idx] = 1
        W[(k0)*2 + i0+1][term_idx] = 1
        term_idx += 1
        # M3
        U[i0*5 + j0][term_idx] = 1
        V[(j0)*6 + k0+1][term_idx] = 1
        V[(j0+1)*6 + k0+1][term_idx] = -1
        W[(k0+1)*2 + i0][term_idx] = 1
        term_idx += 1
        # M4
        U[(i0+1)*5 + j0+1][term_idx] = 1
        V[(j0+1)*6 + k0][term_idx] = 1
        V[(j0)*6 + k0][term_idx] = -1
        W[(k0)*2 + i0+1][term_idx] = 1
        term_idx += 1
        # M5
        U[i0*5 + j0][term_idx] = 1
        U[i0*5 + j0+1][term_idx] = 1
        V[(j0+1)*6 + k0+1][term_idx] = 1
        W[(k0+1)*2 + i0][term_idx] = -1
        W[(k0+1)*2 + i0+1][term_idx] = 1
        term_idx += 1
        # M6
        U[(i0+1)*5 + j0][term_idx] = 1
        U[i0*5 + j0][term_idx] = -1
        V[(j0)*6 + k0][term_idx] = 1
        V[(j0)*6 + k0+1][term_idx] = 1
        W[(k0+1)*2 + i0+1][term_idx] = 1
        term_idx += 1
        # M7
        U[i0*5 + j0+1][term_idx] = 1
        U[(i0+1)*5 + j0+1][term_idx] = -1
        V[(j0+1)*6 + k0][term_idx] = 1
        V[(j0+1)*6 + k0+1][term_idx] = 1
        W[(k0)*2 + i0][term_idx] = 1
        term_idx += 1

    def add_schoolbook_221(i0, j0, k0):
        # 2x2x1, rank 4
        nonlocal term_idx
        # C_i0,k0 = A_i0,j0 B_j0,k0 + A_i0,j0+1 B_j0+1,k0
        # C_i0+1,k0 = A_i0+1,j0 B_j0,k0 + A_i0+1,j0+1 B_j0+1,k0
        
        # Term 1: A_i0,j0 B_j0,k0
        U[i0*5 + j0][term_idx] = 1
        V[(j0)*6 + k0][term_idx] = 1
        W[(k0)*2 + i0][term_idx] = 1
        term_idx += 1
        # Term 2: A_i0,j0+1 B_j0+1,k0
        U[i0*5 + j0+1][term_idx] = 1
        V[(j0+1)*6 + k0][term_idx] = 1
        W[(k0)*2 + i0][term_idx] = 1
        term_idx += 1
        # Term 3: A_i0+1,j0 B_j0,k0
        U[(i0+1)*5 + j0][term_idx] = 1
        V[(j0)*6 + k0][term_idx] = 1
        W[(k0)*2 + i0+1][term_idx] = 1
        term_idx += 1
        # Term 4: A_i0+1,j0+1 B_j0+1,k0
        U[(i0+1)*5 + j0+1][term_idx] = 1
        V[(j0+1)*6 + k0][term_idx] = 1
        W[(k0)*2 + i0+1][term_idx] = 1
        term_idx += 1

    def add_schoolbook_213(i0, j0, k0):
        # 2x1x3, rank 6
        nonlocal term_idx
        for kk in range(3):
            # C_i0,k0+kk = A_i0,j0 B_j0,k0+kk
            U[i0*5 + j0][term_idx] = 1
            V[(j0)*6 + k0+kk][term_idx] = 1
            W[(k0+kk)*2 + i0][term_idx] = 1
            term_idx += 1
            # C_i0+1,k0+kk = A_i0+1,j0 B_j0,k0+kk
            U[(i0+1)*5 + j0][term_idx] = 1
            V[(j0)*6 + k0+kk][term_idx] = 1
            W[(k0+kk)*2 + i0+1][term_idx] = 1
            term_idx += 1

    # Block 1: k0=0
    add_strassen_222(0, 0, 0)
    add_schoolbook_221(0, 0, 2)
    add_schoolbook_213(0, 2, 0)
    add_strassen_222(0, 3, 0)
    add_schoolbook_221(0, 3, 2)

    # Block 2: k0=3
    add_strassen_222(0, 0, 3)
    add_schoolbook_221(0, 0, 5)
    add_schoolbook_213(0, 2, 3)
    add_strassen_222(0, 3, 3)
    add_schoolbook_221(0, 3, 5)

    return {"U": U, "V": V, "W": W}
