import numpy as np
import time

def construct():
    n, m, p = 2, 4, 5
    
    # We use the same block structure as the current program:
    # 4 Strassen 2x2x2 blocks (rank 7 each) + 2 schoolbook 2x2x1 blocks (rank 4 each)
    # Total rank = 4*7 + 2*4 = 28 + 8 = 36
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen(col_offset, j_offset, k_offset):
        # j_offset: 0 for J1, 2 for J2
        # k_offset: 0 for K1, 2 for K2
        # Strassen for 2x2x2
        # U indices: i*m + (j_offset + j')
        # V indices: (j_offset + j')*p + (k_offset + k')
        # W indices: (k_offset + k')*n + i
        
        # Term 1: M1 = (A00 + A11) * (B00 + B11)
        U[0 * m + j_offset + 0][col_offset + 0] += 1
        U[1 * m + j_offset + 1][col_offset + 0] += 1
        V[(j_offset + 0) * p + k_offset + 0][col_offset + 0] += 1
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 0] += 1
        W[(k_offset + 0) * n + 0][col_offset + 0] += 1
        W[(k_offset + 1) * n + 1][col_offset + 0] += 1

        # Term 2: M2 = (A10 + A11) * B00
        U[1 * m + j_offset + 0][col_offset + 1] += 1
        U[1 * m + j_offset + 1][col_offset + 1] += 1
        V[(j_offset + 0) * p + k_offset + 0][col_offset + 1] += 1
        W[(k_offset + 0) * n + 1][col_offset + 1] += 1
        W[(k_offset + 1) * n + 1][col_offset + 1] += -1

        # Term 3: M3 = A00 * (B01 - B11)
        U[0 * m + j_offset + 0][col_offset + 2] += 1
        V[(j_offset + 0) * p + k_offset + 1][col_offset + 2] += 1
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 2] += -1
        W[(k_offset + 1) * n + 0][col_offset + 2] += 1
        W[(k_offset + 1) * n + 1][col_offset + 2] += 1

        # Term 4: M4 = A11 * (B10 - B00)
        U[1 * m + j_offset + 1][col_offset + 3] += 1
        V[(j_offset + 0) * p + k_offset + 0][col_offset + 3] += -1
        V[(j_offset + 1) * p + k_offset + 0][col_offset + 3] += 1
        W[(k_offset + 0) * n + 0][col_offset + 3] += 1
        W[(k_offset + 0) * n + 1][col_offset + 3] += 1

        # Term 5: M5 = (A00 + A01) * B11
        U[0 * m + j_offset + 0][col_offset + 4] += 1
        U[0 * m + j_offset + 1][col_offset + 4] += 1
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 4] += 1
        W[(k_offset + 0) * n + 0][col_offset + 4] += -1
        W[(k_offset + 1) * n + 0][col_offset + 4] += 1

        # Term 6: M6 = (A10 - A00) * (B00 + B01)
        U[0 * m + j_offset + 0][col_offset + 5] += -1
        U[1 * m + j_offset + 0][col_offset + 5] += 1
        V[(j_offset + 0) * p + k_offset + 0][col_offset + 5] += 1
        V[(j_offset + 0) * p + k_offset + 1][col_offset + 5] += 1
        W[(k_offset + 1) * n + 1][col_offset + 5] += 1

        # Term 7: M7 = (A01 - A11) * (B10 + B11)
        U[0 * m + j_offset + 1][col_offset + 6] += 1
        U[1 * m + j_offset + 1][col_offset + 6] += -1
        V[(j_offset + 1) * p + k_offset + 0][col_offset + 6] += 1
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 6] += 1
        W[(k_offset + 0) * n + 0][col_offset + 6] += 1

    def add_schoolbook_2x2x1(col_offset, j_offset, k_offset):
        # Schoolbook for 2x2x1
        # k is fixed to k_offset
        for di in range(2):
            for dj in range(2):
                U[di * m + j_offset + dj][col_offset + di*2 + dj] = 1
                V[(j_offset + dj) * p + k_offset][col_offset + di*2 + dj] = 1
                W[k_offset * n + di][col_offset + di*2 + dj] = 1

    col = 0
    # J1 (j_offset=0)
    add_strassen(col, 0, 0); col += 7
    add_strassen(col, 0, 2); col += 7
    add_schoolbook_2x2x1(col, 0, 4); col += 4
    
    # J2 (j_offset=2)
    add_strassen(col, 2, 0); col += 7
    add_strassen(col, 2, 2); col += 7
    add_schoolbook_2x2x1(col, 2, 4); col += 4
    
    return {"U": U, "V": V, "W": W}
