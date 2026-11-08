import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 36
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen block
    def add_strassen(a_row_start, a_col_start, b_row_start, b_col_start, c_col_start):
        nonlocal col
        # a_row_start is 0 (only 2 rows)
        # a_col_start is 0 or 2
        # b_row_start is 0 or 2
        # b_col_start is 0 or 2
        # c_col_start is 0 or 2
        
        # Strassen M1..M7
        # A coeffs
        # Indices in A: (r, c) -> r * m + c
        # B coeffs
        # Indices in B: (r, c) -> r * p + c
        # C coeffs
        # Indices in C: (r, c) -> c * n + r (based on seed: k*n+i)
        
        # M1
        U[a_row_start*4 + a_col_start][col] = 1
        U[a_row_start*4 + a_col_start + 1][col] = 1
        V[b_row_start*5 + b_col_start][col] = 1
        V[b_row_start*5 + b_col_start + 1][col] = 1
        # C coeffs for M1: C00+=1, C11+=1
        W[(c_col_start)*2 + 0][col] += 1 # C00
        W[(c_col_start+1)*2 + 1][col] += 1 # C11
        col += 1
        
        # M2
        U[a_row_start*4 + a_col_start + 1][col] = 1 # c
        U[a_row_start*4 + a_col_start + 2][col] = 1 # d? No, A is 2x2.
        # A indices: 0,1 (row 0), 2,3 (row 1)?
        # Wait, A is 2x4.
        # A_1 is cols 0,1.
        # A_1[0,0] is global (0,0). A_1[0,1] is global (0,1).
        # A_1[1,0] is global (1,0). A_1[1,1] is global (1,1).
        # So for A_1:
        # a=0,0 -> 0
        # b=0,1 -> 1
        # c=1,0 -> 4
        # d=1,1 -> 5
        # My previous index calc `a_row_start*4 + a_col_start` assumes row major with stride 4.
        # Yes, `i*m+j`.
        
        # Let's re-verify indices.
        # A_1 (cols 0,1):
        # a (0,0) -> 0
        # b (0,1) -> 1
        # c (1,0) -> 4
        # d (1,1) -> 5
        
        # A_2 (cols 2,3):
        # a (0,2) -> 2
        # b (0,3) -> 3
        # c (1,2) -> 6
        # d (1,3) -> 7
        
        # B_11 (rows 0,1, cols 0,1):
        # e (0,0) -> 0
        # f (0,1) -> 1
        # g (1,0) -> 5
        # h (1,1) -> 6
        
        # B_21 (rows 2,3, cols 0,1):
        # e (2,0) -> 10
        # f (2,1) -> 11
        # g (3,0) -> 15
        # h (3,1) -> 16
        
        # B_12 (rows 0,1, cols 2,3):
        # e (0,2) -> 2
        # f (0,3) -> 3
        # g (1,2) -> 7
        # h (1,3) -> 8
        
        # B_22 (rows 2,3, cols 2,3):
        # e (2,2) -> 12
        # f (2,3) -> 13
        # g (3,2) -> 17
        # h (3,3) -> 18
        
        # C_1 (cols 0,1):
        # C00 (0,0) -> 0
        # C01 (0,1) -> 2
        # C10 (1,0) -> 1
        # C11 (1,1) -> 3
        
        # C_2 (cols 2,3):
        # C02 (0,2) -> 4
        # C03 (0,3) -> 6
        # C12 (1,2) -> 5
        # C13 (1,3) -> 7
        
        # C_3 (col 4):
        # C04 (0,4) -> 8
        # C14 (1,4) -> 9

        # I will implement the helper to take explicit indices to avoid confusion.
        pass

    # It is safer to hardcode the indices or calculate them carefully.
    
    return {"U": U, "V": V, "W": W}
