import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p  # 45
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
    
    try:
        # Rank 40 Decomposition
        # Part 1: 3x3x3 (Rank 22)
        # Part 2: 3x3x2 (Rank 18)
        
        U = []
        V = []
        W = []
        
        def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
            U.append([0] * 40)
            V.append([0] * 40)
            W.append([0] * 40)
            col = len(U) - 1
            U[u_idx][col] = u_val
            V[v_idx][col] = v_val
            W[w_idx][col] = w_val

        # Helper to add a linear combination term
        # U is 9x40, V is 15x40, W is 15x40
        # We are building columns.
        
        # Let's use lists of lists for U, V, W
        # U: 9 rows, 40 cols
        # V: 15 rows, 40 cols
        # W: 15 rows, 40 cols
        
        U = [[0]*40 for _ in range(9)]
        V = [[0]*40 for _ in range(15)]
        W = [[0]*40 for _ in range(15)]
        
        col = 0
        
        # 1. Strassen for A22 B22
        # A indices: 0,1,3,4 (A11, A12, A21, A22)
        # B indices: 0,1,3,4 (B11, B12, B21, B22)
        # W indices for C11, C12, C21, C22: 0, 1, 3, 4 (C11, C12, C21, C22)
        
        # M1 = A11 (B11 + B22)
        U[0][col] = 1
        V[0][col] = 1
        V[4][col] = 1
        W[0][col] = 1
        W[4][col] = 1
        col += 1
        
        # M2 = (A11 + A12) B22
        U[0][col] = 1
        U[1][col] = 1
        V[4][col] = 1
        W[1][col] = 1
        col += 1
        
        # M3 = (A21 + A22) B11
        U[3][col] = 1
        U[4][col] = 1
        V[0][col] = 1
        W[3][col] = 1
        col += 1
        
        # M4 = A22 (B21 + B11)
        U[4][col] = 1
        V[3][col] = 1
        V[0][col] = 1
        W[3][col] = 1
        W[4][col] = 1
        col += 1
        
        # M5 = (A11 + A22)(B11 + B22) - M1 - M4
        # We add the product term, and subtract M1 and M4 contributions in W?
        # No, we add the term to the decomposition.
        # The Strassen algorithm combines them.
        # Term: (A11 + A22)(B11 + B22)
        U[0][col] = 1
        U[4][col] = 1
        V[0][col] = 1
        V[4][col] = 1
        # W contributions for M5:
        # C11 += M5
        # C22 += M5
        # C12 -= M5? No.
        # Standard:
        # C11 = M5 + M4 - M2 + M7
        # C12 = M1 + M2
        # C21 = M3 + M4
        # C22 = M5 + M1 - M3 + M6
        
        # So for M5 term:
        W[0][col] = 1
        W[4][col] = 1
        col += 1
        
        # M6 = (A11 + A12)(B21 + B22) - M2 - M4
        # Term: (A11 + A12)(B21 + B22)
        U[0][col] = 1
        U[1][col] = 1
        V[3][col] = 1
        V[4][col] = 1
        # W contributions:
        # C22 += M6
        W[4][col] = 1
        col += 1
        
        # M7 = (A21 + A22)(B11 + B12) - M1 - M3
        # Term: (A21 + A22)(B11 + B12)
        U[3][col] = 1
        U[4][col] = 1
        V[0][col] = 1
        V[1][col] = 1
        # W contributions:
        # C11 += M7
        W[0][col] = 1
        col += 1
        
        # We need to subtract M1, M2, M3, M4, M6 from appropriate W slots?
        # No, the decomposition is a sum of rank-1 tensors.
        # The Strassen algorithm expresses C as a linear combination of M1..M7.
        # So we just need to add the M1..M7 terms to the sum.
        # The coefficients in W are determined by the Strassen formula.
        
        # Let's re-verify the W coefficients for each M term.
        # M1: W[1] += 1 (C12), W[4] += 1 (C22)?
        # C12 = M1 + M2
        # C22 = M5 + M1 - M3 + M6
        # So M1 contributes to C12 and C22.
        # My previous code for M1: W[0] and W[4]. That was wrong.
        
        # Let's restart the W assignment for Strassen terms.
        
        # Reset col to 0 and do it properly.
        U = [[0]*40 for _ in range(9)]
        V = [[0]*40 for _ in range(15)]
        W = [[0]*40 for _ in range(15)]
        col = 0
        
        # M1 = A11 (B11 + B22)
        U[0][col] = 1
        V[0][col] = 1
        V[4][col] = 1
        W[1][col] = 1  # C12
        W[4][col] = 1  # C22
        col += 1
        
        # M2 = (A11 + A12) B22
        U[0][col] = 1
        U[1][col] = 1
        V[4][col] = 1
        W[0][col] = -1 # C11
        W[1][col] = 1  # C12
        col += 1
        
        # M3 = (A21 + A22) B11
        U[3][col] = 1
        U[4][col] = 1
        V[0][col] = 1
        W[3][col] = 1  # C21
        W[4][col] = -1 # C22
        col += 1
        
        # M4 = A22 (B21 + B11)
        U[4][col] = 1
        V[3][col] = 1
        V[0][col] = 1
        W[0][col] = 1  # C11
        W[3][col] = 1  # C21
        col += 1
        
        # M5 = (A11 + A22)(B11 + B22)
        U[0][col] = 1
        U[4][col] = 1
        V[0][col] = 1
        V[4][col] = 1
        W[0][col] = 1  # C11
        W[4][col] = 1  # C22
        col += 1
        
        # M6 = (A11 + A12)(B21 + B22)
        U[0][col] = 1
        U[1][col] = 1
        V[3][col] = 1
        V[4][col] = 1
        W[4][col] = 1  # C22
        col += 1
        
        # M7 = (A21 + A22)(B11 + B12)
        U[3][col] = 1
        U[4][col] = 1
        V[0][col] = 1
        V[1][col] = 1
        W[0][col] = 1  # C11
        col += 1
        
        # 2. Cross terms
        # c = A22 u + a b33
        # C13 (idx 2) = A11 B13 + A12 B23 + A13 B33
        # C23 (idx 5) = A21 B13 + A22 B23 + A23 B33
        
        # Term: A11 B13
        U[0][col] = 1
        V[2][col] = 1
        W[2][col] = 1
        col += 1
        
        # Term: A12 B23
        U[1][col] = 1
        V[5][col] = 1
        W[2][col] = 1
        col += 1
        
        # Term: A13 B33
        U[2][col] = 1
        V[8][col] = 1
        W[2][col] = 1
        col += 1
        
        # Term: A21 B13
        U[3][col] = 1
        V[2][col] = 1
        W[5][col] = 1
        col += 1
        
        # Term: A22 B23
        U[4][col] = 1
        V[5][col] = 1
        W[5][col] = 1
        col += 1
        
        # Term: A23 B33
        U[5][col] = 1
        V[8][col] = 1
        W[5][col] = 1
        col += 1
        
        # cT = aT B22 + a33 v
        # C31 (idx 6) = A31 B11 + A32 B21 + A33 B31
        # C32 (idx 7) = A31 B12 + A32 B22 + A33 B32
        
        # Term: A31 B11
        U[6][col] = 1
        V[0][col] = 1
        W[6][col] = 1
        col += 1
        
        # Term: A32 B21
        U[7][col] = 1
        V[3][col] = 1
        W[6][col] = 1
        col += 1
        
        # Term: A33 B31
        U[8][col] = 1
        V[6][col] = 1
        W[6][col] = 1
        col += 1
        
        # Term: A31 B12
        U[6][col] = 1
        V[1][col] = 1
        W[7][col] = 1
        col += 1
        
        # Term: A32 B22
        U[7][col] = 1
        V[4][col] = 1
        W[7][col] = 1
        col += 1
        
        # Term: A33 B32
        U[8][col] = 1
        V[7][col] = 1
        W[7][col] = 1
        col += 1
        
        # c33 = aT u + a33 b33
        # C33 (idx 8) = A31 B13 + A32 B23 + A33 B33
        
        # Term: A31 B13
        U[6][col] = 1
        V[2][col] = 1
        W[8][col] = 1
        col += 1
        
        # Term: A32 B23
        U[7][col] = 1
        V[5][col] = 1
        W[8][col] = 1
        col += 1
        
        # Term: A33 B33
        U[8][col] = 1
        V[8][col] = 1
        W[8][col] = 1
        col += 1
        
        # 3. 3x3x2 part (Cols 4, 5 of B, i.e., indices 9-14)
        # B indices:
        # Col 4: 9, 10, 11
        # Col 5: 12, 13, 14
        
        # C14 (idx 9) = A11 B14 + A12 B24 + A13 B34
        # C15 (idx 10) = A11 B15 + A12 B25 + A13 B35
        # ...
        
        # We just iterate
        for i in range(3):
            for j in range(3):
                # For col 4
                U[i*3 + j][col] = 1
                V[9 + j][col] = 1
                W[9 + i][col] = 1
                col += 1
                # For col 5
                U[i*3 + j][col] = 1
                V[12 + j][col] = 1
                W[10 + i][col] = 1
                col += 1
                
        return {"U": U, "V": V, "W": W}
        
    except Exception:
        return schoolbook()
