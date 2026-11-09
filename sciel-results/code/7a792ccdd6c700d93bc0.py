import time

def construct():
    n, m, p = 2, 3, 4
    deadline = time.time() + 140
    
    # Try Rank 22
    try:
        return _construct_rank_22()
    except Exception:
        return _construct_rank_24()

def _construct_rank_24():
    n, m, p = 2, 3, 4
    rank = n * m * p
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

def _construct_rank_22():
    n, m, p = 2, 3, 4
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Strassen
    # M1: (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[3][col] = 1 # C00, C11
    col += 1
    
    # M2: (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[3][col] = -1 # C01, C11
    col += 1
    
    # M3: A00(B10+B11)
    U[0][col] = 1
    V[4][col] = 1; V[5][col] = 1
    W[2][col] = 1; W[3][col] = 1 # C10, C11
    col += 1
    
    # M4: A11(B01+B10)
    U[4][col] = 1
    V[1][col] = 1; V[4][col] = 1
    W[0][col] = -1; W[1][col] = 1 # C00, C01
    col += 1
    
    # M5: (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = 1; W[2][col] = 1 # C00, C10
    col += 1
    
    # M6: (A10-A00)(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[5][col] = 1 # C? Wait, C11 is 3.
    # M6 is not in standard 7?
    # I used M6 in the list but the reconstruction didn't use it?
    # The reconstruction I used was:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M2 + M4
    # C10 = M3 + M5
    # C11 = M1 - M2 + M3 + M7
    # M6 is unused in this set.
    # So I only need 7 columns.
    
    # M7: (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = -1; W[3][col] = 1 # C00, C11
    col += 1
    
    # M6 was skipped, so col is 7.
    # Wait, I need to finish M6? No, if it's not used, I don't need it.
    # But I listed 7 terms.
    # Term 1: M1
    # Term 2: M2
    # Term 3: M3
    # Term 4: M4
    # Term 5: M5
    # Term 6: M6 (Skipped in reconstruction? No, I skipped it in W assignment)
    # Term 7: M7
    
    # So Block 1 uses 7 columns.
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # Same Strassen logic, but B indices are 2,3,6,7
    # B00->2, B01->3, B10->6, B11->7
    # C00->4, C01->5, C10->6, C11->7
    
    # M1
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[4][col] = 1; W[7][col] = 1
    col += 1
    
    # M2
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[5][col] = 1; W[7][col] = -1
    col += 1
    
    # M3
    U[0][col] = 1
    V[6][col] = 1; V[7][col] = 1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    # M4
    U[4][col] = 1
    V[3][col] = 1; V[6][col] = 1
    W[4][col] = -1; W[5][col] = 1
    col += 1
    
    # M5
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[4][col] = 1; W[6][col] = 1
    col += 1
    
    # M6 (Skipped)
    
    # M7
    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[4][col] = -1; W[7][col] = 1
    col += 1
    
    # Block 2 used 7 columns. Total 14.
    
    # Block 3: A[0:2, 2] * B[2, 0:4] -> C[0:2, 0:4]
    # Schoolbook
    # A02 (index 2), A12 (index 5)
    # B20 (8), B21 (9), B22 (10), B23 (11)
    
    # A02 B20 -> C00 (0)
    U[2][col] = 1; V[8][col] = 1; W[0][col] = 1
    col += 1
    # A02 B21 -> C01 (1)
    U[2][col] = 1; V[9][col] = 1; W[1][col] = 1
    col += 1
    # A02 B22 -> C02 (4)
    U[2][col] = 1; V[10][col] = 1; W[4][col] = 1
    col += 1
    # A02 B23 -> C03 (5)
    U[2][col] = 1; V[11][col] = 1; W[5][col] = 1
    col += 1
    # A12 B20 -> C10 (2)
    U[5][col] = 1; V[8][col] = 1; W[2][col] = 1
    col += 1
    # A12 B21 -> C11 (3)
    U[5][col] = 1; V[9][col] = 1; W[3][col] = 1
    col += 1
    # A12 B22 -> C12 (6)
    U[5][col] = 1; V[10][col] = 1; W[6][col] = 1
    col += 1
    # A12 B23 -> C13 (7)
    U[5][col] = 1; V[11][col] = 1; W[7][col] = 1
    col += 1
    
    if col != 22:
        raise ValueError(f"Expected 22 columns, got {col}")
        
    return {"U": U, "V": V, "W": W}
