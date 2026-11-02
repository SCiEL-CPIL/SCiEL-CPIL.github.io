import time

DIMENSIONS = [2, 2, 16]

# Strassen rank-7 decomposition for 2x2 matrix multiplication
# Standard Strassen formulas:
# M1 = (A11 + A22)(B11 + B22)
# M2 = (A21 + A22)B11
# M3 = A11(B12 - B22)
# M4 = A22(B21 - B11)
# M5 = (A11 + A12)B22
# M6 = (A21 - A11)(B11 + B12)
# M7 = (A12 - A22)(B21 + B22)
#
# C11 = M1 + M4 - M5 + M7
# C12 = M3 + M5
# C21 = M2 + M4
# C22 = M1 - M2 + M3 + M6

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 16
    # We'll group the p=16 columns into 8 pairs of 2 columns each
    # For each pair, apply Strassen's 7-term decomposition
    
    num_pairs = p // 2  # 8 pairs
    rank = 7 * num_pairs  # 56
    
    U = [[0] * rank for _ in range(n * m)]  # shape (4, 56)
    V = [[0] * rank for _ in range(m * p)]  # shape (32, 56)
    W = [[0] * rank for _ in range(p * n)]  # shape (32, 56)
    
    # Strassen intermediate forms for A (2x2):
    # A11 = U[0], A12 = U[1], A21 = U[2], A22 = U[3]
    # Strassen intermediates:
    # M1: (A11+A22)(B11+B22) -> A part: A11+A22
    # M2: (A21+A22)B11       -> A part: A21+A22
    # M3: A11(B12-B22)       -> A part: A11
    # M4: A22(B21-B11)       -> A part: A22
    # M5: (A11+A12)B22       -> A part: A11+A12
    # M6: (A21-A11)(B11+B12) -> A part: A21-A11
    # M7: (A12-A22)(B21+B22) -> A part: A12-A22
    
    # For B (2x16), with column pairs (b, b+1):
    # B11 = B[0, 2b], B12 = B[0, 2b+1], B21 = B[1, 2b], B22 = B[1, 2b+1]
    # (Row 0 = top row of B, Row 1 = bottom row of B)
    
    # V index: V[j*p + k] where j is row of B (0 or 1), k is column (0..15)
    # For pair b: columns are 2b and 2b+1
    # B[0, 2b] = V[0*p + 2b] = V[2b]
    # B[0, 2b+1] = V[2b+1]
    # B[1, 2b] = V[1*p + 2b] = V[16+2b]
    # B[1, 2b+1] = V[16+2b+1]
    
    # W index: W[k*n + i] where k is column of C (0..15), i is row of C (0 or 1)
    # C[0, 2b] = W[2b*2 + 0] = W[4b]
    # C[0, 2b+1] = W[4b+1]
    # C[1, 2b] = W[4b+2]
    # C[1, 2b+1] = W[4b+3]
    
    # Strassen B intermediates for each pair:
    # M1: B11+B22 = B[0,2b] + B[1,2b+1] -> V[2b] + V[16+2b+1]
    # M2: B11 = B[0,2b] -> V[2b]
    # M3: B12-B22 = B[0,2b+1] - B[1,2b+1] -> V[2b+1] - V[16+2b+1]
    # M4: B21-B11 = B[1,2b] - B[0,2b] -> V[16+2b] - V[2b]
    # M5: B22 = B[1,2b+1] -> V[16+2b+1]
    # M6: B11+B12 = B[0,2b] + B[0,2b+1] -> V[2b] + V[2b+1]
    # M7: B21+B22 = B[1,2b] + B[1,2b+1] -> V[16+2b] + V[16+2b+1]
    
    # C reconstruction:
    # C11 = M1 + M4 - M5 + M7 -> C[0,2b] = W[4b]
    # C12 = M3 + M5           -> C[0,2b+1] = W[4b+1]
    # C21 = M2 + M4           -> C[1,2b] = W[4b+2]
    # C22 = M1 - M2 + M3 + M6 -> C[1,2b+1] = W[4b+3]
    
    for b in range(num_pairs):
        for k in range(7):
            col = b * 7 + k
            
            # A part (U)
            if k == 0:   # M1: A11 + A22
                U[0][col] = 1
                U[3][col] = 1
            elif k == 1: # M2: A21 + A22
                U[2][col] = 1
                U[3][col] = 1
            elif k == 2: # M3: A11
                U[0][col] = 1
            elif k == 3: # M4: A22
                U[3][col] = 1
            elif k == 4: # M5: A11 + A12
                U[0][col] = 1
                U[1][col] = 1
            elif k == 5: # M6: A21 - A11
                U[2][col] = 1
                U[0][col] = -1
            elif k == 6: # M7: A12 - A22
                U[1][col] = 1
                U[3][col] = -1
            
            # B part (V)
            if k == 0:   # M1: B11 + B22
                V[2*b][col] = 1
                V[16+2*b+1][col] = 1
            elif k == 1: # M2: B11
                V[2*b][col] = 1
            elif k == 2: # M3: B12 - B22
                V[2*b+1][col] = 1
                V[16+2*b+1][col] = -1
            elif k == 3: # M4: B21 - B11
                V[16+2*b][col] = 1
                V[2*b][col] = -1
            elif k == 4: # M5: B22
                V[16+2*b+1][col] = 1
            elif k == 5: # M6: B11 + B12
                V[2*b][col] = 1
                V[2*b+1][col] = 1
            elif k == 6: # M7: B21 + B22
                V[16+2*b][col] = 1
                V[16+2*b+1][col] = 1
            
            # C part (W)
            if k == 0:   # M1
                W[4*b][col] = 1       # C11
                W[4*b+3][col] = 1     # C22
            elif k == 1: # M2
                W[4*b+2][col] = 1     # C21
                W[4*b+3][col] = -1    # C22
            elif k == 2: # M3
                W[4*b+1][col] = 1     # C12
                W[4*b+3][col] = 1     # C22
            elif k == 3: # M4
                W[4*b][col] = 1       # C11
                W[4*b+2][col] = 1     # C21
            elif k == 4: # M5
                W[4*b][col] = -1      # C11
                W[4*b+1][col] = 1     # C12
            elif k == 5: # M6
                W[4*b+3][col] = 1     # C22
            elif k == 6: # M7
                W[4*b][col] = 1       # C11
    
    return {"U": U, "V": V, "W": W}
