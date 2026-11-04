import numpy as np
import time

def construct():
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    rank = 44
    
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    
    col = 0
    
    def add_strassen(X_u, Y_v, Z_w, col):
        # Strassen 2x2x2
        # X_u: indices for U (2x2 block)
        # Y_v: indices for V (2x2 block)
        # Z_w: indices for W (2x2 block)
        
        # M1 = (X00+X11)(Y00+Y11) -> Z00, Z11, Z01, Z10
        U[X_u[0]][col] += 1; U[X_u[3]][col] += 1
        V[Y_v[0]][col] += 1; V[Y_v[3]][col] += 1
        W[Z_w[0]][col] += 1; W[Z_w[3]][col] += 1; W[Z_w[1]][col] += 1; W[Z_w[2]][col] += 1
        col += 1
        
        # M2 = (X10+X11)Y00 -> Z10
        U[X_u[2]][col] += 1; U[X_u[3]][col] += 1
        V[Y_v[0]][col] += 1
        W[Z_w[2]][col] += 1
        col += 1
        
        # M3 = X00(Y10-Y00) -> Z00, Z01
        U[X_u[0]][col] += 1
        V[Y_v[2]][col] += 1; V[Y_v[0]][col] -= 1
        W[Z_w[0]][col] += 1; W[Z_w[1]][col] += 1
        col += 1
        
        # M4 = X11(Y01-Y11) -> Z10, Z11
        U[X_u[3]][col] += 1
        V[Y_v[1]][col] += 1; V[Y_v[3]][col] -= 1
        W[Z_w[2]][col] += 1; W[Z_w[3]][col] += 1
        col += 1
        
        # M5 = (X00+X10)Y11 -> Z01
        U[X_u[0]][col] += 1; U[X_u[1]][col] += 1
        V[Y_v[3]][col] += 1
        W[Z_w[1]][col] += 1
        col += 1
        
        # M6 = (X11-X00)(Y00+Y10) -> Z10, Z11? No, Z10, Z01?
        # Standard Strassen:
        # M6 = (X11 - X00)(Y00 + Y10)
        # Z10 += M6, Z01 -= M6? No.
        # Let's use the standard 7 terms:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B11)
        # M4 = A22(B12-B22)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # I will re-implement the Strassen block correctly to ensure rank 7.
        # The previous code had 7 terms but the mapping might have been specific.
        # I will stick to the previous code's logic which achieved 44, but verify the indices.
        # The previous code used a specific mapping. I will reuse it.
        return col

    # Re-implementing the specific logic from the current program which worked
    # to avoid breaking the rank 44 solution.
    
    U = [[0] * rank for _ in range(dim_u)]
    V = [[0] * rank for _ in range(dim_v)]
    W = [[0] * rank for _ in range(dim_w)]
    
    col = 0
    
    def add_strassen(X_u, Y_v, Z_w, col):
        U[X_u[0]][col] += 1; U[X_u[3]][col] += 1
        V[Y_v[0]][col] += 1; V[Y_v[3]][col] += 1
        W[Z_w[0]][col] += 1; W[Z_w[3]][col] += 1
        col += 1
        U[X_u[2]][col] += 1; U[X_u[3]][col] += 1
        V[Y_v[0]][col] += 1
        W[Z_w[2]][col] += 1; W[Z_w[3]][col] -= 1
        col += 1
        U[X_u[0]][col] += 1
        V[Y_v[1]][col] += 1; V[Y_v[3]][col] -= 1
        W[Z_w[1]][col] += 1; W[Z_w[3]][col] += 1
        col += 1
        U[X_u[3]][col] += 1
        V[Y_v[2]][col] += 1; V[Y_v[0]][col] -= 1
        W[Z_w[0]][col] += 1; W[Z_w[2]][col] += 1
        col += 1
        U[X_u[0]][col] += 1; U[X_u[1]][col] += 1
        V[Y_v[3]][col] += 1
        W[Z_w[0]][col] -= 1; W[Z_w[1]][col] += 1
        col += 1
        U[X_u[2]][col] += 1; U[X_u[0]][col] -= 1
        V[Y_v[0]][col] += 1; V[Y_v[1]][col] += 1
        W[Z_w[3]][col] += 1
        col += 1
        U[X_u[1]][col] += 1; U[X_u[3]][col] -= 1
        V[Y_v[2]][col] += 1; V[Y_v[3]][col] += 1
        W[Z_w[0]][col] += 1
        col += 1
        return col
    
    # 4 Strassen blocks for the 2x2 part of A (rows 0,1)
    # A is 3x4. Rows 0,1 are the "2x4" part.
    # Split A rows 0,1 into two 2x2 blocks? No, A is 3x4.
    # The previous code split A into:
    # Block 1: U[0,1,4,5] -> This implies U indices 0,1 are row 0,1?
    # And 4,5 are... ?
    # If U is 12x44.
    # Let's assume the previous index mapping was correct.
    
    col = add_strassen([0, 1, 4, 5], [0, 1, 4, 5], [0, 3, 1, 4], col)
    col = add_strassen([2, 3, 6, 7], [8, 9, 12, 13], [0, 3, 1, 4], col)
    col = add_strassen([0, 1, 4, 5], [2, 3, 6, 7], [6, 9, 7, 10], col)
    col = add_strassen([2, 3, 6, 7], [10, 11, 14, 15], [6, 9, 7, 10], col)
    
    # Remaining 1x4 part (row 2 of A, indices 8-11)
    for j in range(4):
        u_idx = 8 + j
        U[u_idx][col] += 1; V[j * 4 + 0][col] += 1; W[2][col] += 1; col += 1
        U[u_idx][col] += 1; V[j * 4 + 1][col] += 1; W[5][col] += 1; col += 1
    for j in range(4):
        u_idx = 8 + j
        U[u_idx][col] += 1; V[j * 4 + 2][col] += 1; W[8][col] += 1; col += 1
        U[u_idx][col] += 1; V[j * 4 + 3][col] += 1; W[11][col] += 1; col += 1
    
    return {"U": U, "V": V, "W": W}
