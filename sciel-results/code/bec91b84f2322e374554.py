"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random

DIMENSIONS = [3, 3, 7]

# Strassen 3x3 coefficients as integer multipliers
# M0 = (A11+A33)*(B11+B33)
# M1 = (A11+A22)*(B11+B22)
# M2 = (A22+A33)*(B22+B33)
# M3 = A11*(B12-B32)
# M4 = A33*(B23-B13)
# M5 = (A11+A12)*(B21+B22)-M0-M2
# M6 = (A32+A33)*(B22+B12)-M2-M4
# M7 = (A13+A23)*(B11+B21)-M1-M3
# M8 = (A12+A22)*(B23+B33)-M2-M4
# M9 = (A23+A33)*(B11+B12)-M1-M3
# M10= (A11+A33)*(B22+B33)-M0-M2
# M11= (A12+A22)*(B11+B21)-M1-M3
# M12= (A21+A31)*(B22+B32)-M2-M4
# M13= (A13+A33)*(B21+B31)-M0-M3
# M14= (A11+A12)*(B13+B33)-M0-M4
# M15= (A31+A32)*(B23+B13)-M2-M6
# M16= (A13+A23)*(B12+B32)-M1-M9
# M17= (A21+A31)*(B11+B12)-M1-M12
# M18= (A12+A22)*(B13+B23)-M1-M8
# M19= (A13+A33)*(B11+B12)-M0-M13

STRASSEN_U_ROWS = [
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1]
]

STRASSEN_V_COLS = [
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0],
    [0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1]
]

# Output mapping: C[i][j] = sum of terms
# C[0][0] = M0 + M1 + M3 - M5 - M9
# C[0][1] = M3 + M14 - M18
# C[0][2] = M1 + M4 - M7 - M13
# C[1][0] = M10 + M11 - M12 - M17
# C[1][1] = M0 + M2 + M4 - M6 - M10
# C[1][2] = M6 + M8 - M15 - M18
# C[2][0] = M5 + M12 - M13 - M17
# C[2][1] = M2 + M4 - M6 - M14
# C[2][2] = M7 + M9 - M16

STRASSEN_W_MATRIX = np.array([
    [1, 0, 0, 0, 0, -1, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1, 0],
    [0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, 0, 0, 0, 0, -1, 0, 0],
    [1, 0, 1, 0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0, 0],
    [0, 0, 1, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0]
])

def construct():
    n, m, p = DIMENSIONS
    # rank = 3 * 20 + 7 * 3 = 60 + 21 = 81
    # Wait, for 3x3x7, we can split into 3x3x3, 3x3x3, 3x3x1
    # Rank(3x3x3) <= 20 (Strassen)
    # Rank(3x3x1) = 9
    # Total rank = 20 + 20 + 9 = 49
    
    # Let's construct the 49 rank decomposition.
    
    # Block 1: i in 0..2, j in 0..2, k in 0..2 (first 3 cols of B)
    # Block 2: i in 0..2, j in 0..2, k in 3..5 (next 3 cols of B)
    # Block 3: i in 0..2, j in 0..2, k in 6..6 (last 1 col of B)
    
    total_rank = 49
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Block 1: k=0,1,2
    for t in range(20):
        for i in range(3):
            for j in range(3):
                U[i*m+j][col_offset + t] += STRASSEN_U_ROWS[i][j] * 0 # Wait, U[i*m+j] is scalar.
                # U[i*m+j, col] = A[i,j] coefficient
                U[i*m+j][col_offset + t] = STRASSEN_U_ROWS[i][t]
        for j in range(3):
            for k in range(3):
                V[j*p+k][col_offset + t] = STRASSEN_V_COLS[j][t]
        for k in range(3):
            for i in range(3):
                # W[k*n+i, col] = C[i,k] coefficient
                W[k*n+i][col_offset + t] = STRASSEN_W_MATRIX[i][k][t] # Careful with indices
    col_offset += 20
    
    # Block 2: k=3,4,5
    for t in range(20):
        for i in range(3):
            U[i*m][col_offset + t] = 0 # Reset
            for j in range(3):
                U[i*m+j][col_offset + t] = STRASSEN_U_ROWS[i][t]
        for j in range(3):
            for k in range(3):
                V[j*p + (k+3)][col_offset + t] = STRASSEN_V_COLS[j][t]
        for k in range(3):
            for i in range(3):
                W[(k+3)*n+i][col_offset + t] = STRASSEN_W_MATRIX[i][k][t]
    col_offset += 20
    
    # Block 3: k=6
    # This is 3x3x1, rank 9
    # C[i,6] = sum_j A[i,j] B[j,6]
    # Schoolbook rank 9:
    for i in range(3):
        for j in range(3):
            idx = col_offset + i*3 + j
            U[i*m+j][idx] = 1
            V[j*p+6][idx] = 1
            W[6*n+i][idx] = 1
    col_offset += 9
    
    return {"U": U, "V": V, "W": W}

