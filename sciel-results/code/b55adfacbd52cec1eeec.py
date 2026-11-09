"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

DEADLINE = 140.0

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition using Fraction arithmetic."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if total != 1:
                    return False
    return True

def _strassen_2x2x2():
    """Return Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
    Returns (U, V, W) where U is 4x7, V is 4x7, W is 4x7.
    U indexed by [i*2+j], V by [j*2+k], W by [k*2+i] for i,j,k in {0,1}.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # U: 0=A00, 1=A01, 2=A10, 3=A11
    # V: 0=B00, 1=B01, 2=B10, 3=B11
    # W: 0=C00, 1=C01, 2=C10, 3=C11
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5
    
    # Block decomposition: split middle dim 5 into 2+3
    # A (2×5): A_left (2×2) cols 0-1, A_right (2×3) cols 2-4
    # B (5×5): B00 (2×2) rows 0-1 cols 0-1, B01 (2×3) rows 0-1 cols 2-4
    #           B10 (3×2) rows 2-4 cols 0-1, B11 (3×3) rows 2-4 cols 2-4
    # C (2×5): C_left (2×2) cols 0-1, C_right (2×3) cols 2-4
    # C_left = A_left·B00 + A_right·B10
    # C_right = A_left·B01 + A_right·B11
    
    # Ranks: Strassen(2,2,2)=7, schoolbook(2,3,2)=6, schoolbook(2,2,3)=6, schoolbook(2,3,3)=18
    total_rank = 7 + 6 + 6 + 18  # = 37
    U = [[0]*total_rank for _ in range(n*m)]  # 10 rows
    V = [[0]*total_rank for _ in range(m*p)]  # 25 rows
    W = [[0]*total_rank for _ in range(p*n)]  # 10 rows
    
    # 1. Strassen for A_left·B00 → C_left
    # A_left[i][j] = A[i][j], i∈{0,1}, j∈{0,1} → U[i*5+j]: U[0],U[1],U[5],U[6]
    # B00[j][k] = B[j][k], j∈{0,1}, k∈{0,1} → V[j*5+k]: V[0],V[1],V[5],V[6]
    # C_left[i][k] = C[i][k], i∈{0,1}, k∈{0,1} → W[k*2+i]: W[0],W[2],W[1],W[3]
    #   C[0][0]→W[0], C[0][1]→W[2], C[1][0]→W[1], C[1][1]→W[3]
    
    U_s, V_s, W_s = _strassen_2x2x2()
    # Map Strassen indices to our indices
    a_map = [0, 1, 5, 6]  # A00→U[0], A01→U[1], A10→U[5], A11→U[6]
    b_map = [0, 1, 5, 6]  # B00→V[0], B01→V[1], B10→V[5], B11→V[6]
    c_map = [0, 2, 1, 3]  # C00→W[0], C01→W[2], C10→W[1], C11→W[3]
    
    for c in range(7):
        for a_idx in range(4):
            U[a_map[a_idx]][c] = U_s[a_idx][c]
        for b_idx in range(4):
            V[b_map[b_idx]][c] = V_s[b_idx][c]
        for c_idx in range(4):
            W[c_map[c_idx]][c] = W_s[c_idx][c]
    
    # 2. Schoolbook for A_right·B10 → C_left
    # A_right[i][j] = A[i][j], i∈{0,1}, j∈{2,3,4} → U[i*5+j]: U[2],U[3],U[4],U[7],U[8],U[9]
    # B10[j][k] = B[j][k], j∈{2,3,4}, k∈{0,1} → V[j*5+k]: V[10],V[11],V[15],V[16],V[20],V[21]
    # C_left[i][k] = C[i][k], i∈{0,1}, k∈{0,1} → W[k*2+i]: W[0],W[2],W[1],W[3]
    
    col = 7
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                U[i*5 + j][col] = 1
                V[j*5 + k][col] = 1
                W[k*2 + i][col] = 1
                col += 1
    # col = 13
    
    # 3. Schoolbook for A_left·B01 → C_right
    # A_left[i][j] = A[i][j], i∈{0,1}, j∈{0,1} → U[i*5+j]: U[0],U[1],U[5],U[6]
    # B01[j][k] = B[j][k], j∈{0,1}, k∈{2,3,4} → V[j*5+k]: V[2],V[3],V[4],V[7],V[8],V[9]
    # C_right[i][k] = C[i][k], i∈{0,1}, k∈{2,3,4} → W[k*2+i]: W[4],W[6],W[5],W[8],W[7],W[9]
    #   C[0][2]→W[4], C[0][3]→W[6], C[0][4]→W[8], C[1][2]→W[5], C[1][3]→W[7], C[1][4]→W[9]
    
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                U[i*5 + j][col] = 1
                V[j*5 + k][col] = 1
                W[k*2 + i][col] = 1
                col += 1
    # col = 19
    
    # 4. Schoolbook for A_right·B11 → C_right
    # A_right[i][j] = A[i][j], i∈{0,1}, j∈{2,3,4} → U[i*5+j]: U[2],U[3],U[4],U[7],U[8],U[9]
    # B11[j][k] = B[j][k], j∈{2,3,4}, k∈{2,3,4} → V[j*5+k]: V[12],V[13],V[14],V[17],V[18],V[19],V[22],V[23],V[24]
    # C_right[i][k] = C[i][k], i∈{0,1}, k∈{2,3,4} → W[k*2+i]: W[4],W[6],W[8],W[5],W[7],W[9]
    
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 5):
                U[i*5 + j][col] = 1
                V[j*5 + k][col] = 1
                W[k*2 + i][col] = 1
                col += 1
    # col = 37
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
