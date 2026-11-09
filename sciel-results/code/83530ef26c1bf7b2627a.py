"""Matrix multiplication 3x4x8: rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    # Rank calculation:
    # Top 2x4x8 part:
    #   Split B into two 4x4 blocks.
    #   For each 4x4 block, split A (2x4) into two 2x2 and B (4x4) into two 2x4.
    #   Result is sum of two 2x2x4 multiplications.
    #   Each 2x2x4 is two 2x2x2 (Strassen, rank 7). So 2*7=14.
    #   Two 2x2x4s per 4x4 block -> 28.
    #   Two 4x4 blocks -> 56.
    # Bottom 1x4x8 part:
    #   Trivial rank 1*4*8 = 32.
    # Total rank = 56 + 32 = 88.
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]  # shape (12, 88)
    V = [[0] * rank for _ in range(m * p)]  # shape (32, 88)
    W = [[0] * rank for _ in range(p * n)]  # shape (24, 88)
    
    col = 0
    
    # Helper to add Strassen 2x2x2 decomposition
    # A indices: (i1, j1), (i1, j2), (i2, j1), (i2, j2)
    # B indices: (j1, k1), (j1, k2), (j2, k1), (j2, k2)
    # C indices: (i1, k1), (i1, k2), (i2, k1), (i2, k2)
    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
        U[i1*m+j1][col] += 1
        U[i2*m+j2][col] += 1
        V[j1*p+k1][col] += 1
        V[j2*p+k2][col] += 1
        W[k1*n+i1][col] += 1
        W[k2*n+i1][col] += 1
        W[k1*n+i2][col] += 1
        W[k2*n+i2][col] -= 1
        col += 1
        
        # M2 = (A[i2,j1]+A[i2,j2]) * B[j1,k1]
        U[i2*m+j1][col] += 1
        U[i2*m+j2][col] += 1
        V[j1*p+k1][col] += 1
        W[k1*n+i2][col] += 1
        W[k2*n+i2][col] -= 1
        col += 1
        
        # M3 = A[i1,j1] * (B[j1,k2]-B[j2,k2])
        U[i1*m+j1][col] += 1
        V[j1*p+k2][col] += 1
        V[j2*p+k2][col] -= 1
        W[k2*n+i1][col] += 1
        W[k2*n+i2][col] += 1
        col += 1
        
        # M4 = A[i2,j2] * (B[j2,k1]-B[j1,k1])
        U[i2*m+j2][col] += 1
        V[j2*p+k1][col] += 1
        V[j1*p+k1][col] -= 1
        W[k1*n+i2][col] += 1
        W[k2*n+i2][col] -= 1
        col += 1
        
        # M5 = (A[i1,j1]+A[i1,j2]) * B[j2,k2]
        U[i1*m+j1][col] += 1
        U[i1*m+j2][col] += 1
        V[j2*p+k2][col] += 1
        W[k2*n+i1][col] -= 1
        W[k2*n+i2][col] += 1
        col += 1
        
        # M6 = (A[i2,j1]-A[i1,j1]) * (B[j1,k1]+B[j1,k2])
        U[i2*m+j1][col] += 1
        U[i1*m+j1][col] -= 1
        V[j1*p+k1][col] += 1
        V[j1*p+k2][col] += 1
        W[k1*n+i2][col] += 1
        W[k2*n+i2][col] += 1
        col += 1
        
        # M7 = (A[i1,j2]-A[i2,j2]) * (B[j2,k1]+B[j2,k2])
        U[i1*m+j2][col] += 1
        U[i2*m+j2][col] -= 1
        V[j2*p+k1][col] += 1
        V[j2*p+k2][col] += 1
        W[k1*n+i1][col] -= 1
        W[k2*n+i1][col] -= 1
        col += 1

    # Top 2x4x8 Part (Rows 0,1 of A)
    # Split B (4x8) into B_left (4x4, cols 0-3) and B_right (4x4, cols 4-7)
    
    # B_left (cols 0-3)
    # A_top (rows 0,1, cols 0,1) * B_left_top (rows 0,1, cols 0,3) -> C_top_left (rows 0,1, cols 0,3)
    # Wait, the 2x2x4 decomposition splits B (2x4) into two 2x2.
    # So for A(2x4) * B(4x4):
    # A = [A11 A12] (2x2 each)
    # B = [B11; B21] (2x4 each)
    # C = A11 B11 + A12 B21
    # A11 B11 is 2x2x4. B11 is 2x4. Split B11 into B11_L (2x2) and B11_R (2x2).
    # So A11 B11_L and A11 B11_R are 2x2x2.
    
    # Let's map indices carefully.
    # A indices: i in {0,1}, j in {0,1,2,3}
    # B indices: j in {0,1,2,3}, k in {0,1,2,3} (for left block)
    # C indices: i in {0,1}, k in {0,1,2,3}
    
    # Term 1: A[0:2, 0:2] * B[0:2, 0:4]
    # A11 = A[0:2, 0:2]
    # B11 = B[0:2, 0:4]
    # B11 split into B11_L (cols 0,1) and B11_R (cols 2,3)
    # A11 * B11_L: i1=0, i2=1, j1=0, j2=1, k1=0, k2=1
    add_strassen(0, 1, 0, 1, 0, 1)
    # A11 * B11_R: i1=0, i2=1, j1=0, j2=1, k1=2, k2=3
    add_strassen(0, 1, 0, 1, 2, 3)
    
    # Term 2: A[0:2, 2:4] * B[2:4, 0:4]
    # A12 = A[0:2, 2:4]
    # B21 = B[2:4, 0:4]
    # B21 split into B21_L (cols 0,1) and B21_R (cols 2,3)
    # A12 * B21_L: i1=0, i2=1, j1=2, j2=3, k1=0, k2=1
    add_strassen(0, 1, 2, 3, 0, 1)
    # A12 * B21_R: i1=0, i2=1, j1=2, j2=3, k1=2, k2=3
    add_strassen(0, 1, 2, 3, 2, 3)
    
    # B_right (cols 4-7)
    # Term 3: A[0:2, 0:2] * B[0:2, 4:8]
    # A11 * B12_L (cols 4,5): k1=4, k2=5
    add_strassen(0, 1, 0, 1, 4, 5)
    # A11 * B12_R (cols 6,7): k1=6, k2=7
    add_strassen(0, 1, 0, 1, 6, 7)
    
    # Term 4: A[0:2, 2:4] * B[2:4, 4:8]
    # A12 * B22_L (cols 4,5): k1=4, k2=5
    add_strassen(0, 1, 2, 3, 4, 5)
    # A12 * B22_R (cols 6,7): k1=6, k2=7
    add_strassen(0, 1, 2, 3, 6, 7)
    
    # Bottom 1x4x8 Part (Row 2 of A)
    for j in range(4):
        for k in range(8):
            U[2*m+j][col] = 1
            V[j*p+k][col] = 1
            W[k*n+2][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
