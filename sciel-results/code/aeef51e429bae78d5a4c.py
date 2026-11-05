import time
import sys

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 46 decomposition using two disjoint Strassen blocks
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {0,1}, j in {2,3}, k in {2,3}
    # Rest: Schoolbook
    
    # Total rank: 7 (Strassen 1) + 7 (Strassen 2) + (48 - 8 - 8) (Schoolbook) = 14 + 32 = 46
    rank = 46
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        # Indices for A: i in {i0, i0+1}, j in {j0, j0+1}
        # Indices for B: j in {j0, j0+1}, k in {k0, k0+1}
        # Indices for C: k in {k0, k0+1}, i in {i0, i0+1}
        
        # A indices (row, col) -> index in U
        a00 = i0 * m + j0
        a01 = i0 * m + (j0 + 1)
        a10 = (i0 + 1) * m + j0
        a11 = (i0 + 1) * m + (j0 + 1)
        
        # B indices (row, col) -> index in V
        # B is m x p. Row j, Col k.
        b00 = j0 * p + k0
        b01 = j0 * p + (k0 + 1)
        b10 = (j0 + 1) * p + k0
        b11 = (j0 + 1) * p + (k0 + 1)
        
        # C indices (row, col) -> index in W
        # W is p x n. Row k, Col i.
        c00 = k0 * n + i0
        c01 = k0 * n + (i0 + 1)
        c10 = (k0 + 1) * n + i0
        c11 = (k0 + 1) * n + (i0 + 1)
        
        # Strassen Multiplications
        # M1 = A00 * (B00 + B11)
        U[a00][col] += 1
        V[b00][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1  # C00 += M1
        W[c11][col] += 1  # C11 += M1
        col += 1
        
        # M2 = (A00 + A10) * B11
        U[a00][col] += 1
        U[a10][col] += 1
        V[b11][col] += 1
        W[c10][col] += 1  # C10 += M2
        W[c11][col] += -1 # C11 -= M2
        col += 1
        
        # M3 = (A01 + A11) * B00
        U[a01][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        W[c01][col] += 1  # C01 += M3
        W[c11][col] += 1  # C11 += M3
        col += 1
        
        # M4 = A11 * (B10 - B00)
        U[a11][col] += 1
        V[b10][col] += 1
        V[b00][col] += -1
        W[c00][col] += 1  # C00 += M4
        W[c10][col] += 1  # C10 += M4
        col += 1
        
        # M5 = (A00 + A01) * B10
        U[a00][col] += 1
        U[a01][col] += 1
        V[b10][col] += 1
        W[c00][col] += -1 # C00 -= M5
        W[c01][col] += 1  # C01 += M5
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[a10][col] += 1
        U[a00][col] += -1
        V[b00][col] += 1
        V[b01][col] += 1
        W[c11][col] += 1  # C11 += M6
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[a01][col] += 1
        U[a11][col] += -1
        V[b10][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1  # C00 += M7
        col += 1

    # Apply Strassen to Block 1: i={0,1}, j={0,1}, k={0,1}
    add_strassen_block(0, 0, 0)
    
    # Apply Strassen to Block 2: i={0,1}, j={2,3}, k={2,3}
    add_strassen_block(0, 2, 2)
    
    # Apply Schoolbook to the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this term is covered by Strassen blocks
                in_block1 = (i < 2 and j < 2 and k < 2)
                in_block2 = (i < 2 and j >= 2 and j < 4 and k >= 2 and k < 4)
                
                if in_block1 or in_block2:
                    continue
                
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
