import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 57
    rank = 57
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        # Indices for A: i in {i0, i0+1}, j in {j0, j0+1}
        # Indices for B: j in {j0, j0+1}, k in {k0, k0+1}
        # Indices for C: i in {i0, i0+1}, k in {k0, k0+1}
        
        # Strassen coefficients
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A11+A22)B21
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A12+A22)B22
        # M6 = (A11-A12)(B11+B12)
        # M7 = (A21+A22)(B21-B11)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # A indices: 
        # A11 -> i0, j0
        # A12 -> i0, j0+1
        # A21 -> i0+1, j0
        # A22 -> i0+1, j0+1
        
        # B indices:
        # B11 -> j0, k0
        # B12 -> j0, k0+1
        # B21 -> j0+1, k0
        # B22 -> j0+1, k0+1
        
        # C indices:
        # C11 -> i0, k0
        # C12 -> i0, k0+1
        # C21 -> i0+1, k0
        # C22 -> i0+1, k0+1
        
        # M1
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M2
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        W[(k0)*n + (i0+1)][col] += 1
        col += 1
        
        # M3
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += -1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        
        # M4
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] += -1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        # M5
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += -1
        col += 1
        
        # M6
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += -1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M7
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] += -1
        W[k0*n + i0][col] += 1
        col += 1

    # Helper to add a schoolbook block of size n_b x m_b x p_b
    def add_schoolbook_block(i0, j0, k0, n_b, m_b, p_b):
        nonlocal col
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U[(i0+i)*m + (j0+j)][col] = 1
                    V[(j0+j)*p + (k0+k)][col] = 1
                    W[(k0+k)*n + (i0+i)][col] = 1
                    col += 1

    # Split m=5 into 2 and 3
    # Block 1: A1 (2x2), B1 (2x6)
    # Split B1 into three 2x2 blocks: (j=0..1, k=0..1), (j=0..1, k=2..3), (j=0..1, k=4..5)
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # Block 2: A2 (2x3), B2 (3x6)
    # Split B2 into three 3x2 blocks: (j=2..4, k=0..1), (j=2..4, k=2..3), (j=2..4, k=4..5)
    # Each is 2x3x2 schoolbook
    add_schoolbook_block(0, 2, 0, 2, 3, 2)
    add_schoolbook_block(0, 2, 2, 2, 3, 2)
    add_schoolbook_block(0, 2, 4, 2, 3, 2)
    
    return {"U": U, "V": V, "W": W}
