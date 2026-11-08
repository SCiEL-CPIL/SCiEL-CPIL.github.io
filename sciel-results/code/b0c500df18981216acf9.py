import numpy as np
import time

DIMENSIONS = [2, 4, 7]
START_TIME = time.time()
TIME_LIMIT = 140  # safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # n=2 (rows of A), m=4 (inner), p=7 (cols of B)
    
    # Try a different block partitioning to potentially reduce rank
    # Split p=7 into blocks of size 3, 2, 2
    # For 2x4x3: use a custom decomposition
    # For 2x4x2: use two 2x2x2 Strassen (rank 14)
    # For 2x4x2: use two 2x2x2 Strassen (rank 14)
    
    # Actually, let's try to find a better decomposition for 2x4x3
    # 2x4x3 can be split as 2x2x3 + 2x2x3
    # 2x2x3: split k=3 into 2+1: 2x2x2 (rank 7) + 2x2x1 (rank 4) = 11
    # So 2x4x3: 2*11 = 22
    # 2x4x2: 2*7 = 14
    # Total: 22 + 14 + 14 = 50
    
    # Let's try yet another approach: use a 2x3x7 block and 2x1x7 block
    # 2x1x7: rank 7
    # 2x3x7: need to find a good decomposition
    # 2x3x7: split inner dim 3 into... this is tricky
    
    # For now, let's stick with the known rank 50 decomposition but verify it's correct
    rank = 50
    U = [[0] * rank for _ in range(n * m)]   # shape (8, 50)
    V = [[0] * rank for _ in range(m * p)]   # shape (28, 50)
    W = [[0] * rank for _ in range(p * n)]   # shape (14, 50)
    
    col = 0
    
    def add_strassen_222(j0, k0):
        """Strassen for A[:, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[:, k0:k0+2]"""
        nonlocal col
        # A indices
        a00 = 0 * m + j0
        a01 = 0 * m + (j0 + 1)
        a10 = 1 * m + j0
        a11 = 1 * m + (j0 + 1)
        
        # B indices
        b00 = j0 * p + k0
        b01 = j0 * p + (k0 + 1)
        b10 = (j0 + 1) * p + k0
        b11 = (j0 + 1) * p + (k0 + 1)
        
        # C indices (col, row)
        c00 = k0 * n + 0
        c01 = (k0 + 1) * n + 0
        c10 = k0 * n + 1
        c11 = (k0 + 1) * n + 1
        
        # M1 = (a00+a11)(b00+b11) -> C00 += M1, C11 += M1
        U[a00][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M2 = (a10+a11)*b00 -> C10 += M2, C11 -= M2
        U[a10][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1
        W[c11][col] -= 1
        col += 1
        
        # M3 = a00*(b01-b11) -> C01 += M3, C11 += M3
        U[a00][col] += 1
        V[b01][col] += 1
        V[b11][col] -= 1
        W[c01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M4 = a11*(b10-b00) -> C00 += M4, C10 += M4
        U[a11][col] += 1
        V[b10][col] += 1
        V[b00][col] -= 1
        W[c00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # M5 = (a00+a01)*b11 -> C00 -= M5, C01 += M5
        U[a00][col] += 1
        U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1
        W[c01][col] += 1
        col += 1
        
        # M6 = (a10-a00)*(b00+b01) -> C11 += M6
        U[a10][col] += 1
        U[a00][col] -= 1
        V[b00][col] += 1
        V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7 = (a01-a11)*(b10+b11) -> C00 += M7
        U[a01][col] += 1
        U[a11][col] -= 1
        V[b10][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        col += 1
    
    def add_schoolbook_221(j0, k0):
        """Schoolbook for A[:, j0:j0+2] * B[j0:j0+2, k0:k0+1] -> C[:, k0:k0+1]"""
        nonlocal col
        a00 = 0 * m + j0
        a01 = 0 * m + (j0 + 1)
        a10 = 1 * m + j0
        a11 = 1 * m + (j0 + 1)
        
        b00 = j0 * p + k0
        b10 = (j0 + 1) * p + k0
        
        c00 = k0 * n + 0
        c10 = k0 * n + 1
        
        # C[0,k0] = a00*b00 + a01*b10
        U[a00][col] += 1
        V[b00][col] += 1
        W[c00][col] += 1
        col += 1
        
        U[a01][col] += 1
        V[b10][col] += 1
        W[c00][col] += 1
        col += 1
        
        # C[1,k0] = a10*b00 + a11*b10
        U[a10][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1
        col += 1
        
        U[a11][col] += 1
        V[b10][col] += 1
        W[c10][col] += 1
        col += 1
    
    # Process each inner block (j=0..1 and j=2..3)
    for j_start in [0, 2]:
        # For each column block of B: [0,1], [2,3], [4,5], [6]
        for k_start in [0, 2, 4]:
            add_strassen_222(j_start, k_start)
        add_schoolbook_221(j_start, 6)
    
    return {"U": U, "V": V, "W": W}
