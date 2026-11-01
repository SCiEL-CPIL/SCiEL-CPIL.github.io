import numpy as np

DIMENSIONS = [2, 2, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 25
    rank = 25
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)] # 4 x 25
    V = [[0] * rank for _ in range(m * p)] # 14 x 25
    W = [[0] * rank for _ in range(p * n)] # 14 x 25
    
    # Indices for A (U)
    # A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    
    # Indices for B (V)
    # B[j,k] -> j*p + k
    # j=0: 0..6
    # j=1: 7..13
    
    # Indices for C (W)
    # C[i,k] -> W[k*n + i]
    # k=0: 0,1
    # k=1: 2,3
    # k=2: 4,5
    # k=3: 6,7
    # k=4: 8,9
    # k=5: 10,11
    # k=6: 12,13
    
    col = 0
    
    # Strassen helper for a 2x2 block of B at columns k0, k0+1
    def add_strassen_block(k0, start_col):
        nonlocal col
        # B indices
        b11_idx = 0 * p + k0
        b12_idx = 0 * p + (k0 + 1)
        b21_idx = 1 * p + k0
        b22_idx = 1 * p + (k0 + 1)
        
        # C indices
        c11_idx = k0 * n + 0
        c12_idx = (k0 + 1) * n + 0
        c21_idx = k0 * n + 1
        c22_idx = (k0 + 1) * n + 1
        
        # M1 = (a11 + a22)(b11 + b22)
        U[0][start_col] += 1
        U[3][start_col] += 1
        V[b11_idx][start_col] += 1
        V[b22_idx][start_col] += 1
        W[c11_idx][start_col] += 1
        W[c22_idx][start_col] += 1
        col += 1
        
        # M2 = (a21 + a22)b11
        U[2][start_col] += 1
        U[3][start_col] += 1
        V[b11_idx][start_col] += 1
        W[c21_idx][start_col] += 1
        W[c22_idx][start_col] -= 1
        col += 1
        
        # M3 = a11(b12 - b22)
        U[0][start_col] += 1
        V[b12_idx][start_col] += 1
        V[b22_idx][start_col] -= 1
        W[c12_idx][start_col] += 1
        W[c22_idx][start_col] += 1
        col += 1
        
        # M4 = a22(b21 - b11)
        U[3][start_col] += 1
        V[b21_idx][start_col] += 1
        V[b11_idx][start_col] -= 1
        W[c11_idx][start_col] += 1
        W[c21_idx][start_col] += 1
        col += 1
        
        # M5 = (a11 + a12)b22
        U[0][start_col] += 1
        U[1][start_col] += 1
        V[b22_idx][start_col] += 1
        W[c11_idx][start_col] -= 1
        W[c12_idx][start_col] += 1
        col += 1
        
        # M6 = (a21 - a11)(b11 + b12)
        U[2][start_col] += 1
        U[0][start_col] -= 1
        V[b11_idx][start_col] += 1
        V[b12_idx][start_col] += 1
        W[c22_idx][start_col] += 1
        col += 1
        
        # M7 = (a12 - a22)(b21 + b22)
        U[1][start_col] += 1
        U[3][start_col] -= 1
        V[b21_idx][start_col] += 1
        V[b22_idx][start_col] += 1
        W[c11_idx][start_col] += 1
        col += 1

    # Block 1: columns 0, 1
    add_strassen_block(0, col)
    
    # Block 2: columns 2, 3
    add_strassen_block(2, col)
    
    # Block 3: columns 4, 5
    add_strassen_block(4, col)
    
    # Block 4: column 6 (Schoolbook 2x2x1)
    # C[0,6] = A[0,0]B[0,6] + A[0,1]B[1,6]
    # C[1,6] = A[1,0]B[0,6] + A[1,1]B[1,6]
    
    # Term 1: A[0,0] B[0,6] -> C[0,6]
    U[0][col] = 1
    V[0 * p + 6][col] = 1
    W[6 * n + 0][col] = 1
    col += 1
    
    # Term 2: A[0,1] B[1,6] -> C[0,6]
    U[1][col] = 1
    V[1 * p + 6][col] = 1
    W[6 * n + 0][col] = 1
    col += 1
    
    # Term 3: A[1,0] B[0,6] -> C[1,6]
    U[2][col] = 1
    V[0 * p + 6][col] = 1
    W[6 * n + 1][col] = 1
    col += 1
    
    # Term 4: A[1,1] B[1,6] -> C[1,6]
    U[3][col] = 1
    V[1 * p + 6][col] = 1
    W[6 * n + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
