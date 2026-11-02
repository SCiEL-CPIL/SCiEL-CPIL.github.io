import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank estimate: 21 + 12 + 21 = 54
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2 block
    # A cols: a_j1, a_j2. B rows: b_j1, b_j2. B cols: b_k1, b_k2.
    # C rows: c_i0, c_i1. C cols: c_k1, c_k2.
    def add_strassen(a_j1, a_j2, b_j1, b_j2, b_k1, b_k2):
        nonlocal col
        # M1
        U[a_j1][col] += 1
        U[a_j2][col] += 1
        V[b_j1][col] += 1
        V[b_j2][col] += 1
        W[b_k1 * n + 0][col] += 1
        W[b_k2 * n + 0][col] += 1
        W[b_k1 * n + 1][col] += 1
        W[b_k2 * n + 1][col] -= 1
        col += 1
        
        # M2
        U[a_j2][col] += 1
        U[a_j2+1][col] += 1 # Wait, a_j2 is index in A. A is 2x5.
        # A indices are i*m + j.
        # a_j1 is j1. a_j2 is j2.
        # U row index is i*m + j.
        # For M2: (A[1, j1] + A[1, j2]) * B[0, k]
        # A[1, j1] is index 1*m + j1.
        # A[1, j2] is index 1*m + j2.
        
        # Let's rewrite helper properly.
        
    # Redo helper
    def add_strassen_block(j1, j2, k1, k2):
        nonlocal col
        # A indices: 0*m+j1, 0*m+j2, 1*m+j1, 1*m+j2
        # B indices: j1*p+k1, j1*p+k2, j2*p+k1, j2*p+k2
        # C indices: k1*n+0, k1*n+1, k2*n+0, k2*n+1
        
        # M1 = (A00+A11)(B00+B11)
        U[0*m+j1][col] += 1
        U[1*m+j2][col] += 1
        V[j1*p+k1][col] += 1
        V[j2*p+k2][col] += 1
        W[k1*n+0][col] += 1
        W[k2*n+0][col] += 1
        W[k1*n+1][col] += 1
        W[k2*n+1][col] -= 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[1*m+j1][col] += 1
        U[1*m+j2][col] += 1
        V[j1*p+k1][col] += 1
        W[k1*n+1][col] += 1
        W[k2*n+1][col] += 1
        col += 1
        
        # M3 = A00(B01-B11)
        U[0*m+j1][col] += 1
        V[j1*p+k2][col] += 1
        V[j2*p+k2][col] -= 1
        W[k2*n+0][col] += 1
        W[k2*n+1][col] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[1*m+j2][col] += 1
        V[j2*p+k1][col] += 1
        V[j1*p+k1][col] -= 1
        W[k1*n+0][col] += 1
        W[k1*n+1][col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[0*m+j1][col] += 1
        U[0*m+j2][col] += 1
        V[j2*p+k2][col] += 1
        W[k2*n+0][col] -= 1
        W[k2*n+1][col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[1*m+j1][col] += 1
        U[0*m+j1][col] -= 1
        V[j1*p+k1][col] += 1
        V[j1*p+k2][col] += 1
        W[k2*n+1][col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[0*m+j2][col] += 1
        U[1*m+j2][col] -= 1
        V[j2*p+k1][col] += 1
        V[j2*p+k2][col] += 1
        W[k1*n+0][col] += 1
        W[k1*n+1][col] += 1
        col += 1

    # Block 1: A cols 0,1. B rows 0,1.
    add_strassen_block(0, 1, 0, 1)
    add_strassen_block(0, 1, 2, 3)
    add_strassen_block(0, 1, 4, 5)
    
    # Block 2: A col 2. B row 2.
    for i in range(n):
        for k in range(p):
            U[i*m+2][col] = 1
            V[2*p+k][col] = 1
            W[k*n+i][col] = 1
            col += 1
            
    # Block 3: A cols 3,4. B rows 3,4.
    add_strassen_block(3, 4, 0, 1)
    add_strassen_block(3, 4, 2, 3)
    add_strassen_block(3, 4, 4, 5)
    
    return {"U": U, "V": V, "W": W}
