def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    col = 0
    for a in range(2):
        for c in range(4):
            # A_block[a]: entries A[i, 2a+b]
            # U row for A[i, j] is i*m + j = i*4 + j
            # B_block[a,c]: entries B[2a+b, 4c+d]
            # V row for B[j, k] is j*p + k = j*8 + k
            # W row for C[i, k] is k*n + i = k*2 + i
            
            A11_row = 0*4 + (2*a+0)  # A[0, 2a]
            A12_row = 0*4 + (2*a+1)  # A[0, 2a+1]
            A21_row = 1*4 + (2*a+0)  # A[1, 2a]
            A22_row = 1*4 + (2*a+1)  # A[1, 2a+1]
            
            B11_row = (2*a+0)*8 + (4*c+0)  # B[2a, 4c]
            B12_row = (2*a+0)*8 + (4*c+1)  # B[2a, 4c+1]
            B21_row = (2*a+1)*8 + (4*c+0)  # B[2a+1, 4c]
            B22_row = (2*a+1)*8 + (4*c+1)  # B[2a+1, 4c+1]
            
            C00_row = (4*c+0)*2 + 0  # C[0, 4c]
            C01_row = (4*c+1)*2 + 0  # C[0, 4c+1]
            C10_row = (4*c+0)*2 + 1  # C[1, 4c]
            C11_row = (4*c+1)*2 + 1  # C[1, 4c+1]
            
            # P1 = (A11+A22)(B11+B22)
            U[A11_row][col] = 1
            U[A22_row][col] = 1
            V[B11_row][col] = 1
            V[B22_row][col] = 1
            W[C00_row][col] = 1
            W[C11_row][col] = 1
            col += 1
            
            # P2 = (A21+A22)B11
            U[A21_row][col] = 1
            U[A22_row][col] = 1
            V[B11_row][col] = 1
            W[C10_row][col] = 1
            W[C11_row][col] = -1
            col += 1
            
            # P3 = A11(B12-B22)
            U[A11_row][col] = 1
            V[B12_row][col] = 1
            V[B22_row][col] = -1
            W[C01_row][col] = 1
            W[C11_row][col] = 1
            col += 1
            
            # P4 = A22(B21-B11)
            U[A22_row][col] = 1
            V[B21_row][col] = 1
            V[B11_row][col] = -1
            W[C00_row][col] = 1
            W[C10_row][col] = 1
            col += 1
            
            # P5 = (A11+A12)B22
            U[A11_row][col] = 1
            U[A12_row][col] = 1
            V[B22_row][col] = 1
            W[C00_row][col] = -1
            W[C01_row][col] = 1
            col += 1
            
            # P6 = (A21-A11)(B11+B12)
            U[A21_row][col] = 1
            U[A11_row][col] = -1
            V[B11_row][col] = 1
            V[B12_row][col] = 1
            W[C11_row][col] = 1
            col += 1
            
            # P7 = (A12-A22)(B21+B22)
            U[A12_row][col] = 1
            U[A22_row][col] = -1
            V[B21_row][col] = 1
            V[B22_row][col] = 1
            W[C00_row][col] = 1
            W[C11_row][col] = -1
            col += 1
    
    return {"U": U, "V": V, "W": W}
