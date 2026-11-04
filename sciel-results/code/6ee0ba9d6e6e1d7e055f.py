import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    rank = 71
    
    U = [[0] * rank for _ in range(n * m)]  # 15 x 71
    V = [[0] * rank for _ in range(m * p)]  # 25 x 71
    W = [[0] * rank for _ in range(p * n)]  # 15 x 71
    
    col = 0
    
    def schoolbook(i0, j0, k0, n_s, m_s, p_s):
        nonlocal col
        for i_s in range(n_s):
            for j_s in range(m_s):
                for k_s in range(p_s):
                    i = i0 + i_s
                    j = j0 + j_s
                    k = k0 + k_s
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    def strassen_2x2x2(i0, j0, k0):
        nonlocal col
        # M1 = (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
        U[i0*m+j0][col] = 1; U[(i0+1)*m+(j0+1)][col] = 1
        V[j0*p+k0][col] = 1; V[(j0+1)*p+(k0+1)][col] = 1
        W[k0*n+i0][col] = 1; W[(k0+1)*n+(i0+1)][col] = 1
        col += 1
        
        # M2 = (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
        U[(i0+1)*m+j0][col] = 1; U[(i0+1)*m+(j0+1)][col] = 1
        V[j0*p+k0][col] = 1
        W[k0*n+(i0+1)][col] = 1; W[(k0+1)*n+(i0+1)][col] = -1
        col += 1
        
        # M3 = A[i0,j0] * (B[j0,k0+1]-B[j0+1,k0+1])
        U[i0*m+j0][col] = 1
        V[j0*p+(k0+1)][col] = 1; V[(j0+1)*p+(k0+1)][col] = -1
        W[(k0+1)*n+i0][col] = 1; W[(k0+1)*n+(i0+1)][col] = 1
        col += 1
        
        # M4 = A[i0+1,j0+1] * (B[j0+1,k0]-B[j0,k0])
        U[(i0+1)*m+(j0+1)][col] = 1
        V[(j0+1)*p+k0][col] = 1; V[j0*p+k0][col] = -1
        W[k0*n+i0][col] = 1; W[(k0+1)*n+i0][col] = 1
        col += 1
        
        # M5 = (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
        U[i0*m+j0][col] = 1; U[i0*m+(j0+1)][col] = 1
        V[(j0+1)*p+(k0+1)][col] = 1
        W[k0*n+i0][col] = -1; W[(k0+1)*n+i0][col] = 1
        col += 1
        
        # M6 = (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
        U[(i0+1)*m+j0][col] = 1; U[i0*m+j0][col] = -1
        V[j0*p+k0][col] = 1; V[j0*p+(k0+1)][col] = 1
        W[(k0+1)*n+(i0+1)][col] = 1
        col += 1
        
        # M7 = (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0+1,k0+1])
        U[i0*m+(j0+1)][col] = 1; U[(i0+1)*m+(j0+1)][col] = -1
        V[(j0+1)*p+k0][col] = 1; V[(j0+1)*p+(k0+1)][col] = 1
        W[k0*n+i0][col] = 1
        col += 1
    
    # Block 1: A[0:2,0:2] x B[0:2,0:2] -> C[0:2,0:2], Strassen rank 7
    strassen_2x2x2(0, 0, 0)
    
    # Block 2: A[0:2,2:5] x B[2:5,0:2] -> C[0:2,0:2]
    # Split: 2x2x2 (j=2,3) + 2x1x2 (j=4)
    strassen_2x2x2(0, 2, 0)  # 2x2x2: A[0:2,2:4] x B[2:4,0:2]
    schoolbook(0, 4, 0, 2, 1, 2)  # 2x1x2: A[0:2,4:5] x B[4:5,0:2]
    
    # Block 3: A[0:2,0:2] x B[0:2,2:5] -> C[0:2,2:5]
    # Split: 2x2x2 (k=2,3) + 2x2x1 (k=4)
    strassen_2x2x2(0, 0, 2)  # 2x2x2: A[0:2,0:2] x B[0:2,2:4]
    schoolbook(0, 0, 4, 2, 2, 1)  # 2x2x1: A[0:2,0:2] x B[0:2,4:5]
    
    # Block 4: A[0:2,2:5] x B[2:5,2:5] -> C[0:2,2:5]
    # Split: 2x2x3 (j=2,3) + 2x1x3 (j=4)
    # 2x2x3 = 2x2x2 (k=2,3) + 2x2x1 (k=4)
    strassen_2x2x2(0, 2, 2)  # 2x2x2: A[0:2,2:4] x B[2:4,2:4]
    schoolbook(0, 2, 4, 2, 2, 1)  # 2x2x1: A[0:2,2:4] x B[2:4,4:5]
    schoolbook(0, 4, 2, 2, 1, 3)  # 2x1x3: A[0:2,4:5] x B[4:5,2:5]
    
    # Block 5: A[2:3,0:2] x B[0:2,0:2] -> C[2:3,0:2], schoolbook rank 4
    schoolbook(2, 0, 0, 1, 2, 2)
    
    # Block 6: A[2:3,2:5] x B[2:5,0:2] -> C[2:3,0:2], schoolbook rank 6
    schoolbook(2, 2, 0, 1, 3, 2)
    
    # Block 7: A[2:3,0:2] x B[0:2,2:5] -> C[2:3,2:5], schoolbook rank 6
    schoolbook(2, 0, 2, 1, 2, 3)
    
    # Block 8: A[2:3,2:5] x B[2:5,2:5] -> C[2:3,2:5], schoolbook rank 9
    schoolbook(2, 2, 2, 1, 3, 3)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
