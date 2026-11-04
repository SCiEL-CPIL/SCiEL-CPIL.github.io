import numpy as np
import time

def construct():
    n, m, p = 3, 3, 7
    # Shapes
    # U: (n*m, R)
    # V: (m*p, R)
    # W: (p*n, R)
    
    # We use Rank 62 decomposition
    # 26 terms for 3x3x3 core
    # 36 terms for 3x3x4 extension (schoolbook)
    R = 62
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Indices
    # U[i*m + j]
    # V[j*p + k]
    # W[k*n + i]
    
    # 1. Core 3x3x3 Decomposition (Rank 26)
    # Strassen 7 terms for i,j,k in {0,1}
    
    # M1 = (a00 + a11)(b00 + b11) -> c00, c01, c10, c11 contributions
    # Standard Strassen for 2x2:
    # M1 = (a+e)(j+n)
    # M2 = (d+e)j
    # M3 = a(k-n)
    # M4 = e(m-j)
    # M5 = (a+b)n
    # M6 = (d-a)(j+k)
    # M7 = (b-e)(m+n)
    # C00 = M1 - M4 - M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 - M5 + M6
    
    # Term 0: M1
    U[0, 0] = 1; U[4, 0] = 1  # a00, a11
    V[0, 0] = 1; V[4, 0] = 1  # b00, b11
    W[0, 0] = 1; W[1, 0] = 1; W[3, 0] = 1; W[4, 0] = 1 # c00, c01, c10, c11
    
    # Term 1: M2
    U[3, 1] = 1; U[4, 1] = 1  # a10, a11
    V[0, 1] = 1               # b00
    W[3, 1] = 1               # c10
    
    # Term 2: M3
    U[0, 2] = 1               # a00
    V[1, 2] = 1; V[4, 2] = -1 # b01, -b11
    W[1, 2] = 1               # c01
    
    # Term 3: M4
    U[4, 3] = 1               # a11
    V[3, 3] = 1; V[0, 3] = -1 # b10, -b00
    W[0, 3] = -1; W[3, 3] = 1 # -c00, c10
    
    # Term 4: M5
    U[0, 4] = 1; U[1, 4] = 1  # a00, a01
    V[4, 4] = 1               # b11
    W[1, 4] = 1; W[4, 4] = -1 # c01, -c11
    
    # Term 5: M6
    U[3, 5] = 1; U[0, 5] = -1 # a10, -a00
    V[0, 5] = 1; V[1, 5] = 1  # b00, b01
    W[0, 5] = -1; W[3, 5] = -1 # -c00, -c10
    
    # Term 6: M7
    U[1, 6] = 1; U[4, 6] = -1 # a01, -a11
    V[3, 6] = 1; V[4, 6] = 1  # b10, b11
    W[0, 6] = 1; W[4, 6] = 1  # c00, c11
    
    # Cross terms for 3x3x3 (19 terms)
    # v s^T (4 terms)
    # v = [a02, a12], s^T = [b20, b21]
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1 # a02 b20 -> c00
    U[2, 8] = 1; V[7, 8] = 1; W[1, 8] = 1 # a02 b21 -> c01
    U[5, 9] = 1; V[6, 9] = 1; W[3, 9] = 1 # a12 b20 -> c10
    U[5, 10] = 1; V[7, 10] = 1; W[4, 10] = 1 # a12 b21 -> c11
    
    # A_c u (4 terms)
    # u = [b02, b12]
    U[0, 11] = 1; V[2, 11] = 1; W[6, 11] = 1 # a00 b02 -> c02
    U[1, 12] = 1; V[5, 12] = 1; W[6, 12] = 1 # a01 b12 -> c02
    U[3, 13] = 1; V[2, 13] = 1; W[7, 13] = 1 # a10 b02 -> c12
    U[4, 14] = 1; V[5, 14] = 1; W[7, 14] = 1 # a11 b12 -> c12
    
    # w^T B_c (4 terms)
    # w^T = [a20, a21]
    U[6, 15] = 1; V[0, 15] = 1; W[9, 15] = 1 # a20 b00 -> c20
    U[7, 16] = 1; V[3, 16] = 1; W[9, 16] = 1 # a21 b10 -> c20
    U[6, 17] = 1; V[1, 17] = 1; W[10, 17] = 1 # a20 b01 -> c21
    U[7, 18] = 1; V[4, 18] = 1; W[10, 18] = 1 # a21 b11 -> c21
    
    # v t (2 terms)
    # t = b22
    U[2, 19] = 1; V[8, 19] = 1; W[6, 19] = 1 # a02 b22 -> c02
    U[5, 20] = 1; V[8, 20] = 1; W[7, 20] = 1 # a12 b22 -> c12
    
    # z s^T (2 terms)
    # z = a22
    U[8, 21] = 1; V[6, 21] = 1; W[9, 21] = 1 # a22 b20 -> c20
    U[8, 22] = 1; V[7, 22] = 1; W[10, 22] = 1 # a22 b21 -> c21
    
    # w^T u (2 terms)
    U[6, 23] = 1; V[2, 23] = 1; W[12, 23] = 1 # a20 b02 -> c22
    U[7, 24] = 1; V[5, 24] = 1; W[12, 24] = 1 # a21 b12 -> c22
    
    # z t (1 term)
    U[8, 25] = 1; V[8, 25] = 1; W[12, 25] = 1 # a22 b22 -> c22
    
    # 2. Extension for k=3,4,5,6 (Schoolbook)
    # For each k in 3..6, for each i in 0..2, for each j in 0..2:
    # Term: A_ij B_jk -> C_ik
    # U[i*3 + j, r] = 1
    # V[j*7 + k, r] = 1
    # W[k*3 + i, r] = 1
    
    r_idx = 26
    for k in range(3, 7):
        for i in range(3):
            for j in range(3):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, r_idx] = 1
                V[v_idx, r_idx] = 1
                W[w_idx, r_idx] = 1
                r_idx += 1
                
    return {"U": U, "V": V, "W": W}
