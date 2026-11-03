import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Total rank will be 23
    # 7 from Strassen for 2x2x2 block (i in 0,1; j in 0,1; k in 0,1)
    # 16 from schoolbook for the rest
    
    rank = 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 tensor
    def add_term(u_idx, v_idx, w_idx, coeff, col):
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff

    col = 0
    
    # Strassen part for i in {0,1}, j in {0,1}, k in {0,1}
    # Indices:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # M1 = (A00 + A11)(B00 + B11)
    # U: (0,0)=1, (1,1)=1
    # V: (0,0)=1, (1,1)=1
    # W: (0,0)=1, (0,1)=1, (1,0)=1, (1,1)=1
    # Wait, W is output.
    # C00 += M1, C01 += M1, C10 += M1, C11 += M1?
    # No, look at the output formulas.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for M1:
    # W contributions:
    # W[0*n+0] (C00) += 1
    # W[1*n+1] (C11) += 1
    # Wait, C01 and C10 don't get M1?
    # Let's re-verify Strassen output.
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    # So M1 contributes to C00 and C11.
    
    # Term 1: M1
    U[0*m+0][col] = 1
    U[1*m+1][col] = 1
    V[0*p+0][col] = 1
    V[1*p+1][col] = 1
    W[0*n+0][col] = 1
    W[1*n+1][col] = 1
    col += 1
    
    # Term 2: M2 = (A10 + A11) B00
    U[1*m+0][col] = 1
    U[1*m+1][col] = 1
    V[0*p+0][col] = 1
    W[1*n+0][col] = 1
    W[1*n+1][col] = -1 # C11 gets -M2
    col += 1
    
    # Term 3: M3 = A00 (B01 - B11)
    U[0*m+0][col] = 1
    V[0*p+1][col] = 1
    V[1*p+1][col] = -1
    W[0*n+1][col] = 1
    W[1*n+1][col] = 1
    col += 1
    
    # Term 4: M4 = A11 (B10 - B00)
    U[1*m+1][col] = 1
    V[1*p+0][col] = 1
    V[0*p+0][col] = -1
    W[0*n+0][col] = 1
    W[1*n+0][col] = 1
    col += 1
    
    # Term 5: M5 = (A00 + A01) B11
    U[0*m+0][col] = 1
    U[0*m+1][col] = 1
    V[1*p+1][col] = 1
    W[0*n+0][col] = -1
    W[0*n+1][col] = 1
    col += 1
    
    # Term 6: M6 = (A10 - A00) (B00 + B01)
    U[1*m+0][col] = 1
    U[0*m+0][col] = -1
    V[0*p+0][col] = 1
    V[0*p+1][col] = 1
    W[1*n+1][col] = 1
    col += 1
    
    # Term 7: M7 = (A01 - A11) (B10 + B11)
    U[0*m+1][col] = 1
    U[1*m+1][col] = -1
    V[1*p+0][col] = 1
    V[1*p+1][col] = 1
    W[0*n+0][col] = 1
    col += 1
    
    # Schoolbook part for the remaining 16 terms
    # j in {0,1,2}, k in {0,1,2,3}
    # Exclude j in {0,1} AND k in {0,1}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (j in [0, 1]) and (k in [0, 1]):
                    continue
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
