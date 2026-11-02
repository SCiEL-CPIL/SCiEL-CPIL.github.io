import numpy as np

DIMENSIONS = [3, 4, 7]

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen)."""
    # Strassen's algorithm for 2x2 matrices
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # 7 multiplications:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # U: 4 entries (a,b,c,d), V: 4 entries (e,f,g,h), W: 4 entries (C11,C12,C21,C22)
    # Index: U[i*2+j], V[k*2+l], W[p_idx*2+q_idx] where output C[p_idx, q_idx]
    # Actually for 2x2x2: n=2, m=2, p=2
    # U shape: (2*2, 7) = (4, 7)
    # V shape: (2*2, 7) = (4, 7)  
    # W shape: (2*2, 7) = (4, 7)
    
    # A indices: a=U[0], b=U[1], c=U[2], d=U[3]
    # B indices: e=V[0], f=V[1], g=V[2], h=V[3]
    # C indices: C11=W[0], C12=W[1], C21=W[2], C22=W[3]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # m1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # contributes to C11 and C22
    
    # m2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1  # C21 += m2, C22 -= m2
    
    # m3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1  # C12 += m3, C22 += m3
    
    # m4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1  # C11 += m4, C21 += m4
    
    # m5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1  # C11 -= m5, C12 += m5
    
    # m6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C22 += m6
    
    # m7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C11 += m7
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Use schoolbook for simplicity - rank 84
    # TODO: implement slice-and-dice with lower rank
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
