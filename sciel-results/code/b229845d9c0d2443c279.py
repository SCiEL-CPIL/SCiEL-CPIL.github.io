import time
import sys

# EVOLVE-BLOCK-START

def get_strassen_2x2_2x2_UVW():
    """
    Returns U, V, W for 2x2x2 Strassen.
    U: 4x7, V: 4x7, W: 4x7
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen steps
    # Indices: 0=A00, 1=A01, 2=A10, 3=A11
    #           0=B00, 1=B01, 2=B10, 3=B11
    #           0=C00, 1=C01, 2=C10, 3=C11
    
    # M1 = (A00+A11)(B00+B11)
    # A part: A00, A11 -> 1 at 0, 3
    # B part: B00, B11 -> 1 at 0, 3
    # C part: C00 (+1), C11 (+1)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    # A part: A10, A11 -> 1 at 2, 3
    # B part: B00 -> 1 at 0
    # C part: C10 (+1)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = A00(B01-B11)
    # A part: A00 -> 1 at 0
    # B part: B01, B11 -> 1 at 1, -1 at 3
    # C part: C01 (+1)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4 = A11(B10-B00)
    # A part: A11 -> 1 at 3
    # B part: B10, B00 -> 1 at 2, -1 at 0
    # C part: C00 (+1), C10 (+1)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    # A part: A00, A01 -> 1 at 0, 1
    # B part: B11 -> 1 at 3
    # C part: C01 (+1), C11 (-1)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[3][4] = -1
    
    # M6 = (A10-A00)(B00+B01)
    # A part: A10, A00 -> 1 at 2, -1 at 0
    # B part: B00, B01 -> 1 at 0, 1
    # C part: C11 (+1)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    # A part: A01, A11 -> 1 at 1, -1 at 3
    # B part: B10, B11 -> 1 at 2, 1
    # C part: C00 (+1), C11 (-1)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = -1
    
    return U, V, W

def construct():
    n, m, p = 4, 4, 7
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][t] = 1
                V[v_idx][t] = 1
                W[w_idx][t] = 1
                t += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
