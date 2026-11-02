import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    num_blocks = p // 2  # 6 blocks of 2 columns each
    rank = num_blocks * 7  # 42
    
    U = [[0] * rank for _ in range(n * m)]  # shape (4, 42)
    V = [[0] * rank for _ in range(m * p)]  # shape (24, 42)
    W = [[0] * rank for _ in range(p * n)]  # shape (24, 42)
    
    # Strassen 7-product decomposition for 2x2x2:
    # A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=[[C00,C01],[C10,C11]]
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e)
    # M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(g+h)
    # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    #
    # Local A indices: 0=a(A[0,0]), 1=b(A[0,1]), 2=c(A[1,0]), 3=d(A[1,1])
    # Local B indices: 0=e(B[0,0]), 1=f(B[0,1]), 2=g(B[1,0]), 3=h(B[1,1])
    # Local C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # U coefficients for each Strassen term (A-side)
    U_strassen = [
        {0: 1, 3: 1},       # M1
        {2: 1, 3: 1},       # M2
        {0: 1},             # M3
        {3: 1},             # M4
        {0: 1, 1: 1},       # M5
        {0: -1, 2: 1},      # M6
        {1: 1, 3: -1},      # M7
    ]
    
    # V coefficients for each Strassen term (B-side, local indices)
    V_strassen = [
        {0: 1, 3: 1},       # M1
        {0: 1},             # M2
        {1: 1, 3: -1},      # M3
        {2: 1, 0: -1},      # M4
        {3: 1},             # M5
        {0: 1, 1: 1},       # M6
        {2: 1, 3: 1},       # M7
    ]
    
    # W coefficients for each Strassen term (C-side, local indices)
    W_strassen = [
        {0: 1, 3: 1},       # M1
        {2: 1, 3: -1},      # M2
        {1: 1, 3: 1},       # M3
        {0: 1, 2: 1},       # M4
        {0: -1, 1: 1},      # M5
        {3: 1},             # M6
        {0: 1},             # M7
    ]
    
    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            
            # Set U (A indices are global, no offset needed)
            for a_idx, val in U_strassen[s].items():
                U[a_idx][col] = val
            
            # Set V (map local B index to global)
            for local_b, val in V_strassen[s].items():
                if local_b == 0:
                    global_b = 0 * p + 2 * t        # B[0, 2t]
                elif local_b == 1:
                    global_b = 0 * p + 2 * t + 1    # B[0, 2t+1]
                elif local_b == 2:
                    global_b = 1 * p + 2 * t        # B[1, 2t]
                else:  # local_b == 3
                    global_b = 1 * p + 2 * t + 1    # B[1, 2t+1]
                V[global_b][col] = val
            
            # Set W (map local C index to global W index: W[k*n+i] where k=col, i=row)
            for local_c, val in W_strassen[s].items():
                if local_c == 0:
                    global_w = (2 * t) * n + 0       # C[0, 2t]
                elif local_c == 1:
                    global_w = (2 * t + 1) * n + 0   # C[0, 2t+1]
                elif local_c == 2:
                    global_w = (2 * t) * n + 1       # C[1, 2t]
                else:  # local_c == 3
                    global_w = (2 * t + 1) * n + 1   # C[1, 2t+1]
                W[global_w][col] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
