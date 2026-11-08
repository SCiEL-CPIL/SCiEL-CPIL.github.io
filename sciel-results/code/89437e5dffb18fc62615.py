import time
import sys

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2
    # alpha[t][i][j]
    # beta[t][j][k]
    # gamma[t][i][k]
    
    # t=0: M1 = (A00+A11)(B00+B11)
    # t=1: M2 = (A10+A11)B00
    # t=2: M3 = A00(B01-B11)
    # t=3: M4 = A11(B10-B00)
    # t=4: M5 = (A00+A01)B11
    # t=5: M6 = (A10-A00)(B00+B01)
    # t=6: M7 = (A01-A11)(B10+B11)
    
    alpha = [
        [[1, 0], [1, 0]], # t=0: A00, A11 (Wait, A11 is i=1,j=1)
        # Correction:
        # M1: A00 + A11. i=0,j=0 and i=1,j=1.
        # My previous alpha definition:
        # alpha[0][0][0]=1, alpha[0][1][1]=1.
        # Let's re-verify indices.
        # A is 2x2. Indices 0,1.
        # M1 uses A[0,0] and A[1,1].
        # So alpha[0][0][0]=1, alpha[0][1][1]=1.
        # M2 uses A[1,0] and A[1,1].
        # alpha[1][1][0]=1, alpha[1][1][1]=1.
        # M3 uses A[0,0].
        # alpha[2][0][0]=1.
        # M4 uses A[1,1].
        # alpha[3][1][1]=1.
        # M5 uses A[0,0] and A[0,1].
        # alpha[4][0][0]=1, alpha[4][0][1]=1.
        # M6 uses A[1,0] and A[0,0].
        # alpha[5][1][0]=1, alpha[5][0][0]=-1.
        # M7 uses A[0,1] and A[1,1].
        # alpha[6][0][1]=1, alpha[6][1][1]=-1.
        
        [[1, 0], [0, 1]], # t=0
        [[0, 0], [1, 1]], # t=1
        [[1, 0], [0, 0]], # t=2
        [[0, 0], [0, 1]], # t=3
        [[1, 1], [0, 0]], # t=4
        [[-1, 0], [1, 0]], # t=5
        [[0, 1], [0, -1]]  # t=6
    ]
    
    beta = [
        [[1, 0], [0, 1]], # t=0: B00, B11
        [[1, 0], [0, 0]], # t=1: B00
        [[0, 1], [0, -1]], # t=2: B01, B11
        [[-1, 0], [1, 0]], # t=3: B10, B00
        [[0, 0], [0, 1]], # t=4: B11
        [[1, 1], [0, 0]], # t=5: B00, B01
        [[0, 1], [0, 1]]  # t=6: B10, B11
    ]
    
    gamma = [
        [[1, 0], [0, 1]], # t=0: C00, C11
        [[0, 0], [1, 0]], # t=1: C10
        [[0, 1], [1, 0]], # t=2: C01, C11
        [[1, 0], [0, 1]], # t=3: C00, C11
        [[-1, 1], [0, 0]], # t=4: C00, C01
        [[0, 0], [1, 1]], # t=5: C10, C11
        [[1, 0], [0, 1]]  # t=6: C00, C11
    ]
    
    # Check gamma values
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # t=0 (M1): C00+=1, C11+=1. gamma[0][0][0]=1, gamma[0][1][1]=1.
    # t=1 (M2): C10+=1. gamma[1][1][0]=1.
    # t=2 (M3): C01+=1, C11+=1. gamma[2][0][1]=1, gamma[2][1][1]=1.
    # t=3 (M4): C00+=1, C10+=1. gamma[3][0][0]=1, gamma[3][1][0]=1.
    # t=4 (M5): C00+=-1, C01+=1. gamma[4][0][0]=-1, gamma[4][0][1]=1.
    # t=5 (M6): C10+=1, C11+=1. gamma[5][1][0]=1, gamma[5][1][1]=1.
    # t=6 (M7): C00+=1, C11+=1. gamma[6][0][0]=1, gamma[6][1][1]=1.
    
    # Let's re-verify gamma definition in code above
    # gamma[0] = [[1,0],[0,1]] -> C00=1, C11=1. Correct.
    # gamma[1] = [[0,0],[1,0]] -> C10=1. Correct.
    # gamma[2] = [[0,1],[1,0]] -> C01=1, C11=1. Correct.
    # gamma[3] = [[1,0],[0,1]] -> C00=1, C11=1. 
    # Wait, M4 contributes to C00 and C10.
    # So gamma[3][0][0]=1, gamma[3][1][0]=1.
    # My code has gamma[3] = [[1,0],[0,1]]. This is C00=1, C11=1. Incorrect.
    
    # Correct gamma:
    gamma = [
        [[1, 0], [0, 1]], # t=0
        [[0, 0], [1, 0]], # t=1
        [[0, 1], [1, 0]], # t=2
        [[1, 0], [1, 0]], # t=3: C00=1, C10=1
        [[-1, 1], [0, 0]], # t=4: C00=-1, C01=1
        [[0, 0], [1, 1]], # t=5: C10=1, C11=1
        [[1, 0], [0, 1]]  # t=6: C00=1, C11=1
    ]
    
    # Fill U, V, W
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            for I in range(2):
                for J in range(2):
                    for ip in range(2):
                        for jp in range(2):
                            u_val = alpha[t][ip][jp] * alpha[s][I][J]
                            if u_val != 0:
                                row_u = (2 * I + ip) * 4 + (2 * J + jp)
                                U[row_u][r] += u_val
                            
                            v_val = beta[t][jp][kp] if False else 0 # placeholder
                            
            for Jp in range(2):
                for K in range(2):
                    for jp in range(2):
                        for kp in range(2):
                            v_val = beta[t][jp][kp] * beta[s][Jp][K]
                            if v_val != 0:
                                row_v = (2 * Jp + jp) * 4 + (2 * K + kp)
                                V[row_v][r] += v_val
                                
            for I in range(2):
                for K in range(2):
                    for ip in range(2):
                        for kp in range(2):
                            w_val = gamma[t][ip][kp] * gamma[s][I][K]
                            if w_val != 0:
                                row_w = (2 * K + kp) * 4 + (2 * I + ip)
                                W[row_w][r] += w_val
                                
    return {"U": U, "V": V, "W": W}
