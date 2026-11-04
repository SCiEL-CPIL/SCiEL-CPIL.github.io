import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    rank = 98
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 rank-7 decomposition
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Term 1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # Term 2
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1
    
    # Term 3
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # Term 4
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # Term 5
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # Term 6
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # Term 7
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1; W2[2][6] = 1
    
    # Build 4x4x8 decomposition
    for half in range(2):
        offset = half * 49
        
        for a in range(7):
            for a_prime in range(7):
                col = offset + a * 7 + a_prime
                
                # U is the same for both halves (columns are different)
                for i1 in range(2):
                    for i2 in range(2):
                        i = i1 * 2 + i2
                        u_val = U2[i1][a] * U2[i2][a_prime]
                        if u_val != 0:
                            U[i][col] = u_val
                            
                # V depends on half
                # For half 0, V rows 0-15. For half 1, V rows 16-31.
                v_base = half * 16
                for j1 in range(2):
                    for j2 in range(2):
                        j = j1 * 2 + j2
                        v_val = V2[j1][a] * V2[j2][a_prime]
                        if v_val != 0:
                            V[v_base + j][col] = v_val
                            
                # W depends on half
                # For half 0, W rows 0-15. For half 1, W rows 16-31.
                w_base = half * 16
                for k1 in range(2):
                    for k2 in range(2):
                        k = k1 * 2 + k2
                        w_val = W2[k1][a] * W2[k2][a_prime]
                        if w_val != 0:
                            W[w_base + k][col] = w_val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
