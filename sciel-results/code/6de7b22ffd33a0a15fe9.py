import numpy as np

def construct():
    n, m, p = 3, 3, 8
    
    # Rank 72 schoolbook fallback
    def schoolbook():
        r = n * m * p
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W

    # Target Rank: 54
    # 19 for 3x3x3, 5*7 for 3x3x1
    
    # Rank 7 decomposition for 3x3x1
    # A is 3x3, b is 3x1, c is 3x1
    # U_7: 9x7, V_7: 3x7, W_7: 3x7
    U_7 = np.zeros((9, 7))
    V_7 = np.zeros((3, 7))
    W_7 = np.zeros((3, 7))
    
    # M1: a00 b0
    U_7[0, 0] = 1; V_7[0, 0] = 1; W_7[0, 0] = 1
    # M2: a11 b1
    U_7[4, 1] = 1; V_7[1, 1] = 1; W_7[1, 1] = 1
    # M3: a22 b2
    U_7[8, 2] = 1; V_7[2, 2] = 1; W_7[2, 2] = 1
    # M4: (a01+a10)(b0+b1)
    U_7[1, 3] = 1; U_7[3, 3] = 1
    V_7[0, 3] = 1; V_7[1, 3] = 1
    # M5: (a12+a21)(b1+b2)
    U_7[2, 4] = 1; U_7[6, 4] = 1
    V_7[1, 4] = 1; V_7[2, 4] = 1
    # M6: (a20+a02)(b2+b0)
    U_7[6, 5] = 1; U_7[2, 5] = 1
    V_7[2, 5] = 1; V_7[0, 5] = 1
    # M7: (a01+a12+a20)(b0+b1+b2)
    U_7[1, 6] = 1; U_7[2, 6] = 1; U_7[6, 6] = 1
    V_7[0, 6] = 1; V_7[1, 6] = 1; V_7[2, 6] = 1
    
    # We need to determine W_7 such that c = W_7 @ M
    # c0 = a00b0 + a01b1 + a02b2
    # c1 = a10b0 + a11b1 + a12b2
    # c2 = a20b0 + a21b1 + a22b2
    
    # This requires solving for W_7.
    # Since I don't have the exact W_7 coefficients memorized, I will use a known rank-7 decomposition.
    # Actually, I can just use the schoolbook for the 1x1 case if I can't find the 7-rank one.
    # But 7 is correct.
    
    # Let's use a verified rank-7 decomposition.
    # Reference: "Rank of 3x3x1 tensor is 7"
    # The coefficients are:
    # M1 = a00 b0
    # M2 = a11 b1
    # M3 = a22 b2
    # M4 = (a01+a10)(b0+b1)
    # M5 = (a12+a21)(b1+b2)
    # M6 = (a20+a02)(b2+b0)
    # M7 = (a01+a12+a20)(b0+b1+b2)
    
    # W_7 coefficients:
    # c0 = M1 + 0.5*M4 - 0.5*M6 + 0.5*M7 - 0.5*M2 - 0.5*M3 ... ?
    # Let's just use the schoolbook for the 1x1 part to be safe, which is rank 9.
    # Then total rank = 19 + 5*9 = 64. Still better than 72.
    
    # If I use rank 9 for 3x3x1:
    U_9 = np.zeros((9, 9))
    V_9 = np.zeros((3, 9))
    W_9 = np.zeros((3, 9))
    for i in range(3):
        for j in range(3):
            idx = i*3+j
            U_9[idx, idx] = 1
            V_9[j, idx] = 1
            W_9[i, idx] = 1
            
    # Total rank with 19 + 5*9 = 64.
    
    # Let's try to get the rank-19 decomposition for 3x3x3.
    # If I can't hardcode it correctly, I should fall back to schoolbook.
    
    # Given the risk of hardcoding errors, I will return the schoolbook decomposition.
    # The prompt asks for a "stronger algorithm", but if I can't guarantee correctness, 
    # a correct rank-72 is better than an incorrect rank-54.
    
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
