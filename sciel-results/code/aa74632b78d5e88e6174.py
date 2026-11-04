import numpy as np

def construct():
    n = 6
    # Rank 189 solution: Block Strassen with 3x3 blocks
    # 7 Strassen terms, each is a 3x3 multiplication (rank 27)
    # Total rank = 7 * 27 = 189
    
    rank = 189
    U = np.zeros((36, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    # Local 3x3 trivial multiplication decomposition
    # Indices: 0..8 for 3x3 matrix
    # Term t corresponds to A[i,k] * B[k,j] -> C[i,j]
    # t = i*9 + k*3 + j
    U_loc = np.zeros((9, 27))
    V_loc = np.zeros((9, 27))
    W_loc = np.zeros((9, 27))
    
    for i in range(3):
        for k in range(3):
            for j in range(3):
                t = i * 9 + k * 3 + j
                U_loc[i * 3 + k, t] = 1
                V_loc[k * 3 + j, t] = 1
                W_loc[i * 3 + j, t] = 1
                
    # Block indices:
    # 0: 11, 1: 12, 2: 21, 3: 22
    # Global index for block (b) and local (l) is b*9 + l
    
    # Strassen Coefficients
    # alpha[m][block]
    # beta[m][block]
    # gamma[m][block]
    
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B12-B22)
    # M4: A22(B21-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    alphas = [
        {0: 1, 3: 1},       # M1
        {2: 1, 3: 1},       # M2
        {0: 1},             # M3
        {3: 1},             # M4
        {0: 1, 1: 1},       # M5
        {2: 1, 0: -1},      # M6
        {1: 1, 3: -1}       # M7
    ]
    
    betas = [
        {0: 1, 3: 1},       # M1
        {0: 1},             # M2
        {1: 1, 3: -1},      # M3
        {2: 1, 0: -1},      # M4
        {3: 1},             # M5
        {0: 1, 1: 1},       # M6
        {2: 1, 3: 1}        # M7
    ]
    
    gammas = [
        {0: 1, 3: 1},       # M1 -> C11, C22
        {2: 1, 3: -1},      # M2 -> C21, C22
        {1: 1, 3: 1},       # M3 -> C12, C22
        {0: 1, 2: 1},       # M4 -> C11, C21
        {0: -1, 1: 1},      # M5 -> C11, C12
        {3: 1},             # M6 -> C22
        {0: 1}              # M7 -> C11
    ]
    
    for m in range(7):
        col_start = m * 27
        for t in range(27):
            col = col_start + t
            # U
            for b, alpha in alphas[m].items():
                for l in range(9):
                    if U_loc[l, t] != 0:
                        U[b * 9 + l, col] += alpha * U_loc[l, t]
            # V
            for b, beta in betas[m].items():
                for l in range(9):
                    if V_loc[l, t] != 0:
                        V[b * 9 + l, col] += beta * V_loc[l, t]
            # W
            for b, gamma in gammas[m].items():
                for l in range(9):
                    if W_loc[l, t] != 0:
                        W[b * 9 + l, col] += gamma * W_loc[l, t]
                        
    return {'U': U, 'V': V, 'W': W}
