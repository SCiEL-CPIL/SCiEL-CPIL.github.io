import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 49
    rank = 49
    
    # Initialize U, V, W as zero matrices
    # U shape: (n*m, rank) = (16, 49)
    # V shape: (m*p, rank) = (16, 49)
    # W shape: (p*n, rank) = (16, 49)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Define the base 2x2 Strassen coefficients
    # alpha[k_b][I_b][J_b]
    alpha = [
        [[1, 0], [0, 1]],  # k=0: A00 + A11
        [[0, 0], [1, 1]],  # k=1: A10 + A11
        [[1, 0], [0, 0]],  # k=2: A00
        [[0, 0], [0, 1]],  # k=3: A11
        [[1, 1], [0, 0]],  # k=4: A00 + A01
        [[-1, 0], [1, 0]], # k=5: A10 - A00
        [[0, 1], [0, -1]]  # k=6: A01 - A11
    ]
    
    # beta[k_b][J_b][K_b]
    beta = [
        [[1, 0], [0, 1]],  # k=0: B00 + B11
        [[1, 0], [0, 0]],  # k=1: B00
        [[0, 1], [-1, 0]], # k=2: B01 - B11 (Wait, B01 is J=0,K=1? No. B is 2x2. J is row, K is col? 
                           # In matrix mult C=AB, B is (m x p). Here m=p=2.
                           # B_{j,k}. j is row, k is col.
                           # M3 = A00 * (B01 - B11).
                           # B01 is row 0, col 1. B11 is row 1, col 1.
                           # So beta[2][0][1] = 1, beta[2][1][1] = -1.
                           # My list above: beta[2] = [[0, 1], [-1, 0]]?
                           # Row 0: [0, 1] -> B00=0, B01=1. Correct.
                           # Row 1: [-1, 0] -> B10=-1, B11=0. Incorrect. Should be B11=-1.
                           # Let's re-verify M3.
                           # M3 = A00 (B01 - B11).
                           # So B_lin has B01=1, B11=-1.
                           # beta[2][0][1] = 1. beta[2][1][1] = -1.
                           # So beta[2] should be [[0, 1], [0, -1]].
    ]
    
    # Let's redefine beta carefully.
    # M1: B00 + B11 -> [[1, 0], [0, 1]]
    # M2: B00 -> [[1, 0], [0, 0]]
    # M3: B01 - B11 -> [[0, 1], [0, -1]]
    # M4: B10 - B00 -> [[-1, 0], [1, 0]]
    # M5: B11 -> [[0, 0], [0, 1]]
    # M6: B00 + B01 -> [[1, 1], [0, 0]]
    # M7: B10 + B11 -> [[0, 0], [1, 1]]
    
    beta = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]]
    ]
    
    # gamma[k_b][I_b][K_b]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    gamma = [
        [[1, 0], [0, 1]],  # k=0 (M1): C00=1, C11=1
        [[0, 0], [1, -1]], # k=1 (M2): C10=1, C11=-1
        [[0, 1], [0, 1]],  # k=2 (M3): C01=1, C11=1
        [[1, 0], [1, 0]],  # k=3 (M4): C00=1, C10=1
        [[-1, 1], [0, 0]], # k=4 (M5): C00=-1, C01=1
        [[0, 0], [0, 1]],  # k=5 (M6): C11=1
        [[1, 0], [0, 0]]   # k=6 (M7): C00=1
    ]
    # Wait, let's re-verify gamma indices.
    # gamma[k][I_b][K_b]
    # M1 (k=0): C00 (I=0,K=0) gets 1. C11 (I=1,K=1) gets 1.
    #   gamma[0][0][0] = 1. gamma[0][1][1] = 1.
    #   My list: gamma[0] = [[1, 0], [0, 1]]. Correct.
    # M2 (k=1): C10 (I=1,K=0) gets 1. C11 (I=1,K=1) gets -1.
    #   gamma[1][1][0] = 1. gamma[1][1][1] = -1.
    #   My list: gamma[1] = [[0, 0], [1, -1]]. Correct.
    # M3 (k=2): C01 (I=0,K=1) gets 1. C11 (I=1,K=1) gets 1.
    #   gamma[2][0][1] = 1. gamma[2][1][1] = 1.
    #   My list: gamma[2] = [[0, 1], [0, 1]]. Correct.
    # M4 (k=3): C00 (I=0,K=0) gets 1. C10 (I=1,K=0) gets 1.
    #   gamma[3][0][0] = 1. gamma[3][1][0] = 1.
    #   My list: gamma[3] = [[1, 0], [1, 0]]. Correct.
    # M5 (k=4): C00 (I=0,K=0) gets -1. C01 (I=0,K=1) gets 1.
    #   gamma[4][0][0] = -1. gamma[4][0][1] = 1.
    #   My list: gamma[4] = [[-1, 1], [0, 0]]. Correct.
    # M6 (k=5): C11 (I=1,K=1) gets 1.
    #   gamma[5][1][1] = 1.
    #   My list: gamma[5] = [[0, 0], [0, 1]]. Correct.
    # M7 (k=6): C00 (I=0,K=0) gets 1.
    #   gamma[6][0][0] = 1.
    #   My list: gamma[6] = [[1, 0], [0, 0]]. Correct.

    # Base 2x2 Strassen U, V, W
    # U2[i][j][r], V2[j][k][r], W2[k][i][r]
    U2 = np.zeros((2, 2, 7), dtype=int)
    V2 = np.zeros((2, 2, 7), dtype=int)
    W2 = np.zeros((2, 2, 7), dtype=int)
    
    # r=0 (M1)
    U2[0,0,0]=1; U2[1,1,0]=1
    V2[0,0,0]=1; V2[1,1,0]=1
    W2[0,0,0]=1; W2[1,1,0]=1
    
    # r=1 (M2)
    U2[1,0,1]=1; U2[1,1,1]=1
    V2[0,0,1]=1
    
    # r=2 (M3)
    U2[0,0,2]=1
    V2[0,1,2]=1; V2[1,1,2]=-1
    
    # r=3 (M4)
    U2[1,1,3]=1
    V2[1,0,3]=1; V2[0,0,3]=-1
    
    # r=4 (M5)
    U2[0,0,4]=1; U2[0,1,4]=1
    V2[1,1,4]=1
    
    # r=5 (M6)
    U2[1,0,5]=1; U2[0,0,5]=-1
    V2[0,0,5]=1; V2[0,1,5]=1
    
    # r=6 (M7)
    U2[0,1,6]=1; U2[1,1,6]=-1
    V2[1,0,6]=1; V2[1,1,6]=1
    
    # W2 from gamma logic for 2x2?
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    W2[0,0,0]=1; W2[0,0,3]=1; W2[0,0,4]=-1; W2[0,0,6]=1
    W2[1,0,2]=1; W2[1,0,4]=1
    W2[0,1,1]=1; W2[0,1,3]=1
    W2[1,1,0]=1; W2[1,1,1]=-1; W2[1,1,2]=1; W2[1,1,5]=1
    
    # Construct 4x4 U, V, W
    for I in range(n):
        for J in range(m):
            Ib = I // 2
            iloc = I % 2
            Jb = J // 2
            jloc = J % 2
            
            for R in range(rank):
                kb = R // 7
                rloc = R % 7
                
                u_val = alpha[kb][Ib][Jb] * U2[iloc, jloc, rloc]
                if u_val != 0:
                    U[I * m + J, R] = u_val
                    
    for J in range(m):
        for K in range(p):
            Jb = J // 2
            jloc = J % 2
            Kb = K // 2
            kloc = K % 2
            
            for R in range(rank):
                kb = R // 7
                rloc = R % 7
                
                v_val = beta[kb][Jb][Kb] * V2[jloc, kloc, rloc]
                if v_val != 0:
                    V[J * p + K, R] = v_val
                    
    for K in range(p):
        for I in range(n):
            Kb = K // 2
            kloc = K % 2
            Ib = I // 2
            iloc = I % 2
            
            for R in range(rank):
                kb = R // 7
                rloc = R % 7
                
                w_val = gamma[kb][Ib][Kb] * W2[kloc, iloc, rloc]
                if w_val != 0:
                    W[K * n + I, R] = w_val
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
