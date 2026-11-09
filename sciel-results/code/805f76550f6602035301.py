import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0

    # 1. Define 2x2x2 rank-7 decomposition (Strassen)
    # U, V, W shapes: (4, 7)
    U222 = np.zeros((4, 7))
    V222 = np.zeros((4, 7))
    W222 = np.zeros((4, 7))
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    # U: A00=1, A11=1
    U222[0, 0] = 1; U222[3, 0] = 1
    # V: B00=1, B11=1
    V222[0, 0] = 1; V222[3, 0] = 1
    # W: C00+=1, C11+=1
    # W index: k*2+i. C00 (i=0,k=0) -> 0. C11 (i=1,k=1) -> 3.
    W222[0, 0] = 1; W222[3, 0] = 1

    # Term 2: M2 = (A10+A11)B00
    U222[2, 1] = 1; U222[3, 1] = 1
    V222[0, 1] = 1
    # W: C10+=1, C11-=1
    # C10 (i=1,k=0) -> 2. C11 (i=1,k=1) -> 3.
    W222[2, 1] = 1; W222[3, 1] = -1

    # Term 3: M3 = A00(B01-B11)
    U222[0, 2] = 1
    V222[1, 2] = 1; V222[3, 2] = -1
    # W: C01+=1, C11+=1
    # C01 (i=0,k=1) -> 1. C11 (i=1,k=1) -> 3.
    W222[1, 2] = 1; W222[3, 2] = 1

    # Term 4: M4 = A11(B10-B00)
    U222[3, 3] = 1
    V222[2, 3] = 1; V222[0, 3] = -1
    # W: C00+=1, C10+=1
    # C00 (i=0,k=0) -> 0. C10 (i=1,k=0) -> 2.
    W222[0, 3] = 1; W222[2, 3] = 1

    # Term 5: M5 = (A00+A01)B11
    U222[0, 4] = 1; U222[1, 4] = 1
    V222[3, 4] = 1
    # W: C00-=1, C01+=1
    # C00 (i=0,k=0) -> 0. C01 (i=0,k=1) -> 1.
    W222[0, 4] = -1; W222[1, 4] = 1

    # Term 6: M6 = (A10-A00)(B00+B01)
    U222[2, 5] = 1; U222[0, 5] = -1
    V222[0, 5] = 1; V222[1, 5] = 1
    # W: C11+=1
    # C11 (i=1,k=1) -> 3.
    W222[3, 5] = 1

    # Term 7: M7 = (A01-A11)(B10+B11)
    U222[1, 6] = 1; U222[3, 6] = -1
    V222[2, 6] = 1; V222[3, 6] = 1
    # W: C00+=1
    # C00 (i=0,k=0) -> 0.
    W222[0, 6] = 1

    # 2. Define Strassen coefficients for 4x4
    # alpha[s][b], beta[s][b] for s in 0..6, b in 0..3
    # b: 0=A00, 1=A01, 2=A10, 3=A11
    alphas = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    betas = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    # gamma[s][i_block][k_block]
    # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    gammas = [
        [[1, 0], [0, 1]], # M1: C00(1), C11(1)
        [[0, 0], [1, -1]], # M2: C10(1), C11(-1)
        [[0, 1], [0, 1]], # M3: C01(1), C11(1)
        [[1, 0], [1, 0]], # M4: C00(1), C10(1)
        [[-1, 1], [0, 0]], # M5: C00(-1), C01(1)
        [[0, 0], [0, 1]], # M6: C11(1)
        [[1, 0], [0, 0]]  # M7: C00(1)
    ]

    # 3. Construct 4x4x4 decomposition (Rank 49)
    rank_444 = 49
    U444 = np.zeros((16, rank_444))
    V444 = np.zeros((16, rank_444))
    W444 = np.zeros((16, rank_444))

    for s in range(7):
        for r in range(7):
            c = s * 7 + r
            
            # U444
            for b in range(4):
                if alphas[s][b] != 0:
                    # Block b is at rows 4*b to 4*b+3? No.
                    # A is 4x4. Blocks are 2x2.
                    # Block (i_block, j_block) is at rows 2*i_block .. 2*i_block+1, cols 2*j_block .. 2*j_block+1.
                    # U index is i*m+j. Here n=4, m=4.
                    # Block b corresponds to (i_block, j_block).
                    # b=0: (0,0) -> i in {0,1}, j in {0,1}
                    # b=1: (0,1) -> i in {0,1}, j in {2,3}
                    # b=2: (1,0) -> i in {2,3}, j in {0,1}
                    # b=3: (1,1) -> i in {2,3}, j in {2,3}
                    
                    i_block = b // 2
                    j_block = b % 2
                    
                    for i_local in range(2):
                        for j_local in range(2):
                            i_global = 2 * i_block + i_local
                            j_global = 2 * j_block + j_local
                            u_idx = i_global * 4 + j_global
                            u2_idx = i_local * 2 + j_local
                            U444[u_idx, c] = alphas[s][b] * U222[u2_idx, r]
            
            # V444
            for b in range(4):
                if betas[s][b] != 0:
                    i_block = b // 2
                    j_block = b % 2
                    for i_local in range(2):
                        for j_local in range(2):
                            i_global = 2 * i_block + i_local
                            j_global = 2 * j_block + j_local
                            v_idx = i_global * 4 + j_global
                            v2_idx = i_local * 2 + j_local
                            V444[v_idx, c] = betas[s][b] * V222[v2_idx, r]
            
            # W444
            for i_block in range(2):
                for k_block in range(2):
                    g = gammas[s][i_block][k_block]
                    if g != 0:
                        for i_local in range(2):
                            for k_local in range(2):
                                i_global = 2 * i_block + i_local
                                k_global = 2 * k_block + k_local
                                w_idx = k_global * 4 + i_global
                                w2_idx = k_local * 2 + i_local
                                W444[w_idx, c] = g * W222[w2_idx, r]

    # 4. Construct 4x4x8 decomposition (Rank 98)
    rank_448 = 98
    U448 = np.zeros((16, rank_448))
    V448 = np.zeros((32, rank_448))
    W448 = np.zeros((32, rank_448))

    # First half (cols 0-3 of B)
    U448[:, 0:49] = U444
    V448[0:16, 0:49] = V444
    W448[0:16, 0:49] = W444

    # Second half (cols 4-7 of B)
    U448[:, 49:98] = U444
    V448[16:32, 49:98] = V444
    W448[16:32, 49:98] = W444

    return {"U": U448, "V": V448, "W": W448}
# EVOLVE-BLOCK-END
