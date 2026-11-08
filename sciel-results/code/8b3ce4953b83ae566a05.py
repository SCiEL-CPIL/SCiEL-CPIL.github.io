import numpy as np

def construct():
    n, m, p = 4, 4, 5
    rank_total = 65  # 49 + 16
    
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # --- Part 1: 4x4x4 Strassen (Rank 49) ---
    # 2x2x2 Strassen factors
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 1: (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # Term 2: (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # Term 3: A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # Term 4: A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # Term 5: (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # Term 6: (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # Term 7: (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Top-level 4x4x4 Strassen coefficients (2x2 block coefficients)
    # L[s][i][k], R[s][k][j], W[s][i][j]
    L = [
        [[1, 0], [0, 1]], # s=0
        [[0, 0], [1, 1]], # s=1
        [[1, 0], [0, 0]], # s=2
        [[0, 0], [0, 1]], # s=3
        [[1, 1], [0, 0]], # s=4
        [[-1, 0], [1, 0]],# s=5
        [[0, 1], [0, -1]] # s=6
    ]
    R = [
        [[1, 0], [0, 1]], # s=0
        [[1, 0], [0, 0]], # s=1
        [[0, 1], [0, -1]],# s=2
        [[0, -1], [1, 0]],# s=3
        [[0, 0], [0, 1]], # s=4
        [[1, 1], [0, 0]], # s=5
        [[0, 1], [1, 1]]  # s=6
    ]
    Ws = [
        [[1, 0], [0, 1]], # s=0
        [[0, 0], [1, -1]],# s=1
        [[0, 1], [0, 1]], # s=2
        [[1, 0], [1, 0]], # s=3
        [[-1, 1], [0, 0]],# s=4
        [[0, 0], [0, 1]], # s=5
        [[1, 0], [0, 0]]  # s=6
    ]
    
    col = 0
    for s in range(7):
        for t in range(7):
            # Fill U, V, W for this term
            for I in range(4):
                for J in range(4):
                    i = I // 2
                    p = I % 2
                    k = J // 2
                    v = J % 2
                    U[I * 4 + J, col] = L[s][i][k] * U2[p * 2 + v, t]
                    
                    for K in range(4):
                        j = K // 2
                        q = K % 2
                        V[J * 4 + K, col] = R[s][k][j] * V2[v * 2 + q, t]
                        W[K * 4 + I, col] = Ws[s][i][j] * W2[q * 2 + p, t]
            col += 1
            
    # --- Part 2: 4x4x1 Schoolbook (Rank 16) ---
    # A (4x4) * B2 (4x1) -> C2 (4x1)
    # B2 is column 4 of B. C2 is column 4 of C.
    for r in range(16):
        i = r // 4
        j = r % 4
        U[i * 4 + j, col] = 1
        V[j * 5 + 4, col] = 1
        W[4 * 4 + i, col] = 1
        col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
