import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=6
    # We split B (2x6) into three 2x2 blocks.
    # Each block multiplication is 2x2x2.
    # Rank of 2x2x2 is 7 (Strassen).
    # Total rank = 3 * 7 = 21.
    
    rank = 21
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2x2
    S_A = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [-1, 0, 1, 0],# P6
        [0, 1, 0, -1] # P7
    ]
    S_B = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1],# P3
        [-1, 0, 1, 0],# P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    S_C = []
    # P1
    s = np.zeros((2, 2)); s[0,0] += 1; s[1,1] += 1; S_C.append(s)
    # P2
    s = np.zeros((2, 2)); s[1,0] += 1; s[1,1] -= 1; S_C.append(s)
    # P3
    s = np.zeros((2, 2)); s[0,1] += 1; s[1,1] += 1; S_C.append(s)
    # P4
    s = np.zeros((2, 2)); s[0,0] += 1; s[1,0] += 1; S_C.append(s)
    # P5
    s = np.zeros((2, 2)); s[0,0] -= 1; s[0,1] += 1; S_C.append(s)
    # P6
    s = np.zeros((2, 2)); s[1,1] += 1; S_C.append(s)
    # P7
    s = np.zeros((2, 2)); s[0,0] += 1; S_C.append(s)
    
    for t in range(3):
        for l in range(7):
            k_global = t * 7 + l
            
            # Update U
            for i in range(n):
                for j in range(m):
                    U[i * m + j, k_global] = S_A[l][i * m + j]
            
            # Update V
            for j in range(m):
                for k_local in range(2):
                    k_global_b = t * 2 + k_local
                    V[j * p + k_global_b, k_global] = S_B[l][j * 2 + k_local]
            
            # Update W
            for i in range(n):
                for k_local in range(2):
                    k_global_c = t * 2 + k_local
                    W[k_global_c * n + i, k_global] = S_C[l][i][k_local]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
