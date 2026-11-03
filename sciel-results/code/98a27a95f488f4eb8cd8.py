import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Build 4x4 Strassen decomposition (Rank 49)
    # U2, V2, W2 for 2x2 Strassen
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # P1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # P2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # P3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # P4
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # P5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[1, 4] = 1; W2[0, 4] = -1
    
    # P6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # P7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Strassen coefficients for 2x2 block multiplication
    # c[r, I, J], d[r, J, K], e[r, I, K]
    # r in 0..6
    c = np.zeros((7, 2, 2))
    d = np.zeros((7, 2, 2))
    e = np.zeros((7, 2, 2))
    
    # r=0 (P1)
    c[0, 0, 0] = 1; c[0, 1, 1] = 1
    d[0, 0, 0] = 1; d[0, 1, 1] = 1
    e[0, 0, 0] = 1; e[0, 1, 1] = 1
    
    # r=1 (P2)
    c[1, 1, 0] = 1; c[1, 1, 1] = 1
    d[1, 0, 0] = 1
    e[1, 1, 0] = 1; e[1, 1, 1] = -1
    
    # r=2 (P3)
    c[2, 0, 0] = 1
    d[2, 0, 1] = 1; d[2, 1, 1] = -1
    e[2, 0, 1] = 1; e[2, 1, 1] = 1
    
    # r=3 (P4)
    c[3, 1, 1] = 1
    d[3, 1, 0] = 1; d[3, 0, 0] = -1
    e[3, 0, 0] = 1; e[3, 1, 0] = 1
    
    # r=4 (P5)
    c[4, 0, 0] = 1; c[4, 0, 1] = 1
    d[4, 1, 1] = 1
    e[4, 0, 1] = 1; e[4, 0, 0] = -1
    
    # r=5 (P6)
    c[5, 1, 0] = 1; c[5, 0, 0] = -1
    d[5, 0, 0] = 1; d[5, 0, 1] = 1
    e[5, 1, 1] = 1
    
    # r=6 (P7)
    c[6, 0, 1] = 1; c[6, 1, 1] = -1
    d[6, 1, 0] = 1; d[6, 1, 1] = 1
    e[6, 0, 0] = 1
    
    rank_4x4 = 49
    U = np.zeros((25, rank_4x4 + 45))
    V = np.zeros((25, rank_4x4 + 45))
    W = np.zeros((25, rank_4x4 + 45))
    
    # Fill 4x4 part
    for r in range(7):
        for s in range(7):
            R = r * 7 + s
            # U
            for i in range(2):
                for j in range(2):
                    val = U2[i * 2 + j, s]
                    if val != 0:
                        for I in range(2):
                            for J in range(2):
                                coeff = c[r, I, J]
                                if coeff != 0:
                                    idx_A = (I * 2 + i) * 4 + (J * 2 + j)
                                    U[idx_A, R] += val * coeff
            # V
            for j in range(2):
                for k in range(2):
                    val = V2[j * 2 + k, s]
                    if val != 0:
                        for J in range(2):
                            for K in range(2):
                                coeff = d[r, J, K]
                                if coeff != 0:
                                    idx_B = (J * 2 + j) * 4 + (K * 2 + k)
                                    V[idx_B, R] += val * coeff
            # W
            for k in range(2):
                for i in range(2):
                    val = W2[k * 2 + i, s]
                    if val != 0:
                        for I in range(2):
                            for K in range(2):
                                coeff = e[r, I, K]
                                if coeff != 0:
                                    idx_C = (K * 2 + k) * 4 + (I * 2 + i)
                                    W[idx_C, R] += val * coeff
    
    # Add 5x5 cross terms
    rank_offset = 49
    current_rank = rank_offset
    
    # 1. C[i, 4] = sum_k A[i, k] B[k, 4] + A[i, 4] B[4, 4] for i in 0..3
    for i in range(4):
        for k in range(4):
            U[i * 5 + k, current_rank] = 1
            V[k * 5 + 4, current_rank] = 1
            W[4 * 5 + i, current_rank] = 1
            current_rank += 1
        U[i * 5 + 4, current_rank] = 1
        V[4 * 5 + 4, current_rank] = 1
        W[4 * 5 + i, current_rank] = 1
        current_rank += 1
        
    # 2. C[4, j] = sum_k A[4, k] B[k, j] + A[4, 4] B[4, j] for j in 0..3
    for j in range(4):
        for k in range(4):
            U[4 * 5 + k, current_rank] = 1
            V[k * 5 + j, current_rank] = 1
            W[j * 5 + 4, current_rank] = 1
            current_rank += 1
        U[4 * 5 + 4, current_rank] = 1
        V[4 * 5 + j, current_rank] = 1
        W[j * 5 + 4, current_rank] = 1
        current_rank += 1
        
    # 3. C[4, 4] = sum_k A[4, k] B[k, 4] + A[4, 4] B[4, 4]
    for k in range(4):
        U[4 * 5 + k, current_rank] = 1
        V[k * 5 + 4, current_rank] = 1
        W[4 * 5 + 4, current_rank] = 1
        current_rank += 1
    U[4 * 5 + 4, current_rank] = 1
    V[4 * 5 + 4, current_rank] = 1
    W[4 * 5 + 4, current_rank] = 1
    current_rank += 1
    
    # Trim to actual rank
    U = U[:, :current_rank]
    V = V[:, :current_rank]
    W = W[:, :current_rank]
    
    # Convert to lists as required by interface? 
    # "return {'U', 'V', 'W'}: factor matrices... with integer/half-integer entries"
    # Usually numpy arrays are accepted, but let's check the seed program.
    # Seed program returns lists of lists.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
