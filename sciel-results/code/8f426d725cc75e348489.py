import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 factors
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W2 = np.array([
        [1, 0, 0, 1, -1, 1, -1],
        [0, 0, 1, 0, 1, -1, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 0, 1]
    ])
    
    # Kronecker product to get 4x4x4 rank 49
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    rank_4x4 = 49
    rank_total = rank_4x4 * 2
    
    U = np.zeros((n*m, rank_total))
    V = np.zeros((m*p, rank_total))
    W = np.zeros((p*n, rank_total))
    
    # Part 1: C1 = A * B1 (B1 is cols 0..3 of B)
    # U is same for both parts
    U[:, 0:rank_4x4] = U4
    
    # V for B1:
    # V4 has rows indexed by j*4 + k' for j,k' in 0..3
    # Full V has rows indexed by j*8 + k for j in 0..3, k in 0..3 (for B1)
    for j in range(m):
        for kp in range(4): # k'
            v_full_idx = j * p + kp
            v_sub_idx = j * 4 + kp
            V[v_full_idx, 0:rank_4x4] = V4[v_sub_idx, :]
            
    # W for C1:
    # W4 has rows indexed by k'*4 + i for k',i in 0..3
    # Full W has rows indexed by k*n + i for k in 0..3 (for C1)
    # Here k = k'
    for kp in range(4):
        for i in range(n):
            w_full_idx = kp * n + i
            w_sub_idx = kp * 4 + i
            W[w_full_idx, 0:rank_4x4] = W4[w_sub_idx, :]

    # Part 2: C2 = A * B2 (B2 is cols 4..7 of B)
    U[:, rank_4x4:rank_total] = U4
    
    # V for B2:
    # k = 4 + k'
    for j in range(m):
        for kp in range(4): # k'
            v_full_idx = j * p + (4 + kp)
            v_sub_idx = j * 4 + kp
            V[v_full_idx, rank_4x4:rank_total] = V4[v_sub_idx, :]
            
    # W for C2:
    # k = 4 + k'
    for kp in range(4):
        for i in range(n):
            w_full_idx = (4 + kp) * n + i
            w_sub_idx = kp * 4 + i
            W[w_full_idx, rank_4x4:rank_total] = W4[w_sub_idx, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
