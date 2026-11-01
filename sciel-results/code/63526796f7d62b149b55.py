"""Tensor decomposition for 2x2 * 2x13 matrix multiplication."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 150s kill
    
    n, m, p = 2, 2, 13
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[(i*m+k), (k*p+j), (j*n+i)] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T[i*m+k, k*p+j, j*n+i] = 1.0
    
    # Strategy: Use Strassen's rank-7 decomposition for 2-column blocks
    # 6 blocks of 2 columns (12 cols) + 1 block of 1 column
    # Total rank: 6*7 + 4 = 46
    
    # Strassen decomposition for 2x2x2 sub-tensor
    U_sub = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1,-1],
        [1, 1, 0, 1, 0,  1, 0],
    ])
    V_sub = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 1],
        [0, 0,-1, 0, 0, 0, 0],
        [1, 0, 1,-1, 1, 0, 1],
    ])
    W_sub = np.array([
        [ 1, 0, 1,-1, 0, 0,-1],
        [ 0, 0, 0, 0, 1, 0, 1],
        [ 0, 1,-1, 1, 0, 0, 0],
        [ 1, 0, 0, 0, 1, 1, 0],
    ])
    
    # Verify Strassen decomposition
    S_computed = np.zeros((4, 4, 4))
    for t in range(7):
        S_computed += np.einsum('a,b,c->abc', U_sub[:,t], V_sub[:,t], W_sub[:,t])
    
    S_target = np.zeros((4, 4, 4))
    S_target[0,0,0] = 1; S_target[0,1,2] = 1
    S_target[1,2,0] = 1; S_target[1,3,2] = 1
    S_target[2,0,1] = 1; S_target[2,1,3] = 1
    S_target[3,2,1] = 1; S_target[3,3,3] = 1
    
    strassen_valid = np.allclose(S_computed, S_target, atol=1e-10)
    
    if not strassen_valid:
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k, col] = 1
                    V[k * p + j, col] = 1
                    W[j * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build full decomposition
    rank = 6 * 7 + 4
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for block in range(6):
        k0 = block * 2
        k1 = block * 2 + 1
        v_map = [k0, k1, m + k0, m + k1]
        w_map = [k0 * n, k0 * n + 1, k1 * n, k1 * n + 1]
        
        for t in range(7):
            for a in range(4):
                if U_sub[a, t] != 0:
                    U[a, col + t] = U_sub[a, t]
            for b in range(4):
                if V_sub[b, t] != 0:
                    V[v_map[b], col + t] = V_sub[b, t]
            for c in range(4):
                if W_sub[c, t] != 0:
                    W[w_map[c], col + t] = W_sub[c, t]
        col += 7
    
    # Last single column
    k_single = 12
    for t in range(4):
        i = t // 2
        k = t % 2
        U[i * m + k, col + t] = 1
        V[k * p + k_single, col + t] = 1
        W[k_single * n + i, col + t] = 1
    col += 4
    
    # Verify
    T_reconstructed = np.zeros((n*m, m*p, p*n))
    for t in range(rank):
        T_reconstructed += np.einsum('a,b,c->abc', U[:,t], V[:,t], W[:,t])
    
    if not np.allclose(T_reconstructed, T, atol=1e-10):
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k, c] = 1
                    V[k * p + j, c] = 1
                    W[j * n + i, c] = 1
                    c += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
