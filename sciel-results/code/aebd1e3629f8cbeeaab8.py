"""Tensor decomposition for 2x2 * 2x13 matrix multiplication."""
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = 2, 2, 13
    
    # Target tensor T
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T[i*m+k, k*p+j, j*n+i] = 1.0

    # Strassen decomposition for 2x2x2
    # U, V, W sub-matrices
    # Corrected W_sub for C22 (index 3, col 5 should be -1)
    U_sub = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A11
        [0, 0, 0, 0, 1,  0, 1],   # A12
        [0, 1, 0, 0, 0,  1,-1],   # A21
        [1, 1, 0, 1, 0,  1, 0],   # A22
    ])
    V_sub = np.array([
        [1, 1, 0, 0, 0, 1, 0],   # B11
        [0, 0, 0, 1, 0, 1, 1],   # B12
        [0, 0,-1, 0, 0, 0, 0],   # B21
        [1, 0, 1,-1, 1, 0, 1],   # B22
    ])
    W_sub = np.array([
        [ 1, 0, 1,-1, 0, 0,-1],  # C11
        [ 0, 0, 0, 0, 1, 0, 1],  # C12
        [ 0, 1,-1, 1, 0, 0, 0],  # C21
        [ 1, 0, 0, 0, 1,-1, 0],  # C22 (Corrected)
    ])
    
    # Verify Strassen sub-decomposition
    S_computed = np.zeros((4, 4, 4))
    for t in range(7):
        S_computed += np.einsum('a,b,c->abc', U_sub[:,t], V_sub[:,t], W_sub[:,t])
    
    S_target = np.zeros((4, 4, 4))
    S_target[0,0,0] = 1; S_target[0,1,2] = 1
    S_target[1,2,0] = 1; S_target[1,3,2] = 1
    S_target[2,0,1] = 1; S_target[2,1,3] = 1
    S_target[3,2,1] = 1; S_target[3,3,3] = 1
    
    if not np.allclose(S_computed, S_target, atol=1e-10):
        # Fallback to schoolbook if Strassen is broken
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        column = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k, column] = 1
                    V[k * p + j, column] = 1
                    W[j * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Build full decomposition
    # 6 blocks of 2 columns (Strassen, rank 7 each)
    # 1 block of 1 column (Schoolbook, rank 4)
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
    
    # Verify full decomposition
    T_reconstructed = np.zeros((n*m, m*p, p*n))
    for t in range(rank):
        T_reconstructed += np.einsum('a,b,c->abc', U[:,t], V[:,t], W[:,t])
    
    if not np.allclose(T_reconstructed, T, atol=1e-10):
        # Fallback
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        column = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k, column] = 1
                    V[k * p + j, column] = 1
                    W[j * n + i, column] = 1
                    column += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
