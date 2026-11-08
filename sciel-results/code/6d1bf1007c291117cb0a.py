import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def _verify_fast(U, V, W, n=2, m=4, p=4):
    """Faster verification using numpy."""
    n, m, p = 2, 4, 4
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
    
    return np.allclose(T, np.ones((n,m,p)), atol=1e-9)

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Attempt Rank 28 construction using Block Strassen
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # U (A) coefficients for M1..M7
    S_U = [
        [1, 0, 1, 0, 1, 0, 0], # a00
        [0, 0, 0, 0, 1, 0, 1], # a01
        [0, 1, 0, 0, 0, 1, 0], # a10
        [1, 1, 0, 1, 0, -1, -1] # a11
    ]
    
    # V (B) coefficients for M1..M7
    S_V = [
        [1, 1, 0, 0, 0, 1, 0], # b00
        [0, 0, 1, 0, 0, 1, 0], # b01
        [0, 0, 0, 1, 0, 0, 1], # b10
        [1, 0, -1, -1, 1, 0, 1] # b11
    ]
    
    # W (C) coefficients for M1..M7
    S_W = [
        [1, 0, 0, 1, -1, 0, 1], # c00
        [0, 0, 1, 0, 1, 0, 0], # c01
        [0, 1, 0, 1, 0, 0, 0], # c10
        [1, -1, 1, 0, 0, 1, 0] # c11
    ]
    
    # Define index mappings for the 4 sub-problems
    # Sub-problem s: j_base = 2 * (s // 2), k_base = 2 * (s % 2)
    # Wait, my previous logic was:
    # SP0: j in {0,1}, k in {0,1} -> j_base=0, k_base=0
    # SP1: j in {0,1}, k in {2,3} -> j_base=0, k_base=2
    # SP2: j in {2,3}, k in {0,1} -> j_base=2, k_base=0
    # SP3: j in {2,3}, k in {2,3} -> j_base=2, k_base=2
    
    # Local index l maps to:
    # U: i = l // 2, j_local = l % 2
    # V: j_local = l // 2, k_local = l % 2
    # W: k_local = l // 2, i = l % 2
    
    for s in range(4):
        j_base = 2 * (s // 2)
        k_base = 2 * (s % 2)
        
        for t in range(7):
            col = s * 7 + t
            
            for l in range(4):
                # U
                i = l // 2
                j_local = l % 2
                j_global = j_base + j_local
                u_idx = i * m + j_global
                U[u_idx][col] = S_U[l][t]
                
                # V
                j_local_v = l // 2
                k_local = l % 2
                j_global_v = j_base + j_local_v
                k_global = k_base + k_local
                v_idx = j_global_v * p + k_global
                V[v_idx][col] = S_V[l][t]
                
                # W
                k_local_w = l // 2
                i_w = l % 2
                k_global_w = k_base + k_local_w
                w_idx = k_global_w * n + i_w
                W[w_idx][col] = S_W[l][t]
                
    # Verify the constructed rank-28 decomposition
    if _verify_fast(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if something went wrong
        return {"U": U_sb, "V": V_sb, "W": W_sb}
