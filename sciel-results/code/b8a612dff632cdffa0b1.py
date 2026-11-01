import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 4
    V_dim = m * p  # 28
    W_dim = p * n  # 28
    
    # Rank 49 decomposition using Strassen on 2x2 blocks of size 1x7
    rank = 49
    
    U = np.zeros((U_dim, rank), dtype=np.int8)
    V = np.zeros((V_dim, rank), dtype=np.int8)
    W = np.zeros((W_dim, rank), dtype=np.int8)
    
    # Strassen coefficients
    # s: 0..6
    # u[s][i][j]
    # v[s][j][q]
    # w[s][i'][q']
    
    u_coeffs = [
        [[1, 0], [0, 1]], # s=0: A00+A11
        [[0, 0], [1, 1]], # s=1: A10+A11
        [[1, 0], [0, 0]], # s=2: A00
        [[0, 0], [0, 1]], # s=3: A11
        [[1, 1], [0, 0]], # s=4: A00+A01
        [[-1, 0], [1, 0]],# s=5: A10-A00
        [[0, 1], [0, -1]] # s=6: A01-A11
    ]
    
    v_coeffs = [
        [[1, 0], [0, 1]], # s=0: B00+B11 (j=0,q=0; j=1,q=1)
        [[1, 0], [0, 0]], # s=1: B00
        [[0, 1], [0, -1]],# s=2: B01-B11
        [[-1, 0], [1, 0]],# s=3: B10-B00
        [[0, 0], [0, 1]], # s=4: B11
        [[1, 1], [0, 0]], # s=5: B00+B01
        [[0, 0], [1, 1]]  # s=6: B10+B11
    ]
    
    w_coeffs = [
        [[1, 0], [0, 1]], # s=0: C00+=M1, C11+=M1
        [[0, 0], [1, -1]],# s=1: C10+=M2, C11-=M2
        [[0, 1], [0, 1]], # s=2: C01+=M3, C11+=M3
        [[1, 0], [1, 0]], # s=3: C00+=M4, C10+=M4
        [[-1, 1], [0, 0]],# s=4: C00-=M5, C01+=M5
        [[0, 0], [0, 1]], # s=5: C11+=M6
        [[1, 0], [0, 0]]  # s=6: C00+=M7
    ]
    
    # Note: Indices for v_coeffs are [j][q]
    # j in 0..1, q in 0..1
    # Note: Indices for w_coeffs are [i'][q']
    # i' in 0..1, q' in 0..1
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    U[i * m + j, r] = u_coeffs[s][i][j]
            
            # Fill V
            for j in range(2):
                for q in range(2):
                    k = q * 7 + t
                    V[j * p + k, r] = v_coeffs[s][j][q]
            
            # Fill W
            for ip in range(2):
                for qp in range(2):
                    kp = qp * 7 + t
                    W[kp * n + ip, r] = w_coeffs[s][ip][qp]
    
    # Verification
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # We check a few entries or all
    T_expected = np.zeros((U_dim, V_dim, W_dim), dtype=np.int8)
    for i in range(2):
        for j in range(2):
            for k in range(14):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_expected[a, b, c] = 1
    
    T_reconstructed = np.zeros((U_dim, V_dim, W_dim), dtype=np.int8)
    for r in range(rank):
        T_reconstructed += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    if not np.allclose(T_reconstructed, T_expected):
        # Fallback to schoolbook if something went wrong
        rank_sb = 56
        U_sb = np.zeros((U_dim, rank_sb), dtype=np.int8)
        V_sb = np.zeros((V_dim, rank_sb), dtype=np.int8)
        W_sb = np.zeros((W_dim, rank_sb), dtype=np.int8)
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(14):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U_sb[a, col] = 1
                    V_sb[b, col] = 1
                    W_sb[c, col] = 1
                    col += 1
        U = U_sb
        V = V_sb
        W = W_sb

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
