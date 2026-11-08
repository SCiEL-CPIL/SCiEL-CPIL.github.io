import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    # U: (n*m, R) = (16, R), U[i*m+j, r] for A[i,j]
    # V: (m*p, R) = (24, R), V[j*p+k, r] for B[j,k]
    # W: (p*n, R) = (24, R), W[k*n+i, r] for C[i,k]
    
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Top-level Strassen: 7 multiplications of (2x2) x (2x3)
    # A is 4x4, split into 2x2 blocks: A00, A01, A10, A11
    # B is 4x6, split into 2x3 blocks: B00, B01, B10, B11
    # C is 4x6, split into 2x3 blocks: C00, C01, C10, C11
    
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1}
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1}
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    col = 0
    for s in range(7):
        for si in range(7):
            inner_alpha = alpha
            inner_beta = beta
            inner_recon = recon
            
            for a in range(4):
                if inner_alpha[si].get(a, 0) != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s].get(r, 0) != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += inner_alpha[si][a] * alpha[s][r]
            
            for b in range(4):
                if inner_beta[si].get(b, 0) != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s].get(c, 0) != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += inner_beta[si][b] * beta[s][c]
            
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in inner_recon[si]:
                    # C block (a,b) at position (za, zb)
                    # C global: row = a*2 + za, col = b*3 + zb
                    # W index: col*n + row = (b*3 + zb)*4 + (a*2 + za)
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    if alpha[s].get(r, 0) != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]
                
                for c in range(4):
                    if beta[s].get(c, 0) != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]
                
                for a, b, gamma in recon[s]:
                    # C block (a,b) at position (u, 2)
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    # Verify
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 6)
    C_test = A_test @ B_test
    
    A_flat = A_test.reshape(16)
    B_flat = B_test.reshape(24)
    
    C_computed = np.zeros(24)
    for k in range(R):
        C_computed += np.dot(U[:, k], A_flat) * np.dot(V[:, k], B_flat) * W[:, k]
    
    C_from_decomp = C_computed.reshape(4, 6)
    error = np.max(np.abs(C_from_decomp - C_test))
    
    if error > 1e-6:
        R_fallback = 96
        U_fallback = np.zeros((16, R_fallback))
        V_fallback = np.zeros((24, R_fallback))
        W_fallback = np.zeros((24, R_fallback))
        idx = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    U_fallback[i * 4 + j, idx] = 1
                    V_fallback[j * 6 + k, idx] = 1
                    W_fallback[k * 4 + i, idx] = 1
                    idx += 1
        U, V, W = U_fallback, V_fallback, W_fallback
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
