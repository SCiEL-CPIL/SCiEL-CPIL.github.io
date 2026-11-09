import numpy as np

def construct():
    n, m, p = 3, 5, 7
    rank = 97
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_block(u_idx, v_idx, w_idx, u_mat, v_mat, w_mat, offset):
        for r in range(u_mat.shape[1]):
            for i in range(u_mat.shape[0]):
                if u_mat[i, r] != 0:
                    U[u_idx[i], offset + r] += u_mat[i, r]
            for j in range(v_mat.shape[0]):
                if v_mat[j, r] != 0:
                    V[v_idx[j], offset + r] += v_mat[j, r]
            for k in range(w_mat.shape[0]):
                if w_mat[k, r] != 0:
                    W[w_idx[k], offset + r] += w_mat[k, r]
                    
    # Block 1: 3x1x7 (Rank 21)
    u_idx_1 = [i * m + 0 for i in range(n)]
    v_idx_1 = [0 * p + k for k in range(p)]
    w_idx_1 = [k * n + i for i in range(n) for k in range(p)]
    # Schoolbook 3x1x7
    u_mat_1 = np.zeros((n, 21))
    v_mat_1 = np.zeros((p, 21))
    w_mat_1 = np.zeros((p*n, 21))
    col = 0
    for i in range(n):
        for k in range(p):
            u_mat_1[i, col] = 1
            v_mat_1[k, col] = 1
            w_mat_1[k*n+i, col] = 1
            col += 1
    add_block(u_idx_1, v_idx_1, w_idx_1, u_mat_1, v_mat_1, w_mat_1, 0)
    
    # Block 2: 3x4x7 (Rank 76)
    # Split 3x4x7 into 3x4x3 + 3x4x4
    # 3x4x3 (Rank 32)
    # 3x4x4 (Rank 44)
    
    # Sub-block 2.1: 3x4x3
    # Split 3x4x3 into 3x1x3 + 3x3x3
    # 3x1x3 (Rank 9)
    u_idx_211 = [i * m + 1 for i in range(n)]
    v_idx_211 = [1 * p + k for k in range(3)]
    w_idx_211 = [k * n + i for i in range(n) for k in range(3)]
    u_mat_211 = np.zeros((n, 9))
    v_mat_211 = np.zeros((3, 9))
    w_mat_211 = np.zeros((p*n, 9))
    col = 0
    for i in range(n):
        for k in range(3):
            u_mat_211[i, col] = 1
            v_mat_211[k, col] = 1
            w_mat_211[k*n+i, col] = 1
            col += 1
    add_block(u_idx_211, v_idx_211, w_idx_211, u_mat_211, v_mat_211, w_mat_211, 21)
    
    # 3x3x3 (Rank 23) - Winograd
    u_idx_212 = [i * m + j for j in range(1, 4) for i in range(n)] # Wait, indices are (i,j)
    # U index is i*m+j. j=1,2,3.
    u_idx_212 = []
    for i in range(n):
        for j in range(1, 4):
            u_idx_212.append(i * m + j)
    v_idx_212 = [j * p + k for j in range(1, 4) for k in range(3)]
    w_idx_212 = [k * n + i for i in range(n) for k in range(3)]
    
    # Winograd 3x3x3 Coefficients
    # ... (Hardcode here) ...
    
    # Sub-block 2.2: 3x4x4
    # Split 3x4x4 into 3x1x4 + 3x3x4
    # 3x1x4 (Rank 12)
    u_idx_221 = [i * m + 1 for i in range(n)]
    v_idx_221 = [1 * p + k for k in range(3, 7)]
    w_idx_221 = [k * n + i for i in range(n) for k in range(3, 7)]
    u_mat_221 = np.zeros((n, 12))
    v_mat_221 = np.zeros((4, 12))
    w_mat_221 = np.zeros((p*n, 12))
    col = 0
    for i in range(n):
        for k in range(4):
            u_mat_221[i, col] = 1
            v_mat_221[k, col] = 1
            w_mat_221[(k+3)*n+i, col] = 1
            col += 1
    add_block(u_idx_221, v_idx_221, w_idx_221, u_mat_221, v_mat_221, w_mat_221, 21+9+23)
    
    # 3x3x4 (Rank 32)
    # Split 3x3x4 into 3x3x3 + 3x3x1
    # 3x3x3 (Rank 23)
    # 3x3x1 (Rank 9)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
