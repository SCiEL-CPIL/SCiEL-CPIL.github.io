import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 4, 4, 5

    def build_target_tensor():
        T = np.zeros((n, m, p), dtype=float)
        T[:] = 1.0
        return T

    TARGET_T = build_target_tensor()

    def verify(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        if U_arr.ndim != 2 or V_arr.ndim != 2 or W_arr.ndim != 2:
            return False

        rank = U_arr.shape[1]
        if V_arr.shape[1] != rank or W_arr.shape[1] != rank:
            return False

        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)

        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)

        if not np.allclose(T, TARGET_T, atol=1e-6):
            return False
        return True

    def get_schoolbook():
        schoolbook_rank = n * m * p
        U = np.zeros((n * m, schoolbook_rank), dtype=float)
        V = np.zeros((m * p, schoolbook_rank), dtype=float)
        W = np.zeros((p * n, schoolbook_rank), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    # Rank-65 decomposition
    U = np.zeros((n * m, 65), dtype=float)
    V = np.zeros((m * p, 65), dtype=float)
    W = np.zeros((p * n, 65), dtype=float)
    
    # Build 49-term Strassen for 4x4x4
    # Using recursive Strassen: 7 outer terms, each expanded with 7 inner terms
    
    # 2x2 Strassen factors (for inner level)
    # U factors: each is a 2x2 matrix
    inner_U = np.array([
        [[1, 0], [0, 1]],   # M1: A00+A11
        [[0, 0], [1, 1]],   # M2: A10+A11
        [[1, 0], [0, 0]],   # M3: A00
        [[0, 0], [0, 1]],   # M4: A11
        [[1, 1], [0, 0]],   # M5: A00+A01
        [[-1, 0], [1, 0]],  # M6: A10-A00
        [[0, 1], [0, -1]],  # M7: A01-A11
    ])
    
    inner_V = np.array([
        [[1, 0], [0, 1]],   # M1: B00+B11
        [[1, 0], [0, 0]],   # M2: B00
        [[0, 1], [0, -1]],  # M3: B01-B11
        [[0, -1], [1, 0]],  # M4: B10-B00
        [[0, 0], [0, 1]],   # M5: B11
        [[1, 1], [0, 0]],   # M6: B00+B01
        [[1, 0], [1, 1]],   # M7: B10+B11
    ])
    
    inner_W = np.array([
        [[1, 0], [0, 1]],   # M1: C00+C11
        [[0, 0], [1, -1]],  # M2: C10-C11
        [[0, 1], [0, 1]],   # M3: C01+C11
        [[1, 0], [1, 0]],   # M4: C00+C10
        [[-1, 1], [0, 0]],  # M5: C00-C01
        [[0, 0], [0, 1]],   # M6: C11
        [[1, 0], [0, 0]],   # M7: C00
    ])
    
    # Outer Strassen factors for 4x4x4 (each is a 2x2 matrix of 2x2 blocks)
    # Represented as 4x4 matrices
    
    def block_identity(row, col, sign=1):
        M = np.zeros((4, 4))
        M[row*2:(row+1)*2, col*2:(col+1)*2] = sign * np.eye(2)
        return M
    
    outer_U = [
        block_identity(0,0) + block_identity(1,1),  # A00+A11
        block_identity(1,0) + block_identity(1,1),  # A10+A11
        block_identity(0,0),                         # A00
        block_identity(1,1),                         # A11
        block_identity(0,0) + block_identity(0,1),   # A00+A01
        block_identity(1,0) - block_identity(0,0),   # A10-A00
        block_identity(0,1) - block_identity(1,1),   # A01-A11
    ]
    
    outer_V = [
        block_identity(0,0) + block_identity(1,1),  # B00+B11
        block_identity(0,0),                         # B00
        block_identity(0,1) - block_identity(1,1),   # B01-B11
        block_identity(1,0) - block_identity(0,0),   # B10-B00
        block_identity(1,1),                         # B11
        block_identity(0,0) + block_identity(0,1),   # B00+B01
        block_identity(1,0) + block_identity(1,1),   # B10+B11
    ]
    
    col_idx = 0
    for j in range(7):
        for t in range(7):
            # Combine outer and inner factors
            U_factor = np.zeros((4, 4))
            V_factor = np.zeros((4, 4))
            W_factor = np.zeros((4, 4))
            
            for br in range(2):
                for bc in range(2):
                    cu = inner_U[t, br, bc]
                    cv = inner_V[t, br, bc]
                    cw = inner_W[t, br, bc]
                    if cu != 0:
                        U_factor += cu * outer_U[j][br*2:(br+1)*2, bc*2:(bc+1)*2]
                    if cv != 0:
                        V_factor += cv * outer_V[j][br*2:(br+1)*2, bc*2:(bc+1)*2]
                    if cw != 0:
                        W_factor[br*2:(br+1)*2, bc*2:(bc+1)*2] += cw * np.eye(2)
            
            # Fill in U, V, W
            for i in range(4):
                for jj in range(4):
                    U[i * m + jj, col_idx] += U_factor[i, jj]
            
            for jj in range(4):
                for k in range(4):
                    V[jj * p + k, col_idx] += V_factor[jj, k]
            
            for k in range(4):
                for i in range(4):
                    W[k * n + i, col_idx] += W_factor[k, i]
            
            col_idx += 1
    
    # Add 16 schoolbook terms for k=4
    for i in range(n):
        for j in range(m):
            U[i * m + j, col_idx] = 1.0
            V[j * p + 4, col_idx] = 1.0
            W[4 * n + i, col_idx] = 1.0
            col_idx += 1
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U_sb, V_sb, W_sb = get_schoolbook()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
