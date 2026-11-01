import numpy as np
import time

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0

    # Strassen parameters for 2x2x2
    # U: (4, 7) - A factors
    # V: (4, 7) - B factors  
    # W: (4, 7) - C factors (trace convention: C[k*n+i] for 2x2, so C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[2], C[1,1]=W[3])
    U_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    V_strassen = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    def build_strassen_28():
        num_blocks = p // 2
        rank = num_blocks * 7
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for t in range(num_blocks):
            for r in range(7):
                # U: A indices are i*m+j = i*2+j for i in {0,1}, j in {0,1}
                # A flat: [A00, A01, A10, A11]
                for i in range(n * m):
                    U[i][col] = U_strassen[r][i]
                
                # V: B indices are j*p+k for j in {0,1}, k in {0,...,7}
                # For block t, we use B[:, 2t] and B[:, 2t+1]
                # B flat: [B00, B01, B02, B03, B04, B05, B06, B07, B10, B11, B12, B13, B14, B15, B16, B17]
                # For block t: B[0, 2t], B[0, 2t+1], B[1, 2t], B[1, 2t+1]
                # Indices: 0*8+2t = 2t, 0*8+2t+1 = 2t+1, 1*8+2t = 8+2t, 1*8+2t+1 = 8+2t+1
                v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
                for l in range(4):
                    V[v_indices[l]][col] = V_strassen[r][l]
                
                # W: C indices are k*n+i for k in {0,...,7}, i in {0,1}
                # For block t, we produce C[:, 2t] and C[:, 2t+1]
                # C flat: [C00, C01, C10, C11, C02, C03, C12, C13, ..., C07, C08, C17, C18]
                # Wait, p=8, so k in {0,...,7}
                # C[k*n+i] = C[k*2+i]
                # For k=2t: C[2t*2+0] = C[4t], C[2t*2+1] = C[4t+1]
                # For k=2t+1: C[(2t+1)*2+0] = C[4t+2], C[(2t+1)*2+1] = C[4t+3]
                # So W indices for block t: C[4t], C[4t+1], C[4t+2], C[4t+3]
                # Which correspond to i=0,k=2t; i=1,k=2t; i=0,k=2t+1; i=1,k=2t+1
                # In Strassen W, the 4 entries are [C00, C01, C10, C11] for a 2x2 output
                # But we need to map: W_strassen[r][0] -> C[0,0] of block -> C[i=0,k=2t] -> W[4t]
                # W_strassen[r][1] -> C[0,1] of block -> C[i=0,k=2t+1] -> W[4t+2]
                # W_strassen[r][2] -> C[1,0] of block -> C[i=1,k=2t] -> W[4t+1]
                # W_strassen[r][3] -> C[1,1] of block -> C[i=1,k=2t+1] -> W[4t+3]
                w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
                for l in range(4):
                    W[w_indices[l]][col] = W_strassen[r][l]
                
                col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W, rank, n, m, p):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for trial in range(5):
            A_rand = np.random.randn(n, m)
            B_rand = np.random.randn(m, p)
            C_expected = A_rand @ B_rand
            
            C_check = np.zeros((n, p))
            A_flat = A_rand.flatten()
            B_flat = B_rand.flatten()
            
            for r in range(rank):
                scalar = np.dot(U_arr[:, r], A_flat) * np.dot(V_arr[:, r], B_flat)
                w_vec = W_arr[:, r]
                for k in range(p):
                    for i in range(n):
                        C_check[i, k] += w_vec[k * n + i] * scalar
            
            if not np.allclose(C_check, C_expected, atol=1e-10):
                return False
        return True

    best = build_strassen_28()
    best_rank = 28
    
    if not verify_decomposition(best["U"], best["V"], best["W"], best_rank, n, m, p):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best = {"U": U, "V": V, "W": W}
        best_rank = rank_school
    
    return best
