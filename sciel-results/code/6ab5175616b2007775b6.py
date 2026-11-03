import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40 second margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Strassen decomposition for 2x2x2
    strassen_decomp = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {1: 1, 0: -1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for j1 in range(2):
        for l in range(4):
            for u_coeffs, v_coeffs, w_coeffs in strassen_decomp:
                for li, coeff in u_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    global_i = li_i
                    global_j = 2 * j1 + li_j
                    U[global_i * m + global_j][col] += coeff
                for li, coeff in v_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    global_j = 2 * j1 + li_i
                    global_k = 2 * l + li_j
                    V[global_j * p + global_k][col] += coeff
                for li, coeff in w_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    global_i = li_i
                    global_k = 2 * l + li_j
                    W[global_k * n + global_i][col] += coeff
                col += 1
        
        for k in range(8):
            for j2 in range(2):
                global_i = 2
                global_j = 2 * j1 + j2
                U[global_i * m + global_j][col] = 1
                V[global_j * p + k][col] = 1
                W[k * n + global_i][col] = 1
                col += 1
    
    # Verify
    if time.time() < deadline - 5:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        verified = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                    if abs(s - 1.0) > 1e-9:
                        verified = False
                        break
                if not verified:
                    break
            if not verified:
                break
        
        if not verified:
            rank_fb = n * m * p
            U_fb = [[0] * rank_fb for _ in range(n * m)]
            V_fb = [[0] * rank_fb for _ in range(m * p)]
            W_fb = [[0] * rank_fb for _ in range(p * n)]
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j][idx] = 1
                        V_fb[j * p + k][idx] = 1
                        W_fb[k * n + i][idx] = 1
                        idx += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Try to reduce rank by removing linearly dependent columns
    if time.time() < deadline - 10:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check if any column is zero
        nonzero_cols = []
        for r in range(rank):
            if np.any(U_arr[:, r] != 0) or np.any(V_arr[:, r] != 0) or np.any(W_arr[:, r] != 0):
                nonzero_cols.append(r)
        
        if len(nonzero_cols) < rank:
            U_arr = U_arr[:, nonzero_cols]
            V_arr = V_arr[:, nonzero_cols]
            W_arr = W_arr[:, nonzero_cols]
            rank = len(nonzero_cols)
        
        # Try greedy column removal: for each column, check if it's redundant
        # A column is redundant if removing it still gives correct results
        current_rank = rank
        while current_rank > 0:
            if time.time() > deadline - 15:
                break
            
            U_curr = U_arr[:, :current_rank]
            V_curr = V_arr[:, :current_rank]
            W_curr = W_arr[:, :current_rank]
            
            # Try removing each column and see if we can still verify
            removed = False
            for r in range(current_rank - 1, -1, -1):
                if time.time() > deadline - 15:
                    break
                
                U_test = np.delete(U_curr, r, axis=1)
                V_test = np.delete(V_curr, r, axis=1)
                W_test = np.delete(W_curr, r, axis=1)
                
                # Quick check: verify a sample of (i,j,k) tuples
                sample_verified = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U_test[i * m + j] * V_test[j * p + k], W_test[k * n + i])
                            if abs(s - 1.0) > 1e-9:
                                sample_verified = False
                                break
                        if not sample_verified:
                            break
                    if not sample_verified:
                        break
                
                if sample_verified:
                    U_arr = U_test
                    V_arr = V_test
                    W_arr = W_test
                    current_rank -= 1
                    removed = True
                    break
            
            if not removed:
                break
        
        if current_rank < rank:
            U = U_arr.tolist()
            V = V_arr.tolist()
            W = W_arr.tolist()
    
    return {"U": U, "V": V, "W": W}
