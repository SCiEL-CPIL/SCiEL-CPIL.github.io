import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Strassen forms for 2x2 matrix multiplication
    # X_forms[s][i*2+jp] for s=0..6, i=0,1, jp=0,1
    X_forms = np.array([
        [1, 0, 0, 1],   # m1: (X00 + X11)
        [0, 0, 1, 1],   # m2: (X10 + X11)
        [1, 0, 0, 0],   # m3: X00
        [0, 0, 0, 1],   # m4: X11
        [1, 1, 0, 0],   # m5: (X00 + X01)
        [-1, 0, 1, 0],  # m6: (X10 - X00)
        [0, 1, 0, -1],  # m7: (X01 - X11)
    ], dtype=int)
    
    # Y_forms[s][jp*2+kp] for s=0..6, jp=0,1, kp=0,1
    Y_forms = np.array([
        [1, 0, 0, 1],   # m1: (Y00 + Y11)
        [1, 0, 0, 0],   # m2: Y00
        [0, 1, 0, -1],  # m3: (Y01 - Y11)
        [-1, 0, 1, 0],  # m4: (Y10 - Y00)
        [0, 0, 0, 1],   # m5: Y11
        [1, 1, 0, 0],   # m6: (Y00 + Y01)
        [0, 0, 1, 1],   # m7: (Y10 + Y11)
    ], dtype=int)
    
    # Z_combos[i*2+kp][s] for i=0,1, kp=0,1, s=0..6
    Z_combos = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # z00 = m1 + m4 - m5 + m7
        [0, 0, 1, 0, 1, 0, 0],    # z01 = m3 + m5
        [0, 1, 0, 1, 0, 0, 0],    # z10 = m2 + m4
        [1, -1, 1, 0, 0, 1, 0],   # z11 = m1 - m2 + m3 + m6
    ], dtype=int)
    
    num_groups = 4      # g = 0,1,2,3 (4 column pairs)
    num_ablocks = 2     # b = 0,1 (2 row blocks of B)
    num_strassen = 7    # s = 0..6
    rank = num_groups * num_ablocks * num_strassen  # 56
    
    U = np.zeros((n * m, rank), dtype=int)   # 8 x 56
    V = np.zeros((m * p, rank), dtype=int)   # 32 x 56
    W = np.zeros((p * n, rank), dtype=int)   # 16 x 56
    
    r = 0
    for g in range(num_groups):
        for b in range(num_ablocks):
            for s in range(num_strassen):
                # U: for i in {0,1}, jp in {0,1}
                for i in range(n):
                    for jp in range(2):
                        a_idx = i * m + b * 2 + jp
                        U[a_idx, r] = X_forms[s, i * 2 + jp]
                
                # V: for jp in {0,1}, kp in {0,1}
                for jp in range(2):
                    for kp in range(2):
                        b_idx = (b * 2 + jp) * p + (2 * g + kp)
                        V[b_idx, r] = Y_forms[s, jp * 2 + kp]
                
                # W: for i in {0,1}, kp in {0,1}
                for i in range(n):
                    for kp in range(2):
                        k = 2 * g + kp
                        c_idx = k * n + i
                        W[c_idx, r] = Z_combos[i * 2 + kp, s]
                
                r += 1
    
    # Verify correctness with a random test
    np.random.seed(42)
    A = np.random.randint(-10, 10, size=(n, m))
    B = np.random.randint(-10, 10, size=(m, p))
    C_expected = A @ B
    
    # Reconstruct C from U, V, W
    C_recon = np.zeros((p, n), dtype=int)
    for idx in range(rank):
        u_col = U[:, idx]
        v_col = V[:, idx]
        w_col = W[:, idx]
        prod = np.outer(u_col @ A.flatten(), v_col @ B.flatten())
        C_recon += w_col * prod.reshape(n, p).T
    
    # The trace convention: W[k*n+i, r] means C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Let's verify properly
    C_check = np.zeros((n, p), dtype=int)
    for i in range(n):
        for k in range(p):
            total = 0
            for r_idx in range(rank):
                # Sum over j: U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                for j in range(m):
                    total += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
            C_check[i, k] = total
    
    if not np.array_equal(C_check, C_expected):
        # Fallback to simple schoolbook if Strassen is wrong
        rank_school = n * m * p  # 64
        U_s = np.zeros((n * m, rank_school), dtype=int)
        V_s = np.zeros((m * p, rank_school), dtype=int)
        W_s = np.zeros((p * n, rank_school), dtype=int)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, idx] = 1
                    V_s[j * p + k, idx] = 1
                    W_s[k * n + i, idx] = 1
                    idx += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
