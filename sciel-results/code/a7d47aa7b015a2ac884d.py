import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Build the full multiplication tensor T
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()

    def verify_decomposition(U, V, W, tol=1e-7):
        T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
        return bool(np.allclose(T_recon, T, atol=tol))

    def try_round(U, V, W):
        # Try rounding to half-integers (scale 2) and integers (scale 1)
        for scale in [2, 1]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if verify_decomposition(U_r, V_r, W_r):
                return (U_r, V_r, W_r)
        return None

    # Strassen decomposition for 3x3x3
    # Indices: i,j,k in 0..2
    # a = 3i + j
    # b = 5j + k  (Note: p=5, but for the 3x3x3 core, we only use k=0,1,2)
    # c = 3k + i
    
    U_s = np.zeros((n * m, 7))
    V_s = np.zeros((m * p, 7))
    W_s = np.zeros((p * n, 7))
    
    def set_strassen_term(t, u_idx, u_val, v_idx, v_val, w_idx, w_val):
        U_s[u_idx, t] += u_val
        V_s[v_idx, t] += v_val
        W_s[w_idx, t] += w_val

    # p1 = (A00 + A22)(B00 + B22)
    set_strassen_term(0, 0, 1, 0, 1, 0, 1) # A00, B00, C00
    set_strassen_term(0, 8, 1, 2, 1, 2, 1) # A22, B22, C22 (Wait, C22 is k=2,i=2 -> c=8)
    # Correction: 
    # C indices: c = 3k + i
    # C00: k=0, i=0 -> c=0
    # C02: k=2, i=0 -> c=6
    # C20: k=0, i=2 -> c=2
    # C22: k=2, i=2 -> c=8
    # 
    # Let's re-map carefully.
    # A00: a=0
    # A22: a=8
    # B00: b=0 (j=0, k=0)
    # B22: b=2 (j=2, k=2) -> 5*2+2 = 12? No.
    # b = j*p + k. p=5.
    # B00: j=0, k=0 -> b=0
    # B22: j=2, k=2 -> b=12
    # C00: k=0, i=0 -> c=0
    # C22: k=2, i=2 -> c=8
    
    # Let's restart Strassen setup with correct indices.
    U_s.fill(0)
    V_s.fill(0)
    W_s.fill(0)
    
    # Term 0: p1 = (A00+A22)(B00+B22)
    # A: 0, 8
    # B: 0, 12
    # C: C00(0), C02(6), C20(2), C22(8)
    # Coeffs in C:
    # c00: +1, c02: +1, c20: +1, c22: +1
    U_s[0, 0] = 1; U_s[8, 0] = 1
    V_s[0, 0] = 1; V_s[12, 0] = 1
    W_s[0, 0] = 1; W_s[6, 0] = 1; W_s[2, 0] = 1; W_s[8, 0] = 1
    
    # Term 1: p2 = (A20+A22)B02
    # A20: a=6, A22: a=8
    # B02: b=2 (j=0, k=2)
    # C: c02(+1), c12(+1)
    # c02: k=2,i=0 -> c=6
    # c12: k=2,i=1 -> c=7
    U_s[6, 1] = 1; U_s[8, 1] = 1
    V_s[2, 1] = 1
    W_s[6, 1] = 1; W_s[7, 1] = 1
    
    # Term 2: p3 = A00(B02-B22)
    # A00: a=0
    # B02: b=2, B22: b=12
    # C: c02(-1), c10(-1)
    # c02: c=6, c10: k=0,i=1 -> c=1
    U_s[0, 2] = 1
    V_s[2, 2] = 1; V_s[12, 2] = -1
    W_s[6, 2] = -1; W_s[1, 2] = -1
    
    # Term 3: p4 = A22(B20-B00)
    # A22: a=8
    # B20: b=10 (j=2, k=0), B00: b=0
    # C: c20(+1), c11(+1)
    # c20: c=2, c11: k=1,i=1 -> c=4
    U_s[8, 3] = 1
    V_s[10, 3] = 1; V_s[0, 3] = -1
    W_s[2, 3] = 1; W_s[4, 3] = 1
    
    # Term 4: p5 = (A00+A10)B11
    # A00: a=0, A10: a=3
    # B11: b=6 (j=1, k=1)
    # C: c00(-1), c11(+1)
    # c00: c=0, c11: c=4
    U_s[0, 4] = 1; U_s[3, 4] = 1
    V_s[6, 4] = 1
    W_s[0, 4] = -1; W_s[4, 4] = 1
    
    # Term 5: p6 = (A12+A22)(B11+B22)
    # A12: a=5, A22: a=8
    # B11: b=6, B22: b=12
    # C: c01(-1), c12(-1), c20(-1), c21(-1)
    # c01: k=1,i=0 -> c=3
    # c12: k=2,i=1 -> c=7
    # c20: k=0,i=2 -> c=2
    # c21: k=1,i=2 -> c=5
    U_s[5, 5] = 1; U_s[8, 5] = 1
    V_s[6, 5] = 1; V_s[12, 5] = 1
    W_s[3, 5] = -1; W_s[7, 5] = -1; W_s[2, 5] = -1; W_s[5, 5] = -1
    
    # Term 6: p7 = (A00+A10-A12-A22)(B11+B22-B20-B00)
    # A: 0(+1), 3(+1), 5(-1), 8(-1)
    # B: 6(+1), 12(+1), 10(-1), 0(-1)
    # C: c00(+1), c11(-1), c20(+1), c21(-1)
    # c00: c=0, c11: c=4, c20: c=2, c21: c=5
    U_s[0, 6] = 1; U_s[3, 6] = 1; U_s[5, 6] = -1; U_s[8, 6] = -1
    V_s[6, 6] = 1; V_s[12, 6] = 1; V_s[10, 6] = -1; V_s[0, 6] = -1
    W_s[0, 6] = 1; W_s[4, 6] = -1; W_s[2, 6] = 1; W_s[5, 6] = -1

    # Compute residual tensor
    T_s = np.einsum('ad,bd,cd->abc', U_s, V_s, W_s)
    T_rem = T - T_s
    
    # T_rem should be zero for c in 0..8
    # Non-zero only for c in 9..14
    
    def als_step(U, V, W, T, reg=1e-10):
        r = U.shape[1]
        TW = V[:, np.newaxis, :] * W[np.newaxis, :, :]
        h_U = np.einsum('abc,bcd->ad', T, TW)
        G_U = np.einsum('bcd,bce->de', TW, TW)
        try:
            U_new = np.linalg.solve(G_U + reg * np.eye(r), h_U.T).T
        except np.linalg.LinAlgError:
            U_new = h_U
        
        UW = U_new[:, np.newaxis, :] * W[np.newaxis, :, :]
        h_V = np.einsum('abc,acd->bd', T, UW)
        G_V = np.einsum('acd,ace->de', UW, UW)
        try:
            V_new = np.linalg.solve(G_V + reg * np.eye(r), h_V.T).T
        except np.linalg.LinAlgError:
            V_new = h_V
        
        UV = U_new[:, np.newaxis, :] * V_new[np.newaxis, :, :]
        h_W = np.einsum('abc,abd->cd', T, UV)
        G_W = np.einsum('abd,abe->de', UV, UV)
        try:
            W_new = np.linalg.solve(G_W + reg * np.eye(r), h_W.T).T
        except np.linalg.LinAlgError:
            W_new = h_W
        
        return U_new, V_new, W_new

    def run_als_on_residual(r_extra, seed):
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(seed)
        U_x = np.random.randn(n * m, r_extra) * 0.5
        V_x = np.random.randn(m * p, r_extra) * 0.5
        W_x = np.random.randn(p * n, r_extra) * 0.5
        
        for it in range(100):
            if time.time() > deadline - 5:
                return None
            U_x, V_x, W_x = als_step(U_x, V_x, W_x, T_rem, reg=1e-10)
            
            if it % 10 == 9:
                # Combine with Strassen
                U_full = np.hstack([U_s, U_x])
                V_full = np.hstack([V_s, V_x])
                W_full = np.hstack([W_s, W_x])
                
                res = try_round(U_full, V_full, W_full)
                if res is not None:
                    return res
        
        U_full = np.hstack([U_s, U_x])
        V_full = np.hstack([V_s, V_x])
        W_full = np.hstack([W_s, W_x])
        res = try_round(U_full, V_full, W_full)
        if res is not None:
            return res
        return None

    # Try extra ranks from 1 to 10
    # Total rank = 7 + r_extra
    # Schoolbook is 45.
    # We expect r_extra around 6.
    for r_extra in range(1, 11):
        if time.time() > deadline - 15:
            break
        
        total_rank = 7 + r_extra
        if total_rank >= best_rank:
            continue
            
        found = False
        for seed in [42, 0, 1, 2, 3, 7, 13, 99, 123, 456]:
            if time.time() > deadline - 10:
                break
            
            result = run_als_on_residual(r_extra, seed)
            if result is not None:
                U_f, V_f, W_f = result
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_U = U_f.copy()
                    best_V = V_f.copy()
                    best_W = W_f.copy()
                found = True
                break
        
        if found:
            continue

    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
