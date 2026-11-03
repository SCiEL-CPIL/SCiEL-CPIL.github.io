import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 118
    
    n, m, p = 3, 4, 6
    
    # Fast vectorized verification
    def verify_fast(U, V, W):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U[i*m+j, :] * V[j*p+k, :]
                    s = np.dot(prod, W[k*n+i, :])
                    expected = 1.0 if j == k else 0.0
                    err = abs(s - expected)
                    if err > max_err:
                        max_err = err
        return max_err
    
    # Build the rank-66 decomposition
    U66 = [[0] * 66 for _ in range(n * m)]
    V66 = [[0] * 66 for _ in range(m * p)]
    W66 = [[0] * 66 for _ in range(p * n)]
    
    col = 0
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = a_r * m + a_c
        A01 = a_r * m + a_c + 1
        A10 = (a_r + 1) * m + a_c
        A11 = (a_r + 1) * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C00 = c_c * n + c_r
        C01 = (c_c + 1) * n + c_r
        C10 = c_c * n + (c_r + 1)
        C11 = (c_c + 1) * n + (c_r + 1)
        
        U66[A00][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1; V66[B11][col] += 1
        W66[C00][col] += 1; W66[C11][col] += 1
        col += 1
        U66[A10][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1
        W66[C10][col] += 1; W66[C11][col] -= 1
        col += 1
        U66[A00][col] += 1
        V66[B01][col] += 1; V66[B11][col] -= 1
        W66[C01][col] += 1; W66[C11][col] += 1
        col += 1
        U66[A11][col] += 1
        V66[B10][col] += 1; V66[B00][col] -= 1
        W66[C00][col] += 1; W66[C10][col] += 1
        col += 1
        U66[A00][col] += 1; U66[A01][col] += 1
        V66[B11][col] += 1
        W66[C00][col] -= 1; W66[C01][col] += 1
        col += 1
        U66[A10][col] += 1; U66[A00][col] -= 1
        V66[B00][col] += 1; V66[B01][col] += 1
        W66[C11][col] += 1
        col += 1
        U66[A01][col] += 1; U66[A11][col] -= 1
        V66[B10][col] += 1; V66[B11][col] += 1
        W66[C00][col] += 1
        col += 1
    
    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r
        
        U66[A0][col] += 1; V66[B00][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B10][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A0][col] += 1; V66[B01][col] += 1; W66[C1][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B11][col] += 1; W66[C1][col] += 1
        col += 1
    
    add_1x2_2x2(0, 0, 0, 0, 0, 0)
    add_1x2_2x2(0, 2, 2, 0, 0, 0)
    add_1x2_2x2(0, 0, 0, 2, 0, 2)
    add_1x2_2x2(0, 2, 2, 2, 0, 2)
    add_1x2_2x2(0, 0, 0, 4, 0, 4)
    add_1x2_2x2(0, 2, 2, 4, 0, 4)
    add_strassen(1, 0, 0, 0, 1, 0)
    add_strassen(1, 2, 2, 0, 1, 0)
    add_strassen(1, 0, 0, 2, 1, 2)
    add_strassen(1, 2, 2, 2, 1, 2)
    add_strassen(1, 0, 0, 4, 1, 4)
    add_strassen(1, 2, 2, 4, 1, 4)
    
    # Convert to numpy
    U = np.array(U66, dtype=float)
    V = np.array(V66, dtype=float)
    W = np.array(W66, dtype=float)
    
    err66 = verify_fast(U, V, W)
    if err66 > 1e-10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by checking linear dependence of tensor components
    # Build M[idx, l] = U[i*m+j,l] * V[j*p+k,l] * W[k*n+i,l]
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    n_idx = len(indices)
    M = np.zeros((n_idx, 66))
    for idx, (i, j, k) in enumerate(indices):
        M[idx, :] = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
    
    rank = 66
    best_U, best_V, best_W = U, V, W
    
    # Check rank of M
    rank_M = np.linalg.matrix_rank(M, tol=1e-10)
    
    # If rank_M < 66, there's linear dependence, try to find which column to drop
    if rank_M < 66:
        for c in range(66):
            if time.time() > deadline - 5:
                break
            
            cols = [l for l in range(66) if l != c]
            M_sub = M[:, cols]
            
            # Check if M[:, c] is in column space of M_sub using SVD
            try:
                U_svd, s_vals, Vt_svd = np.linalg.svd(M_sub, full_matrices=False)
            except:
                continue
            
            proj = U_svd @ (U_svd.T @ M[:, c])
            residual = np.linalg.norm(proj - M[:, c])
            
            if residual < 1e-10:
                keep = cols
                U_new = U[:, keep]
                V_new = V[:, keep]
                W_new = W[:, keep]
                
                err = verify_fast(U_new, V_new, W_new)
                if err < 1e-10:
                    rank = 65
                    best_U, best_V, best_W = U_new, V_new, W_new
                    break
    
    # If still rank 66, try optimization-based reduction
    if rank == 66 and time.time() < deadline - 15:
        for c in range(66):
            if time.time() > deadline - 15:
                break
            
            U_try = U.copy()
            V_try = V.copy()
            W_try = W.copy()
            U_try[:, c] = 0
            V_try[:, c] = 0
            W_try[:, c] = 0
            
            for it in range(20):
                if time.time() > deadline - 15:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A_mat = np.zeros((p, 65))
                        b_vec = np.zeros(p)
                        for k in range(p):
                            A_mat[k, :] = V_try[j*p+k, :65] * W_try[k*n+i, :65]
                            b_vec[k] = 1.0 if j == k else 0.0
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        U_try[i*m+j, :65] = sol
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A_mat = np.zeros((n, 65))
                        b_vec = np.zeros(n)
                        for i in range(n):
                            A_mat[i, :] = U_try[i*m+j, :65] * W_try[k*n+i, :65]
                            b_vec[i] = 1.0 if j == k else 0.0
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        V_try[j*p+k, :65] = sol
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A_mat = np.zeros((m, 65))
                        b_vec = np.zeros(m)
                        for j in range(m):
                            A_mat[j, :] = U_try[i*m+j, :65] * V_try[j*p+k, :65]
                            b_vec[j] = 1.0 if j == k else 0.0
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        W_try[k*n+i, :65] = sol
                
                err = verify_fast(U_try[:, :65], V_try[:, :65], W_try[:, :65])
                if err < 1e-10:
                    rank = 65
                    best_U = U_try[:, :65]
                    best_V = V_try[:, :65]
                    best_W = W_try[:, :65]
                    break
                if err > 1.0:
                    break
            
            if rank == 65:
                break
    
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
