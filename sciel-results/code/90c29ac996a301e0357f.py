import numpy as np
import time

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    deadline = start_time + 115 # Safety margin

    # Target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook fallback
    rank_fb = n * m * p
    U_fb = np.zeros((n*m, rank_fb), dtype=float)
    V_fb = np.zeros((m*p, rank_fb), dtype=float)
    W_fb = np.zeros((p*n, rank_fb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = rank_fb

    def check_decomp(U, V, W):
        T_decomp = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
        return np.allclose(T_decomp, T, atol=1e-9)

    def snap_to_half(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        if check_decomp(U_s, V_s, W_s):
            return U_s, V_s, W_s
        return None

    # Strategy: Tensor slicing to reduce to 3x3x3 subproblems
    # C = A * B where A is 3x3, B is 3x4
    # Split B into B = [B1, B2] where B1, B2 are 3x2
    # C = [C1, C2] where C1, C2 are 3x2
    # C1 = A * B1 (3x3 * 3x2 = 3x2)
    # C2 = A * B2 (3x3 * 3x2 = 3x2)
    
    # Each 3x3 * 3x2 multiplication can be done with rank 18 (schoolbook) or potentially less
    # But we can use a block approach: treat B1 and B2 as 3x3 matrices with last column zero
    
    # Alternative: Use Strassen-like decomposition for 3x3*3x3 (rank 7) 
    # and combine for 3x3*3x4
    
    # Let's try a direct approach: find rank r decomposition via ALS
    
    def try_rank_als(target_rank, max_iters=100, n_restarts=5):
        """Try to find decomposition of given rank using ALS"""
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
                
            np.random.seed(42 + restart)
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                
                # Update U
                for a in range(n*m):
                    M = np.einsum('bt,ct->bct', V, W).reshape(m*p * p*n, target_rank)
                    b_vec = T[a, :, :].reshape(-1)
                    try:
                        U[a, :] = np.linalg.lstsq(M.T, b_vec, rcond=None)[0]
                    except:
                        pass
                
                # Update V
                for b in range(m*p):
                    M = np.einsum('at,ct->act', U, W).reshape(n*m * p*n, target_rank)
                    b_vec = T[:, b, :].reshape(-1)
                    try:
                        V[b, :] = np.linalg.lstsq(M.T, b_vec, rcond=None)[0]
                    except:
                        pass
                
                # Update W
                for c in range(p*n):
                    M = np.einsum('at,bt->abt', U, V).reshape(n*m * m*p, target_rank)
                    b_vec = T[:, :, c].reshape(-1)
                    try:
                        W[c, :] = np.linalg.lstsq(M.T, b_vec, rcond=None)[0]
                    except:
                        pass
                
                if check_decomp(U, V, W):
                    snapped = snap_to_half(U, V, W)
                    if snapped:
                        return snapped[0], snapped[1], snapped[2]
                    return U, V, W
                    
        return None

    # Try various ranks from 20 to 35
    for target_rank in range(20, 36):
        if time.time() > deadline - 20:
            break
            
        result = try_rank_als(target_rank, max_iters=30, n_restarts=3)
        if result:
            U_res, V_res, W_res = result
            return {"U": U_res.tolist(), "V": V_res.tolist(), "W": W_res.tolist()}

    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
