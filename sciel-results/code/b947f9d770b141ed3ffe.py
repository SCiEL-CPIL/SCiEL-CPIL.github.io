import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12
    
    # Build the tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Build the rank-42 Strassen decomposition
    rank_full = 42
    U_full = np.zeros((nm, rank_full), dtype=float)
    V_full = np.zeros((mp, rank_full), dtype=float)
    W_full = np.zeros((pn, rank_full), dtype=float)
    
    def add_strassen_block(i_start, j_start, k_start, offset, U, V, W):
        a00 = (i_start + 0) * m + (j_start + 0)
        a01 = (i_start + 0) * m + (j_start + 1)
        a10 = (i_start + 1) * m + (j_start + 0)
        a11 = (i_start + 1) * m + (j_start + 1)
        
        b00 = (j_start + 0) * p + (k_start + 0)
        b01 = (j_start + 0) * p + (k_start + 1)
        b10 = (j_start + 1) * p + (k_start + 0)
        b11 = (j_start + 1) * p + (k_start + 1)
        
        w00 = (k_start + 0) * n + (i_start + 0)
        w01 = (k_start + 1) * n + (i_start + 0)
        w10 = (k_start + 0) * n + (i_start + 1)
        w11 = (k_start + 1) * n + (i_start + 1)
        
        U[a00, offset] += 1.0; U[a11, offset] += 1.0
        V[b00, offset] += 1.0; V[b11, offset] += 1.0
        W[w00, offset] += 1.0; W[w11, offset] += 1.0
        
        U[a10, offset + 1] += 1.0; U[a11, offset + 1] += 1.0
        V[b00, offset + 1] += 1.0
        W[w10, offset + 1] += 1.0; W[w11, offset + 1] -= 1.0
        
        U[a00, offset + 2] += 1.0
        V[b01, offset + 2] += 1.0; V[b11, offset + 2] -= 1.0
        W[w01, offset + 2] += 1.0; W[w11, offset + 2] += 1.0
        
        U[a11, offset + 3] += 1.0
        V[b10, offset + 3] += 1.0; V[b00, offset + 3] -= 1.0
        W[w00, offset + 3] += 1.0; W[w10, offset + 3] += 1.0
        
        U[a00, offset + 4] += 1.0; U[a01, offset + 4] += 1.0
        V[b11, offset + 4] += 1.0
        W[w00, offset + 4] -= 1.0; W[w01, offset + 4] += 1.0
        
        U[a10, offset + 5] += 1.0; U[a00, offset + 5] -= 1.0
        V[b00, offset + 5] += 1.0; V[b01, offset + 5] += 1.0
        W[w11, offset + 5] += 1.0
        
        U[a01, offset + 6] += 1.0; U[a11, offset + 6] -= 1.0
        V[b10, offset + 6] += 1.0; V[b11, offset + 6] += 1.0
        W[w00, offset + 6] += 1.0
    
    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_block(i_start, j_start, k_start, offset, U_full, V_full, W_full)
                offset += 7
    
    def verify(U_mat, V_mat, W_mat, rank, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r in range(rank):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s - 1.0) > tol:
                        return False
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i = a // m
                    j = a % m
                    j_b = b // p
                    k = b % p
                    i_w = c % n
                    k_w = c // n
                    if i == i_w and j == j_b and k == k_w:
                        continue
                    s = 0.0
                    for r in range(rank):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s) > tol:
                        return False
        return True
    
    # Check if rank-42 works
    if not verify(U_full, V_full, W_full, rank_full):
        best_rank = n * m * p
        best_U = np.zeros((nm, best_rank))
        best_V = np.zeros((mp, best_rank))
        best_W = np.zeros((pn, best_rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * (m * p) + j * p + k
                    best_U[i * m + j, r] = 1
                    best_V[j * p + k, r] = 1
                    best_W[k * n + i, r] = 1
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    best_rank = rank_full
    best_U, best_V, best_W = U_full, V_full, W_full
    
    # Try to reduce rank using alternating least squares
    def try_rank(target_rank, max_iters=50, tol=1e-10):
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(42)  # For reproducibility
        U = U_full[:, :target_rank].copy()
        V = V_full[:, :target_rank].copy()
        W = W_full[:, :target_rank].copy()
        
        U += 0.01 * np.random.randn(nm, target_rank)
        V += 0.01 * np.random.randn(mp, target_rank)
        W += 0.01 * np.random.randn(pn, target_rank)
        
        prev_error = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Solve for U
            VW = (V[:, :, None] * W[None, :, :]).reshape(mp * pn, target_rank)
            rhs_U = T.reshape(nm, mp * pn)
            
            U_new = np.zeros((nm, target_rank))
            for a in range(nm):
                sol, residuals, rank_a, sv = np.linalg.lstsq(VW, rhs_U[a, :], rcond=None)
                U_new[a, :] = sol
            
            # Solve for V
            UW = (U_new[:, :, None] * W[None, :, :]).reshape(nm * pn, target_rank)
            rhs_V = T.transpose(1, 2, 0).reshape(mp, nm * pn)
            
            V_new = np.zeros((mp, target_rank))
            for b in range(mp):
                sol, residuals, rank_b, sv = np.linalg.lstsq(UW, rhs_V[b, :], rcond=None)
                V_new[b, :] = sol
            
            # Solve for W
            UV = (U_new[:, :, None] * V_new[None, :, :]).reshape(nm * mp, target_rank)
            rhs_W = T.transpose(2, 0, 1).reshape(pn, nm * mp)
            
            W_new = np.zeros((pn, target_rank))
            for c in range(pn):
                sol, residuals, rank_c, sv = np.linalg.lstsq(UV, rhs_W[c, :], rcond=None)
                W_new[c, :] = sol
            
            U, V, W = U_new, V_new, W_new
            
            # Compute error
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.sum((T_recon - T) ** 2)
            
            if abs(prev_error - error) < tol:
                break
            prev_error = error
        
        if verify(U, V, W, target_rank):
            return U, V, W
        return None
    
    # Try ranks from 36 down to 24
    for target_rank in range(36, 23, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(target_rank)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
