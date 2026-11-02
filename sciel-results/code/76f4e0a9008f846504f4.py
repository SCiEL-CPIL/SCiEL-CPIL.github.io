import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = 2, 3, 8
    
    # Target rank to try to beat 44
    target_rank = 32
    
    def verify(U, V, W):
        r = U.shape[1]
        # Check for exact integer/half-integer match
        T_check = np.einsum('ad,bd,cd->abc', U, V, W)
        T_exp = np.zeros((n*m, m*p, p*n), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    T_exp[a_idx, b_idx, c_idx] = 1
        # Check if T_check is close to T_exp
        # Since we want exact, we check diff
        return np.allclose(T_check, T_exp, atol=1e-9)

    def round_and_verify(U, V, W):
        # Round to nearest 0.5
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify(U_r, V_r, W_r):
            return U_r, V_r, W_r
        return None

    # 1. Try ALS search for rank 32
    try:
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        # Initialize with a valid rank-44 solution and truncate?
        # Better: Use the rank-44 solution as a warm start for higher rank, 
        # but we want lower rank.
        # Let's just run ALS from random init.
        
        # Construct target tensor T
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0

        # Reshape T to matrices for ALS
        # U (nm, r), V (mp, r), W (pn, r)
        # T = sum U[:, d] outer V[:, d] outer W[:, d]
        
        # ALS iterations
        max_iters = 100
        tol = 1e-8
        
        for it in range(max_iters):
            if time.time() > deadline - 20:
                break
                
            # Update U
            # T_unfolded_1 (nm, mp*pn) = sum_d U[:, d] outer (V[:, d] outer W[:, d])
            # V_W = np.einsum('bd,cd->bc', V, W) # (mp, pn)
            # T_mat1 = np.einsum('bc,abc->ab', V_W, T) ? No.
            
            # T[a, b, c] = sum_d U[a, d] V[b, d] W[c, d]
            # For fixed a: T[a, :, :] = sum_d U[a, d] (V[:, d] outer W[:, d])
            # Let M_d = V[:, d] outer W[:, d]
            # T_a = sum_d U[a, d] M_d
            # This is a linear system for U[a, :] if we flatten M_d.
            
            # More efficient:
            # T_1 (nm, mp*pn) where T_1[a, b*pn+c] = T[a, b, c]
            # B_mat (mp*pn, r) where B_mat[b*pn+c, d] = V[b, d] * W[c, d]
            # T_1 = U @ B_mat
            # U = T_1 @ pinv(B_mat)
            
            B_mat = np.einsum('bd,cd->bc', V, W).reshape(m*p, p*n, target_rank).transpose(1, 0, 2).reshape(p*n*m*p, target_rank)
            # Wait, dimensions:
            # V is (mp, r), W is (pn, r).
            # Outer product V[:, d] and W[:, d] is (mp, pn).
            # Stack for all d: (mp, pn, r).
            # Reshape to (mp*pn, r).
            
            T_1 = T.reshape(n*m, m*p*p*n)
            
            # Solve U = T_1 @ B_mat^T (B_mat @ B_mat^T)^-1
            # Use least squares
            try:
                U_new, _, _, _ = np.linalg.lstsq(B_mat.T, T_1.T, rcond=None)
                U = U_new.T
            except:
                pass

            # Update V
            # T_2 (mp, nm*pn)
            # A_mat (nm*pn, r) where A_mat[a*pn+c, d] = U[a, d] * W[c, d]
            A_mat = np.einsum('ad,cd->ac', U, W).reshape(n*m, p*n, target_rank).transpose(1, 0, 2).reshape(p*n*n*m, target_rank)
            T_2 = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n)
            try:
                V_new, _, _, _ = np.linalg.lstsq(A_mat.T, T_2.T, rcond=None)
                V = V_new.T
            except:
                pass

            # Update W
            # T_3 (pn, nm*mp)
            # C_mat (nm*mp, r) where C_mat[a*mp+b, d] = U[a, d] * V[b, d]
            C_mat = np.einsum('ad,bd->ab', U, V).reshape(n*m, m*p, target_rank).transpose(1, 0, 2).reshape(m*p*n*m, target_rank)
            T_3 = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p)
            try:
                W_new, _, _, _ = np.linalg.lstsq(C_mat.T, T_3.T, rcond=None)
                W = W_new.T
            except:
                pass
            
            # Check convergence / exactness
            if it % 10 == 0:
                res = round_and_verify(U, V, W)
                if res:
                    U_r, V_r, W_r = res
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
    except Exception:
        pass

    # 2. Fallback to rank-44 construction
    U = np.zeros((n*m, 44), dtype=np.int64)
    V = np.zeros((m*p, 44), dtype=np.int64)
    W = np.zeros((p*n, 44), dtype=np.int64)
    
    col_offset = 0
    
    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # Block 1: A[:, 0] * B[0, k0:k1]
        U[0*m + 0, col_offset] = 1
        V[0*p + k0, col_offset] = 1
        W[k0*n + 0, col_offset] = 1
        col_offset += 1
        
        U[0*m + 0, col_offset] = 1
        V[0*p + k1, col_offset] = 1
        W[k1*n + 0, col_offset] = 1
        col_offset += 1
        
        U[1*m + 0, col_offset] = 1
        V[0*p + k0, col_offset] = 1
        W[k0*n + 1, col_offset] = 1
        col_offset += 1
        
        U[1*m + 0, col_offset] = 1
        V[0*p + k1, col_offset] = 1
        W[k1*n + 1, col_offset] = 1
        col_offset += 1
        
        # Block 2: A[:, 1:3] * B[1:3, k0:k1] (Strassen)
        # M1
        U[0*m + 1, col_offset] = 1
        U[1*m + 2, col_offset] = 1
        V[1*p + k0, col_offset] = 1
        V[2*p + k1, col_offset] = 1
        W[k0*n + 0, col_offset] = 1
        W[k1*n + 1, col_offset] = 1
        col_offset += 1
        
        # M2
        U[1*m + 1, col_offset] = 1
        U[1*m + 2, col_offset] = 1
        V[1*p + k0, col_offset] = 1
        W[k0*n + 1, col_offset] = 1
        W[k1*n + 1, col_offset] = -1
        col_offset += 1
        
        # M3
        U[0*m + 1, col_offset] = 1
        V[1*p + k1, col_offset] = 1
        V[2*p + k1, col_offset] = -1
        W[k1*n + 0, col_offset] = 1
        W[k1*n + 1, col_offset] = 1
        col_offset += 1
        
        # M4
        U[1*m + 2, col_offset] = 1
        V[2*p + k0, col_offset] = 1
        V[1*p + k0, col_offset] = -1
        W[k0*n + 0, col_offset] = 1
        W[k0*n + 1, col_offset] = 1
        col_offset += 1
        
        # M5
        U[0*m + 1, col_offset] = 1
        U[0*m + 2, col_offset] = 1
        V[2*p + k1, col_offset] = 1
        W[k0*n + 0, col_offset] = -1
        W[k1*n + 0, col_offset] = 1
        col_offset += 1
        
        # M6
        U[1*m + 1, col_offset] = 1
        U[0*m + 1, col_offset] = -1
        V[1*p + k0, col_offset] = 1
        V[1*p + k1, col_offset] = 1
        W[k1*n + 1, col_offset] = 1
        col_offset += 1
        
        # M7
        U[0*m + 2, col_offset] = 1
        U[1*m + 2, col_offset] = -1
        V[2*p + k0, col_offset] = 1
        V[2*p + k1, col_offset] = 1
        W[k0*n + 0, col_offset] = 1
        col_offset += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
