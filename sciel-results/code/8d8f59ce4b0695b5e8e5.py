import numpy as np
import time
import random

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill
    np.random.seed(42)
    random.seed(42)

    def check_validity_residual(U, V, W, tol=1e-8):
        R = U.shape[1]
        V_mat = V.reshape(m, p, R)
        U_mat = U.reshape(n, m, R)
        W_mat = W.reshape(p, n, R)
        
        err = 0.0
        for i in range(n):
            if time.time() > deadline:
                return float('inf')
            for k in range(p):
                U_i = U_mat[i, :, :]
                V_k = V_mat[:, k, :]
                W_k_i = W_mat[k, i, :]
                M_ik = U_i @ np.diag(W_k_i) @ V_k.T
                diff = M_ik - np.eye(m)
                err += np.sum(diff**2)
        return err

    def get_rank_55_solution():
        rank = 55
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        # Naive part
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 2
                v_idx = 2 * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
        
        # Strassen part
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            U[0*m+0, col] += 1
            U[1*m+1, col] += 1
            V[0*p+k0, col] += 1
            V[1*p+k1, col] += 1
            W[k0*n+0, col] += 1
            W[k1*n+1, col] += 1
            col += 1
            
            U[1*m+0, col] += 1
            U[1*m+1, col] += 1
            V[0*p+k0, col] += 1
            W[k0*n+1, col] += 1
            W[k1*n+1, col] += -1
            col += 1
            
            U[0*m+0, col] += 1
            V[0*p+k1, col] += 1
            V[1*p+k1, col] += -1
            W[k1*n+0, col] += 1
            W[k1*n+1, col] += 1
            col += 1
            
            U[1*m+1, col] += 1
            V[1*p+k0, col] += 1
            V[0*p+k0, col] += -1
            W[k0*n+0, col] += 1
            W[k0*n+1, col] += 1
            col += 1
            
            U[0*m+0, col] += 1
            U[0*m+1, col] += 1
            V[1*p+k1, col] += 1
            W[k0*n+0, col] += -1
            W[k1*n+0, col] += 1
            col += 1
            
            U[1*m+0, col] += 1
            U[0*m+0, col] += -1
            V[0*p+k0, col] += 1
            V[0*p+k1, col] += 1
            W[k1*n+1, col] += 1
            col += 1
            
            U[0*m+1, col] += 1
            U[1*m+1, col] += -1
            V[1*p+k0, col] += 1
            V[1*p+k1, col] += 1
            W[k0*n+0, col] += 1
            col += 1
            
        return U, V, W

    U_55, V_55, W_55 = get_rank_55_solution()
    
    # Isolate the Strassen part (indices 20 to 54 in U,V,W)
    # U: rows 0,1,3,4 (j=0,1) are active. Rows 2,5 are 0.
    # V: rows 0..19 (j=0,1) are active. Rows 20..29 are 0.
    # W: all rows active.
    
    U_strassen = U_55[:20, 20:] # Wait, U has shape (6, 55). Rows 0,1,3,4 are for j=0,1.
    # Let's extract the sub-tensor for 2x2x10
    # U_sub: rows corresponding to A[i,j] with j in {0,1}. Indices: 0,1,3,4.
    # V_sub: rows corresponding to B[j,k] with j in {0,1}. Indices: 0..19.
    # W_sub: rows corresponding to C[i,k]. Indices: 0..19.
    
    u_rows = [0, 1, 3, 4]
    v_rows = list(range(20))
    w_rows = list(range(20))
    
    U_sub = U_55[u_rows, 20:]
    V_sub = V_55[v_rows, 20:]
    W_sub = W_55[w_rows, 20:]
    
    # Verify U_sub, V_sub, W_sub form a valid 2x2x10 decomposition
    def check_2x2x10(U, V, W):
        R = U.shape[1]
        U_mat = U.reshape(2, 2, R)
        V_mat = V.reshape(2, 10, R)
        W_mat = W.reshape(10, 2, R)
        
        err = 0.0
        for i in range(2):
            for k in range(10):
                U_i = U_mat[i, :, :]
                V_k = V_mat[:, k, :]
                W_k_i = W_mat[k, i, :]
                M_ik = U_i @ np.diag(W_k_i) @ V_k.T
                diff = M_ik - np.eye(2)
                err += np.sum(diff**2)
        return err

    err_sub = check_2x2x10(U_sub, V_sub, W_sub)
    
    best_U_sub, best_V_sub, best_W_sub = U_sub, V_sub, W_sub
    best_rank_sub = 35
    
    # Local search on 2x2x10
    # Try to reduce rank
    def try_reduce_2x2x10(U, V, W, R, max_time=10.0):
        start = time.time()
        while time.time() - start < max_time and R > 0:
            # Try removing a column
            reduced = False
            for c in range(R):
                if time.time() - start > max_time:
                    break
                U_r = np.delete(U, c, axis=1)
                V_r = np.delete(V, c, axis=1)
                W_r = np.delete(W, c, axis=1)
                err = check_2x2x10(U_r, V_r, W_r)
                if err < 1e-6:
                    U, V, W = U_r, V_r, W_r
                    R -= 1
                    reduced = True
                    break
            
            if not reduced:
                # Try random perturbation and solve
                if R > 10: # Don't go too low
                    U_p = U + 0.01 * np.random.randn(U.shape[0], R)
                    V_p = V + 0.01 * np.random.randn(V.shape[0], R)
                    
                    W_p = np.zeros((20, R))
                    valid = True
                    for k in range(10):
                        for i in range(2):
                            A_mat = np.zeros((4, R))
                            b_vec = np.zeros(4)
                            # Equations: sum_r U[i,j,r] W[k,i,r] V[j',k,r] = delta_jj'
                            # j, j' in {0,1}
                            for j in range(2):
                                for jp in range(2):
                                    idx = j * 2 + jp
                                    A_mat[idx, :] = U_p[i*2+j, :] * V_p[jp*10+k, :]
                                    if j == jp:
                                        b_vec[idx] = 1
                            try:
                                sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                                W_p[k*2+i, :] = sol
                            except:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        err = check_2x2x10(U_p, V_p, W_p)
                        if err < 1e-6:
                            U, V, W = U_p, V_p, W_p
                            # Now try to prune again
                            continue
            # If no reduction in this loop, break to avoid infinite loop if stuck
            # Actually, if no column removal worked, we might be at a local minimum.
            # We can break if we don't reduce in a few iterations.
            break 

        return U, V, W, R

    if time.time() < deadline - 20:
        best_U_sub, best_V_sub, best_W_sub, best_rank_sub = try_reduce_2x2x10(U_sub, V_sub, W_sub, 35, max_time=20.0)
    
    # Reconstruct full solution
    if best_rank_sub < 35:
        # Combine Naive (20) and optimized Strassen (best_rank_sub)
        total_rank = 20 + best_rank_sub
        U_new = np.zeros((6, total_rank))
        V_new = np.zeros((30, total_rank))
        W_new = np.zeros((20, total_rank))
        
        # Copy Naive part
        U_new[:, :20] = U_55[:, :20]
        V_new[:, :20] = V_55[:, :20]
        W_new[:, :20] = W_55[:, :20]
        
        # Copy Strassen part
        U_new[u_rows, 20:] = best_U_sub
        V_new[v_rows, 20:] = best_V_sub
        W_new[w_rows, 20:] = best_W_sub
        
        # Verify
        err_final = check_validity_residual(U_new, V_new, W_new)
        if err_final < 1e-6:
            return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    
    return {"U": U_55.tolist(), "V": V_55.tolist(), "W": W_55.tolist()}
