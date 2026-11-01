# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = 2, 2, 13
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    rank_schoolbook = n * m * p
    U_school = np.zeros((dim_u, rank_schoolbook))
    V_school = np.zeros((dim_v, rank_schoolbook))
    W_school = np.zeros((dim_w, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school

    start_time = time.time()
    timeout = 115.0

    for r in range(28, 41):
        if time.time() - start_time > timeout:
            break
            
        found = False
        for restart in range(50):
            if time.time() - start_time > timeout:
                break
                
            U = np.random.randn(dim_u, r)
            V = np.random.randn(dim_v, r)
            W = np.random.randn(dim_w, r)
            
            for it in range(100):
                # Update U
                for a in range(dim_u):
                    for r_idx in range(r):
                        M_r = np.outer(V[:, r_idx], W[:, r_idx])
                        # Compute residual
                        R = T[a]
                        for k in range(r):
                            if k != r_idx:
                                R -= U[a, k] * np.outer(V[:, k], W[:, k])
                        
                        denom = np.sum(M_r * M_r)
                        if denom > 1e-10:
                            U[a, r_idx] = np.sum(R * M_r) / denom

                # Update V
                for b in range(dim_v):
                    for r_idx in range(r):
                        M_r = np.outer(U[:, r_idx], W[:, r_idx]) # Shape (dim_u, dim_w)? No.
                        # Term is U[:,r] V[b,r] W[:,r].
                        # For fixed b, V[b,r] is scalar.
                        # R = T[:, b, :] - sum_{k != r} U[:,k] V[b,k] W[:,k]
                        # R is (dim_u, dim_w).
                        # M_r is (dim_u, dim_w) = U[:,r] outer W[:,r].
                        # V[b, r_idx] = sum(R * M_r) / sum(M_r * M_r)
                        
                        M_r = np.outer(U[:, r_idx], W[:, r_idx])
                        R = T[:, b, :]
                        for k in range(r):
                            if k != r_idx:
                                R -= U[:, k] * V[b, k] * W[:, k]
                        
                        denom = np.sum(M_r * M_r)
                        if denom > 1e-10:
                            V[b, r_idx] = np.sum(R * M_r) / denom

                # Update W
                for c in range(dim_w):
                    for r_idx in range(r):
                        M_r = np.outer(U[:, r_idx], V[:, r_idx])
                        R = T[:, :, c]
                        for k in range(r):
                            if k != r_idx:
                                R -= U[:, k] * V[:, k] * W[c, k]
                        
                        denom = np.sum(M_r * M_r)
                        if denom > 1e-10:
                            W[c, r_idx] = np.sum(R * M_r) / denom

            # Check convergence
            T_approx = np.zeros((dim_u, dim_v, dim_w))
            for k in range(r):
                T_approx += U[:, k] * np.outer(V[:, k], W[:, k])
            error = np.linalg.norm(T - T_approx)
            if error < 1e-5:
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                T_approx_int = np.zeros((dim_u, dim_v, dim_w))
                for k in range(r):
                    T_approx_int += U_int[:, k] * np.outer(V_int[:, k], W_int[:, k])
                
                if np.allclose(T, T_approx_int, atol=1e-9):
                    best_rank = r
                    best_U = U_int
                    best_V = V_int
                    best_W = W_int
                    found = True
                    break
        
        if found:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
