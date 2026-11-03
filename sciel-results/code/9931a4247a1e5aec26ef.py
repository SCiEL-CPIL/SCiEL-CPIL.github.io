import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit
    
    # Target: 3x3 by 3x8
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to solve 3x3x4 and double it
    # If we find rank r for 3x3x4, total rank is 2r
    # We try r from 18 to 35
    for r_sub in range(18, 36):
        if time.time() > deadline:
            break
        if 2 * r_sub >= best_rank:
            continue
            
        # Solve 3x3x4
        n_sub, m_sub, p_sub = 3, 3, 4
        dim_u = n_sub * m_sub
        dim_v = m_sub * p_sub
        dim_w = p_sub * n_sub
        
        # Target tensor T
        T = np.zeros((dim_u, dim_v, dim_w))
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    idx_u = i * m_sub + j
                    idx_v = j * p_sub + k
                    idx_w = k * n_sub + i
                    T[idx_u, idx_v, idx_w] = 1
        
        # Initialize random decomposition
        # Use multiple restarts
        found = False
        for restart in range(10):
            if time.time() > deadline:
                break
                
            U_sub = np.random.randn(dim_u, r_sub)
            V_sub = np.random.randn(dim_v, r_sub)
            W_sub = np.random.randn(dim_w, r_sub)
            
            # Gradient descent
            lr = 0.01
            for step in range(1000):
                if time.time() > deadline:
                    break
                
                # Compute residual
                # R = T - sum(U_c V_c W_c)
                # Efficient computation:
                # Residual R[a,b,c] = T[a,b,c] - sum_k U[a,k] V[b,k] W[c,k]
                
                # Sum term
                S = np.einsum('ak,bk,ck->abc', U_sub, V_sub, W_sub)
                R = T - S
                
                err = np.linalg.norm(R)
                if err < 1e-6:
                    found = True
                    break
                
                # Gradients
                # dR/du[a,k] = - sum_b,c R[a,b,c] V[b,k] W[c,k]
                # dR/dv[b,k] = - sum_a,c R[a,b,c] U[a,k] W[c,k]
                # dR/dw[c,k] = - sum_a,b R[a,b,c] U[a,k] V[b,k]
                
                grad_u = -np.einsum('abc,bk,ck->ak', R, V_sub, W_sub)
                grad_v = -np.einsum('abc,ak,ck->bk', R, U_sub, W_sub)
                grad_w = -np.einsum('abc,ak,bk->ck', R, U_sub, V_sub)
                
                U_sub -= lr * grad_u
                V_sub -= lr * grad_v
                W_sub -= lr * grad_w
                
            if found:
                break
        
        if found:
            # Construct full 3x3x8 decomposition
            rank_full = 2 * r_sub
            U_full = np.zeros((n * m, rank_full))
            V_full = np.zeros((m * p, rank_full))
            W_full = np.zeros((p * n, rank_full))
            
            # First half (k=0..3)
            for c in range(r_sub):
                # U
                U_full[:, c] = U_sub[:, c]
                # V
                for j in range(m_sub):
                    for k in range(p_sub):
                        V_full[j * p + k, c] = V_sub[j * p_sub + k, c]
                # W
                for k in range(p_sub):
                    for i in range(n_sub):
                        W_full[k * n + i, c] = W_sub[k * n_sub + i, c]
                        
            # Second half (k=4..7)
            for c in range(r_sub):
                c2 = c + r_sub
                # U
                U_full[:, c2] = U_sub[:, c]
                # V
                for j in range(m_sub):
                    for k in range(p_sub):
                        # k in 0..3 maps to k+4 in 4..7
                        V_full[j * p + (k + p_sub), c2] = V_sub[j * p_sub + k, c]
                # W
                for k in range(p_sub):
                    for i in range(n_sub):
                        W_full[(k + p_sub) * n + i, c2] = W_sub[k * n_sub + i, c]
                        
            # Convert to list of lists
            U_list = U_full.tolist()
            V_list = V_full.tolist()
            W_list = W_full.tolist()
            
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = rank_full
            break # Found a better rank, stop searching for lower
            
    return best_result
