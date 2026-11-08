import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Construct the target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a = i * m + j
                idx_b = j * p + k
                idx_c = k * n + i
                T[idx_a, idx_b, idx_c] = 1.0
                
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
                
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    
    start_time = time.time()
    time_limit = 145.0
    
    min_rank = max(n*m, m*p, p*n)
    
    for r in range(min_rank, rank_school):
        if time.time() - start_time > time_limit:
            break
            
        max_restarts = 100
        found = False
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            np.random.seed(1000 + r * 100 + restart)
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            max_iters = 50
            prev_err = np.inf
            tol = 1e-10
            
            for it in range(max_iters):
                # Update U
                T_mat = T.reshape(n*m, (m*p)*(p*n))
                Z = np.einsum('jk,ik->jki', V, W).reshape((m*p)*(p*n), r)
                U_new, _, _, _ = np.linalg.lstsq(Z.T, T_mat.T, rcond=None)
                U = U_new.T
                
                # Update V
                T_mat_v = np.transpose(T, (1, 0, 2)).reshape(m*p, (n*m)*(p*n))
                Y = np.einsum('ik,jk->ijk', U, W).reshape((n*m)*(p*n), r)
                V_new, _, _, _ = np.linalg.lstsq(Y.T, T_mat_v.T, rcond=None)
                V = V_new.T
                
                # Update W
                T_mat_w = np.transpose(T, (2, 0, 1)).reshape(p*n, (n*m)*(m*p))
                X = np.einsum('ik,jk->ijk', U, V).reshape((n*m)*(m*p), r)
                W_new, _, _, _ = np.linalg.lstsq(X.T, T_mat_w.T, rcond=None)
                W = W_new.T
                
                T_est = np.einsum('ik,jk,lk->ijl', U, V, W)
                err = np.linalg.norm(T - T_est)
                
                if abs(prev_err - err) < tol:
                    break
                prev_err = err
            
            # Check rounding
            U2 = U * 2
            V2 = V * 2
            W2 = W * 2
            
            U_round = np.round(U2)
            V_round = np.round(V2)
            W_round = np.round(W2)
            
            dist = np.linalg.norm(U2 - U_round) + np.linalg.norm(V2 - V_round) + np.linalg.norm(W2 - W_round)
            
            if dist < 1e-3:
                U_h = U_round / 2.0
                V_h = V_round / 2.0
                W_h = W_round / 2.0
                
                T_check = np.einsum('ik,jk,lk->ijl', U_h, V_h, W_h)
                if np.linalg.norm(T - T_check) < 1e-9:
                    return {
                        "U": U_h.tolist(),
                        "V": V_h.tolist(),
                        "W": W_h.tolist()
                    }

    return best_result
