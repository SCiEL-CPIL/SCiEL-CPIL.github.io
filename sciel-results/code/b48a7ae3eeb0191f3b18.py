import time
import numpy as np

def construct():
    n, m, p = 2, 3, 7
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Construct Schoolbook Decomposition (Rank 42)
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_schoolbook

    # Pre-compute target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def run_als(U, V, W, T, max_iters=100):
        A, B, C = T.shape
        R = U.shape[1]
        for _ in range(max_iters):
            # Update U
            for a in range(A):
                D_a = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(-1, R)
                t_a = T[a].reshape(-1)
                u_a, _, _, _ = np.linalg.lstsq(D_a, t_a, rcond=None)
                U[a, :] = u_a
            # Update V
            for b in range(B):
                t_b = T[:, b, :].reshape(-1)
                D_b = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(-1, R)
                v_b, _, _, _ = np.linalg.lstsq(D_b, t_b, rcond=None)
                V[b, :] = v_b
            # Update W
            for c in range(C):
                t_c = T[:, :, c].reshape(-1)
                D_c = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(-1, R)
                w_c, _, _, _ = np.linalg.lstsq(D_c, t_c, rcond=None)
                W[c, :] = w_c
                
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.linalg.norm(T_rec - T)

    # Search for lower ranks
    # Try ranks from 32 to 41
    for target_rank in range(32, 42):
        if time.time() > deadline - 10:
            break
            
        # Try a few restarts for this rank
        max_restarts = 5
        for attempt in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            U_rand = np.random.randn(n * m, target_rank)
            V_rand = np.random.randn(m * p, target_rank)
            W_rand = np.random.randn(p * n, target_rank)
            
            err = run_als(U_rand, V_rand, W_rand, T_target, max_iters=50)
            
            if err < 1e-8:
                # Check half-integer constraint
                U_half = 2 * U_rand
                V_half = 2 * V_rand
                W_half = 2 * W_rand
                
                U_rounded = np.round(U_half)
                V_rounded = np.round(V_half)
                W_rounded = np.round(W_half)
                
                if np.max(np.abs(U_half - U_rounded)) < 1e-6 and \
                   np.max(np.abs(V_half - V_rounded)) < 1e-6 and \
                   np.max(np.abs(W_half - W_rounded)) < 1e-6:
                   
                    U_final = U_rounded / 2.0
                    V_final = V_rounded / 2.0
                    W_final = W_rounded / 2.0
                    
                    T_rec = np.einsum('ar,br,cr->abc', U_final, V_final, W_final)
                    if np.allclose(T_rec, T_target, atol=1e-9):
                        return {
                            "U": U_final.tolist(),
                            "V": V_final.tolist(),
                            "W": W_final.tolist()
                        }

    return best_result
