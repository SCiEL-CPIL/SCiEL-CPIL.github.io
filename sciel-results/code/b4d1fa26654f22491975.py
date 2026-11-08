import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    
    # Rank 42 solution based on 6 blocks of Strassen 2x2x2
    rank_base = 42
    U_base = np.zeros((n * m, rank_base), dtype=int)
    V_base = np.zeros((m * p, rank_base), dtype=int)
    W_base = np.zeros((p * n, rank_base), dtype=int)
    
    # Strassen 2x2x2 coefficients
    U_s = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,1,1), (1,1,-1)]
    ]
    V_s = [
        [(0,0,1), (1,1,1)],
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (0,0,-1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (1,1,1)]
    ]
    W_s = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (1,1,1)],
        [(0,0,1), (1,0,1)],
        [(0,0,-1), (0,1,1)],
        [(1,1,1)],
        [(0,0,1)]
    ]
    
    col = 0
    for a_block in range(2):
        for b_block in range(3):
            for s in range(7):
                for (i_loc, j_loc, coeff) in U_s[s]:
                    i = i_loc
                    j = a_block * 2 + j_loc
                    U_base[i * m + j, col] += coeff
                for (j_loc, k_loc, coeff) in V_s[s]:
                    j = a_block * 2 + j_loc
                    k = b_block * 2 + k_loc
                    V_base[j * p + k, col] += coeff
                for (i_loc, k_loc, coeff) in W_s[s]:
                    i = i_loc
                    k = b_block * 2 + k_loc
                    W_base[k * n + i, col] += coeff
                col += 1

    # Attempt to find a rank 40 solution (rank 20 for 2x2x6 sub-problem)
    # Sub-problem: A (2x2), B (2x6), C (2x6)
    n_sub, m_sub, p_sub = 2, 2, 6
    T = np.zeros((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub))
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                a_idx = i * m_sub + j
                b_idx = j * p_sub + k
                c_idx = k * n_sub + i
                T[a_idx, b_idx, c_idx] = 1.0
                
    rank_sub = 20
    np.random.seed(42)
    U_sub = np.random.randn(n_sub * m_sub, rank_sub) / np.sqrt(rank_sub)
    V_sub = np.random.randn(m_sub * p_sub, rank_sub) / np.sqrt(rank_sub)
    W_sub = np.random.randn(p_sub * n_sub, rank_sub) / np.sqrt(rank_sub)
    
    # ALS optimization
    best_error = np.inf
    for it in range(200):
        if time.time() > deadline - 10:
            break
            
        # Update U
        # T = sum U_s V_s^T W_s^T? No, T_ijk = sum U_is V_js W_ks
        # Reshape T to (I, J*K)
        T_ik = T.reshape(n_sub * m_sub, -1)
        # V W^T is (J, K) -> (J*K,)? No.
        # For fixed V, W, U = T_ik (V W^T)^+ ?
        # Let M = V @ W.T -> (J, K)
        # T_ik = U @ M^T ? No.
        # T_ijk = sum_s U_is V_js W_ks
        # Let's vectorize J and K.
        # T_ik (I, J*K) = U (I, R) @ (V (J,R) @ W.T (R,K) -> M (J,K) -> M.T (K,J) -> reshape?)
        # Actually, (V @ W.T) is (J, K).
        # We want U such that U @ (V @ W.T).T = T_ik?
        # (V @ W.T).T = W @ V.T (K, J).
        # Reshape W @ V.T to (J*K, 1)? No.
        # Let's use the property: T_ik = sum_s U_is (V_js W_ks)
        # Let M_jk = sum_s V_js W_ks. M is (J, K).
        # Then T_ik = sum_j,k T_ijk delta? No.
        # T_ik (flattened J,K) = U @ M_flattened?
        # M is (J, K). Flatten to (J*K,).
        # Then T_ik (I, J*K) = U (I, R) @ M_flat (R, J*K)?
        # No, M depends on s.
        # M_s is (J, K).
        # T_ik = sum_s U_is M_s (flattened to J*K)
        # So T_ik = U @ M_all.T where M_all is (R, J*K).
        # M_all[s, j*p_sub+k] = V_sub[j, s] * W_sub[k, s]
        
        M_all = np.zeros((rank_sub, m_sub * p_sub))
        for s in range(rank_sub):
            M_s = np.outer(V_sub[:, s], W_sub[:, s]) # (J, K)
            M_all[s, :] = M_s.flatten()
            
        # Solve U M_all.T = T_ik
        # U = T_ik @ pinv(M_all.T)
        U_sub = T_ik @ np.linalg.pinv(M_all.T)
        
        # Update V
        # T_jk (J, I*K) = V (J, R) @ (U W^T) (R, I*K)
        T_jk = T.transpose(1, 0, 2).reshape(m_sub * p_sub, -1) # (J, I*K)
        N_all = np.zeros((rank_sub, n_sub * p_sub)) # (R, I*K)
        for s in range(rank_sub):
            N_s = np.outer(U_sub[:, s], W_sub[:, s]) # (I, K)
            N_all[s, :] = N_s.flatten()
        V_sub = T_jk @ np.linalg.pinv(N_all.T)
        
        # Update W
        # T_ki (K, I*J) = W (K, R) @ (U V^T) (R, I*J)
        T_ki = T.transpose(2, 0, 1).reshape(p_sub * n_sub, -1) # (K, I*J)
        P_all = np.zeros((rank_sub, n_sub * m_sub)) # (R, I*J)
        for s in range(rank_sub):
            P_s = np.outer(U_sub[:, s], V_sub[:, s]) # (I, J)
            P_all[s, :] = P_s.flatten()
        W_sub = T_ki @ np.linalg.pinv(P_all.T)
        
        # Calculate error
        # T_approx = sum U_s V_s W_s
        # Error = || T - T_approx ||_F
        # We can compute T_approx using the current factors
        # But it's expensive. We can estimate from the residual of the last step?
        # Or just compute it every 10 steps.
        if it % 10 == 0:
            T_approx = np.zeros_like(T)
            for s in range(rank_sub):
                T_approx += np.einsum('i,j,k->ijk', U_sub[:, s], V_sub[:, s], W_sub[:, s])
            error = np.linalg.norm(T - T_approx)
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U_sub.copy(), V_sub.copy(), W_sub.copy()
                
    if best_error < 1e-5:
        # Check if we can round to integers/half-integers
        # Scale and round
        # The entries of T are 1. The factors should be O(1).
        # Try rounding to nearest 0.5
        U_int = np.round(best_U * 2) / 2
        V_int = np.round(best_V * 2) / 2
        W_int = np.round(best_W * 2) / 2
        
        # Verify integer solution
        T_approx_int = np.zeros_like(T)
        for s in range(rank_sub):
            T_approx_int += np.einsum('i,j,k->ijk', U_int[:, s], V_int[:, s], W_int[:, s])
        error_int = np.linalg.norm(T - T_approx_int)
        
        if error_int < 1e-6:
            # Construct full rank 40 solution
            rank_final = 40
            U_final = np.zeros((n * m, rank_final), dtype=float)
            V_final = np.zeros((m * p, rank_final), dtype=float)
            W_final = np.zeros((p * n, rank_final), dtype=float)
            
            # A_L B_T (rank 20)
            for s in range(rank_sub):
                for i in range(n_sub):
                    for j in range(m_sub):
                        U_final[i * m + j, s] = U_int[i * m_sub + j, s]
                for j in range(m_sub):
                    for k in range(p_sub):
                        V_final[j * p + k, s] = V_int[j * p_sub + k, s]
                for k in range(p_sub):
                    for i in range(n_sub):
                        W_final[k * n + i, s] = W_int[k * n_sub + i, s]
                        
            # A_R B_B (rank 20)
            for s in range(rank_sub):
                col = rank_sub + s
                for i in range(n_sub):
                    for j in range(m_sub):
                        U_final[i * m + (m_sub + j), col] = U_int[i * m_sub + j, s]
                for j in range(m_sub):
                    for k in range(p_sub):
                        V_final[(m_sub + j) * p + k, col] = V_int[j * p_sub + k, s]
                for k in range(p_sub):
                    for i in range(n_sub):
                        W_final[k * n + i, col] = W_int[k * n_sub + i, s]
                        
            return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
            
    return {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}
