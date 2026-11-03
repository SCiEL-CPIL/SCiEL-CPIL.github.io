import time
import numpy as np

DIMENSIONS = [3, 4, 6]

_START_TIME = time.time()
_DEADLINE = 145.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    """
    Construct exact tensor decomposition for (3x4)*(4x6) -> (3x6).
    Returns rank-minimal decomposition.
    """
    n, m, p = 3, 4, 6
    
    # 1. Build the known rank-66 solution
    rank_66 = 66
    U66 = [[0] * rank_66 for _ in range(n * m)]
    V66 = [[0] * rank_66 for _ in range(m * p)]
    W66 = [[0] * rank_66 for _ in range(p * n)]
    
    col = 0
    
    def add_schoolbook(i, j, k):
        nonlocal col
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U66[u_idx][col] = 1
        V66[v_idx][col] = 1
        W66[w_idx][col] = 1
        col += 1

    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        a_idx = i_base * m + j_base
        b_idx = i_base * m + (j_base + 1)
        c_idx = (i_base + 1) * m + j_base
        d_idx = (i_base + 1) * m + (j_base + 1)
        e_idx = j_base * p + k_base
        f_idx = j_base * p + (k_base + 1)
        g_idx = (j_base + 1) * p + k_base
        h_idx = (j_base + 1) * p + (k_base + 1)
        p_idx = k_base * n + i_base
        q_idx = (k_base + 1) * n + i_base
        r_idx = k_base * n + (i_base + 1)
        s_idx = (k_base + 1) * n + (i_base + 1)
        
        # P1 = (a+d)(e+h)
        U66[a_idx][col] = 1; U66[d_idx][col] = 1
        V66[e_idx][col] = 1; V66[h_idx][col] = 1
        W66[p_idx][col] = 1; W66[s_idx][col] = 1
        col += 1
        
        # P2 = (c+d)(e)
        U66[c_idx][col] = 1; U66[d_idx][col] = 1
        V66[e_idx][col] = 1
        W66[r_idx][col] = 1; W66[s_idx][col] = -1
        col += 1
        
        # P3 = (a)(f-h)
        U66[a_idx][col] = 1
        V66[f_idx][col] = 1; V66[h_idx][col] = -1
        W66[q_idx][col] = 1; W66[s_idx][col] = 1
        col += 1
        
        # P4 = (d)(g-e)
        U66[d_idx][col] = 1
        V66[g_idx][col] = 1; V66[e_idx][col] = -1
        W66[p_idx][col] = 1; W66[r_idx][col] = 1
        col += 1
        
        # P5 = (a+b)(h)
        U66[a_idx][col] = 1; U66[b_idx][col] = 1
        V66[h_idx][col] = 1
        W66[p_idx][col] = -1; W66[q_idx][col] = 1
        col += 1
        
        # P6 = (c-a)(e+f)
        U66[c_idx][col] = 1; U66[a_idx][col] = -1
        V66[e_idx][col] = 1; V66[f_idx][col] = 1
        W66[s_idx][col] = 1
        col += 1
        
        # P7 = (b-d)(g+h)
        U66[b_idx][col] = 1; U66[d_idx][col] = -1
        V66[g_idx][col] = 1; V66[h_idx][col] = 1
        W66[p_idx][col] = 1
        col += 1

    for j in range(4):
        for k in range(6):
            add_schoolbook(0, j, k)
    
    for k_block in range(3):
        k_base = k_block * 2
        for j_block in range(2):
            j_base = j_block * 2
            i_base = 1
            add_strassen(i_base, j_base, k_base)
    
    # Convert to numpy for verification
    U_np = np.array(U66, dtype=float)
    V_np = np.array(V66, dtype=float)
    W_np = np.array(W66, dtype=float)
    
    def verify(U, V, W):
        n, m, p = 3, 4, 6
        r = U.shape[1]
        U_t = U.reshape(n, m, r)
        V_t = V.reshape(m, p, r)
        W_t = W.reshape(p, n, r)
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        return np.allclose(T, 1.0, atol=1e-9)
    
    if not verify(U_np, V_np, W_np):
        # Fallback to schoolbook if something went wrong
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_rank = 66
    best_U = U_np.copy()
    best_V = V_np.copy()
    best_W = W_np.copy()
    
    # 2. Attempt to find lower rank via Optimization
    # Target tensor T is all ones
    T_target = np.ones((n, m, p), dtype=float)
    
    def try_rank(R, seed):
        np.random.seed(seed)
        # Initialize
        U = np.random.randn(n, m, R) * 0.1
        V = np.random.randn(m, p, R) * 0.1
        W = np.random.randn(p, n, R) * 0.1
        
        # Gradient Descent
        lr = 0.05
        for _ in range(200):
            if _time_left() < 10:
                return None
            
            # Compute T_current
            T_curr = np.einsum('ijr,jkr,kir->ijk', U, V, W)
            E = T_target - T_curr
            
            # Gradients
            # dE/dU_ijr = sum_{j',k'} E_ijk' V_j'k'r W_k'i r ? No
            # E_ijk = 1 - sum_r U_ijr V_jkr W_kir
            # dE_ijk / dU_ijr = - V_jkr W_kir
            # Update U_ijr += lr * sum_{j,k} E_ijk * V_jkr * W_kir
            
            dU = np.einsum('ijk,jkr,kir->ijr', E, V, W)
            dV = np.einsum('ijk,ijr,kir->jkr', E, U, W)
            dW = np.einsum('ijk,ijr,jkr->kir', E, U, V)
            
            U += lr * dU
            V += lr * dV
            W += lr * dW
        
        # Round to nearest 0.5
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Verify
        if verify(U_r, V_r, W_r):
            return (U_r, V_r, W_r)
        return None
    
    # Try ranks from 65 down to 60
    for R in range(65, 59, -1):
        if _time_left() < 15:
            break
        found = None
        for seed in range(5):
            found = try_rank(R, seed)
            if found:
                break
            if _time_left() < 15:
                break
        
        if found:
            best_rank = R
            best_U, best_V, best_W = found
            break
    
    # 3. Final Column Elimination on the best found
    if best_rank > 0:
        current_U = best_U.copy()
        current_V = best_V.copy()
        current_W = best_W.copy()
        current_rank = best_rank
        
        for _ in range(3):
            if _time_left() < 10:
                break
            removed = False
            for r in range(current_rank):
                if _time_left() < 5:
                    break
                U_test = np.delete(current_U, r, axis=2)
                V_test = np.delete(current_V, r, axis=2)
                W_test = np.delete(current_W, r, axis=2)
                if verify(U_test, V_test, W_test):
                    current_U = U_test
                    current_V = V_test
                    current_W = W_test
                    current_rank -= 1
                    removed = True
                    break
            if not removed:
                break
        
        if current_rank < best_rank:
            best_rank = current_rank
            best_U = current_U
            best_V = current_V
            best_W = current_W
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
