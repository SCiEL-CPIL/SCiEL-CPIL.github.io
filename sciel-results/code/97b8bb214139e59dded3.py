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
    Returns rank-66 Strassen-based decomposition.
    """
    n, m, p = 3, 4, 6
    rank = 66
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_schoolbook(i, j, k):
        nonlocal col
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
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
        
        # P1 = (a+d)(e+h) -> p+s
        U[a_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        # P2 = (c+d)(e) -> r-s
        U[c_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1
        W[r_idx][col] = 1; W[s_idx][col] = -1
        col += 1
        
        # P3 = (a)(f-h) -> q+s
        U[a_idx][col] = 1
        V[f_idx][col] = 1; V[h_idx][col] = -1
        W[q_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        # P4 = (d)(g-e) -> p+r
        U[d_idx][col] = 1
        V[g_idx][col] = 1; V[e_idx][col] = -1
        W[p_idx][col] = 1; W[r_idx][col] = 1
        col += 1
        
        # P5 = (a+b)(h) -> -p+q
        U[a_idx][col] = 1; U[b_idx][col] = 1
        V[h_idx][col] = 1
        W[p_idx][col] = -1; W[q_idx][col] = 1
        col += 1
        
        # P6 = (c-a)(e+f) -> s
        U[c_idx][col] = 1; U[a_idx][col] = -1
        V[e_idx][col] = 1; V[f_idx][col] = 1
        W[s_idx][col] = 1
        col += 1
        
        # P7 = (b-d)(g+h) -> p
        U[b_idx][col] = 1; U[d_idx][col] = -1
        V[g_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1
        col += 1

    # Build rank-66 solution
    for j in range(4):
        for k in range(6):
            add_schoolbook(0, j, k)
    
    for k_block in range(3):
        k_base = k_block * 2
        for j_block in range(2):
            j_base = j_block * 2
            i_base = 1
            add_strassen(i_base, j_base, k_base)
    
    # Verify correctness
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    def verify_fast(U_np, V_np, W_np):
        n, m, p = 3, 4, 6
        r = U_np.shape[1]
        U_t = U_np.reshape(n, m, r)
        V_t = V_np.reshape(m, p, r)
        W_t = W_np.reshape(p, n, r)
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        return np.allclose(T, 1.0, atol=1e-9)
    
    if not verify_fast(U_np, V_np, W_np):
        # Fallback to schoolbook
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
    
    best_rank = rank
    best_U = U_np.copy()
    best_V = V_np.copy()
    best_W = W_np.copy()
    
    # Fast column elimination
    current_U = U_np.copy()
    current_V = V_np.copy()
    current_W = W_np.copy()
    current_rank = rank
    
    # Precompute tensor for fast updates? 
    # For now, use einsum for removal checks, but limit passes.
    
    max_passes = 5
    for pass_num in range(max_passes):
        if _time_left() < 20:
            break
        
        removed_this_pass = False
        for remove_col in range(current_rank):
            if _time_left() < 15:
                break
            
            U_test = np.delete(current_U, remove_col, axis=1)
            V_test = np.delete(current_V, remove_col, axis=1)
            W_test = np.delete(current_W, remove_col, axis=1)
            
            if verify_fast(U_test, V_test, W_test):
                current_rank -= 1
                current_U = U_test
                current_V = V_test
                current_W = W_test
                removed_this_pass = True
                break
        
        if not removed_this_pass:
            break
    
    if current_rank < best_rank:
        best_rank = current_rank
        best_U = current_U
        best_V = current_V
        best_W = current_W
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
