import numpy as np
import time

DIMENSIONS = [2, 2, 5]

def _build_target(n, m, p):
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1
    return T

def _check_decomposition(U, V, W, T):
    U_a = np.array(U, dtype=np.float64)
    V_a = np.array(V, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    
    # Efficient check using einsum
    # T_approx = sum_r U[:,r] V[:,r] W[:,r]
    # This is a rank-R tensor.
    # We can compute the difference.
    diff = np.zeros_like(T_a)
    for r in range(U_a.shape[1]):
        diff += np.einsum('u,v,w->uvw', U_a[:,r], V_a[:,r], W_a[:,r])
    return np.allclose(diff, T_a, atol=1e-9)

def _als_solve_U(T, V, W, R):
    V_a = np.array(V, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    
    # VvW shape (m*p, p*n, R)
    VvW = np.einsum('vr,wr->vwr', V_a, W_a)
    VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
    
    n_m = T_a.shape[0]
    U = np.zeros((n_m, R), dtype=np.float64)
    for u in range(n_m):
        b = T_a[u].reshape(-1)
        sol, res, rank_s, sv = np.linalg.lstsq(VvW_flat, b, rcond=None)
        U[u] = sol
    return U

def _als_solve_V(T, U, W, R):
    U_a = np.array(U, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    
    Uw = np.einsum('ur,wr->uwr', U_a, W_a)
    Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
    
    m_p = T_a.shape[1]
    V = np.zeros((m_p, R), dtype=np.float64)
    for v in range(m_p):
        b = T_a[:, v, :].reshape(-1)
        sol, res, rank_s, sv = np.linalg.lstsq(Uw_flat, b, rcond=None)
        V[v] = sol
    return V

def _als_solve_W(T, U, V, R):
    U_a = np.array(U, dtype=np.float64)
    V_a = np.array(V, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    
    UV = np.einsum('ur,vr->uvr', U_a, V_a)
    UV_flat = UV.reshape(-1, R)
    
    p_n = T_a.shape[2]
    W = np.zeros((p_n, R), dtype=np.float64)
    for w in range(p_n):
        b = T_a[:, :, w].reshape(-1)
        sol, res, rank_s, sv = np.linalg.lstsq(UV_flat, b, rcond=None)
        W[w] = sol
    return W

def _find_rank_decomposition(n, m, p, R, max_iters=200, n_restarts=10, deadline=None):
    T = _build_target(n, m, p)
    T_a = np.array(T, dtype=np.float64)
    
    for restart in range(n_restarts):
        if deadline and time.time() > deadline:
            break
        np.random.seed(42 + restart)
        U = np.random.randn(n*m, R) * 0.5
        V = np.random.randn(m*p, R) * 0.5
        W = np.random.randn(p*n, R) * 0.5
        
        prev_err = np.inf
        for it in range(max_iters):
            if deadline and time.time() > deadline:
                break
            U = _als_solve_U(T, V, W, R)
            V = _als_solve_V(T, U, W, R)
            W = _als_solve_W(T, U, V, R)
            
            # Calculate error
            diff = np.zeros_like(T_a)
            for r in range(R):
                diff += np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r])
            err = np.linalg.norm(diff - T_a)
            
            if abs(prev_err - err) < 1e-12:
                break
            prev_err = err
        
        if _check_decomposition(U, V, W, T):
            # Try to snap to half-integers
            for denom in [2, 4, 8]:
                U_s = np.round(U * denom) / denom
                V_s = np.round(V * denom) / denom
                W_s = np.round(W * denom) / denom
                if _check_decomposition(U_s, V_s, W_s, T):
                    return U_s.tolist(), V_s.tolist(), W_s.tolist()
            return U.tolist(), V.tolist(), W.tolist()
    return None

def _build_rank18(n, m, p):
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1
        W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
        
        r = base + 2
        U[0][r] = 1
        V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
        W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 3
        U[3][r] = 1
        V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
        W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
        
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[1*p+col1][r] = 1
        W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
        
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
        W[col1*n+1][r] = 1
        
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1
    
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    # Schoolbook for column 4
    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try to construct rank 17: Rank 7 (2x2x2) + Rank 10 (2x2x3)
    # 1. Get Rank 7 for 2x2x2
    U7, V7, W7 = _build_rank18(n, m, 2) # Wait, _build_rank18 assumes p=5? No, it takes p.
    # But _build_rank18 is hardcoded for p=5 in indices?
    # Let's write a specific rank 7 builder for 2x2x2.
    
    # Rank 7 for 2x2x2
    U_222 = [[0]*7 for _ in range(4)]
    V_222 = [[0]*7 for _ in range(4)] # m*p = 2*2 = 4
    W_222 = [[0]*7 for _ in range(4)] # p*n = 2*2 = 4
    
    # Standard Strassen
    U_222[0][0] = 1; U_222[3][0] = 1
    V_222[0][0] = 1; V_222[3][0] = 1
    W_222[0][0] = 1; W_222[3][0] = 1
    
    U_222[2][1] = 1; U_222[3][1] = 1
    V_222[0][1] = 1
    W_222[1][1] = 1; W_222[3][1] = -1
    
    U_222[0][2] = 1
    V_222[1][2] = 1; V_222[3][2] = -1
    W_222[2][2] = 1; W_222[3][2] = 1
    
    U_222[3][3] = 1
    V_222[2][3] = 1; V_222[0][3] = -1
    W_222[0][3] = 1; W_222[1][3] = 1
    
    U_222[0][4] = 1; U_222[1][4] = 1
    V_222[3][4] = 1
    W_222[0][4] = -1; W_222[2][4] = 1
    
    U_222[2][5] = 1; U_222[0][5] = -1
    V_222[0][5] = 1; V_222[1][5] = 1
    W_222[3][5] = 1
    
    U_222[1][6] = 1; U_222[3][6] = -1
    V_222[2][6] = 1; V_222[3][6] = 1
    W_222[0][6] = 1
    
    # 2. Find Rank 10 for 2x2x3
    U_223, V_223, W_223 = _find_rank_decomposition(2, 2, 3, 10, max_iters=100, n_restarts=20, deadline=deadline)
    
    if U_223 is not None:
        # Combine
        R_total = 17
        U_full = [[0]*R_total for _ in range(4)]
        V_full = [[0]*R_total for _ in range(10)]
        W_full = [[0]*R_total for _ in range(10)]
        
        # Copy 222 part (cols 0,1)
        for i in range(4):
            for r in range(7):
                U_full[i][r] = U_222[i][r]
        
        # V_222 indices: 0,1,2,3 correspond to global 0,1,5,6
        v_map_222 = [0, 1, 5, 6]
        for i in range(4):
            for r in range(7):
                V_full[v_map_222[i]][r] = V_222[i][r]
        
        # W_222 indices: 0,1,2,3 correspond to global 0,1,2,3
        for i in range(4):
            for r in range(7):
                W_full[i][r] = W_222[i][r]
        
        # Copy 223 part (cols 2,3,4)
        for i in range(4):
            for r in range(7, 17):
                U_full[i][r] = U_223[i][r-7]
        
        # V_223 indices: 0,1,2,3,4,5 correspond to global 2,3,4,7,8,9
        v_map_223 = [2, 3, 4, 7, 8, 9]
        for i in range(6):
            for r in range(7, 17):
                V_full[v_map_223[i]][r] = V_223[i][r-7]
        
        # W_223 indices: 0,1,2,3,4,5 correspond to global 4,5,6,7,8,9
        # W index k*n+i. For k=2,3,4.
        # k=2: 4,5
        # k=3: 6,7
        # k=4: 8,9
        w_map_223 = [4, 5, 6, 7, 8, 9]
        for i in range(6):
            for r in range(7, 17):
                W_full[w_map_223[i]][r] = W_223[i][r-7]
        
        # Verify
        T_full = _build_target(2, 2, 5)
        if _check_decomposition(U_full, V_full, W_full, T_full):
            return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback to Rank 18
    U18, V18, W18 = _build_rank18(2, 2, 5)
    return {"U": U18, "V": V18, "W": W18}
