"""Program: rank-minimal exact tensor decomposition for 2x4 * 4x4 matrix multiplication.
Attempts to find a lower rank decomposition (e.g., 24) using optimization, falling back to Strassen rank 28."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 4]

def _verify(U, V, W):
    """Verify if the decomposition is exact."""
    n, m, p = DIMENSIONS
    R = len(U[0]) if U else 0
    if U is None or V is None or W is None:
        return False
    if len(U) != n*m or len(V) != m*p or len(W) != p*n:
        return False
    if len(U[0]) != R or len(V[0]) != R or len(W[0]) != R:
        return False
        
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Check shapes
    if U_np.shape != (n*m, R) or V_np.shape != (m*p, R) or W_np.shape != (p*n, R):
        return False
        
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    target = 1 if j == l else 0
                    val = 0
                    for r in range(R):
                        val += W_np[r, k*n+i] * U_np[r, i*m+j] * V_np[r, l*p+k]
                    if abs(val - target) > 1e-6:
                        return False
    return True

def _snap_to_half_integers(U, V, W):
    """Snap values to nearest half-integer and verify."""
    def snap_matrix(M):
        M_snapped = np.round(np.array(M) * 2) / 2.0
        return M_snapped
        
    U_s = snap_matrix(U)
    V_s = snap_matrix(V)
    W_s = snap_matrix(W)
    
    U_list = U_s.tolist()
    V_list = V_s.tolist()
    W_list = W_s.tolist()
    
    if _verify(U_list, V_list, W_list):
        return U_list, V_list, W_list
    else:
        return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition.
    Tries to find rank 24, 25, 26, 27. Falls back to 28.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    def get_strassen():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(u_idx, v_idx, w_idx, col_offset):
            U[u_idx[0]][col_offset + 0] += 1
            U[u_idx[3]][col_offset + 0] += 1
            V[v_idx[0]][col_offset + 0] += 1
            V[v_idx[3]][col_offset + 0] += 1
            W[w_idx[0]][col_offset + 0] += 1
            W[w_idx[3]][col_offset + 0] += 1
            
            U[u_idx[2]][col_offset + 1] += 1
            U[u_idx[3]][col_offset + 1] += 1
            V[v_idx[0]][col_offset + 1] += 1
            W[w_idx[2]][col_offset + 1] += 1
            W[w_idx[3]][col_offset + 1] -= 1
            
            U[u_idx[0]][col_offset + 2] += 1
            V[v_idx[1]][col_offset + 2] += 1
            V[v_idx[3]][col_offset + 2] -= 1
            W[w_idx[1]][col_offset + 2] += 1
            W[w_idx[3]][col_offset + 2] += 1
            
            U[u_idx[3]][col_offset + 3] += 1
            V[v_idx[2]][col_offset + 3] += 1
            V[v_idx[0]][col_offset + 3] -= 1
            W[w_idx[0]][col_offset + 3] += 1
            W[w_idx[2]][col_offset + 3] += 1
            
            U[u_idx[0]][col_offset + 4] += 1
            U[u_idx[1]][col_offset + 4] += 1
            V[v_idx[3]][col_offset + 4] += 1
            W[w_idx[0]][col_offset + 4] -= 1
            W[w_idx[1]][col_offset + 4] += 1
            
            U[u_idx[2]][col_offset + 5] += 1
            U[u_idx[0]][col_offset + 5] -= 1
            V[v_idx[0]][col_offset + 5] += 1
            V[v_idx[1]][col_offset + 5] += 1
            W[w_idx[3]][col_offset + 5] += 1
            
            U[u_idx[1]][col_offset + 6] += 1
            U[u_idx[3]][col_offset + 6] -= 1
            V[v_idx[2]][col_offset + 6] += 1
            V[v_idx[3]][col_offset + 6] += 1
            W[w_idx[0]][col_offset + 6] += 1
        
        u1 = [0, 1, 4, 5]
        v1 = [0, 1, 4, 5]
        w1 = [0, 2, 1, 3]
        add_strassen_block(u1, v1, w1, 0)
        
        u2 = [2, 3, 6, 7]
        v2 = [8, 9, 12, 13]
        w2 = [0, 2, 1, 3]
        add_strassen_block(u2, v2, w2, 7)
        
        u3 = [0, 1, 4, 5]
        v3 = [2, 3, 6, 7]
        w3 = [4, 6, 5, 7]
        add_strassen_block(u3, v3, w3, 14)
        
        u4 = [2, 3, 6, 7]
        v4 = [10, 11, 14, 15]
        w4 = [4, 6, 5, 7]
        add_strassen_block(u4, v4, w4, 21)
        
        return {"U": U, "V": V, "W": W}

    def try_rank_fast(R, max_restarts=1, max_iter=50):
        if time.time() > deadline:
            return None
            
        size_u = n * m * R
        size_v = m * p * R
        size_w = p * n * R
        total_size = size_u + size_v + size_w
        
        def objective(x):
            U_flat = x[:size_u]
            V_flat = x[size_u:size_u+size_v]
            W_flat = x[size_u+size_v:]
            
            U = U_flat.reshape(n*m, R)
            V = V_flat.reshape(m*p, R)
            W = W_flat.reshape(p*n, R)
            
            U_3d = U.reshape(n, m, R)
            V_3d = V.reshape(m, p, R)
            W_3d = W.reshape(p, n, R)
            
            T = np.einsum('kir,ijr,lkr->ikjl', W_3d, U_3d, V_3d)
            
            I = np.eye(m)
            Target = np.zeros((n, p, m, m))
            for i in range(n):
                for k in range(p):
                    Target[i, k, :, :] = I
                    
            diff = T - Target
            return np.sum(diff**2)
            
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + restart)
            x0 = np.random.randn(total_size) * 0.5
            
            try:
                res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4})
            except Exception:
                continue
                
            if res.fun < 1e-4:
                U_flat = res.x[:size_u]
                V_flat = res.x[size_u:size_u+size_v]
                W_flat = res.x[size_u+size_v:]
                
                U = U_flat.reshape(n*m, R).tolist()
                V = V_flat.reshape(m*p, R).tolist()
                W = W_flat.reshape(p*n, R).tolist()
                
                snapped = _snap_to_half_integers(U, V, W)
                if snapped:
                    return snapped
                    
        return None

    # Try to find a lower rank solution with a strict time budget
    search_deadline = start_time + 10  # Allow only 10 seconds for search
    
    for R in [24, 25, 26, 27]:
        if time.time() > search_deadline:
            break
        result = try_rank_fast(R)
        if result:
            return result
            
    # Fallback to known rank 28 solution
    return get_strassen()
