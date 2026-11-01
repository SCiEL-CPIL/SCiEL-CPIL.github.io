import time
import sys
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 6]
DEADLINE = 140.0

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook rank
    sb_rank = n * m * p  # 36
    
    # Build schoolbook decomposition as fallback
    U_sb = np.zeros((n*m, sb_rank), dtype=float)
    V_sb = np.zeros((m*p, sb_rank), dtype=float)
    W_sb = np.zeros((p*n, sb_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best_rank = sb_rank
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    # Define the target tensor T for 2x3x6
    T_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_indices.append((i*m+j, j*p+k, k*n+i))
                
    T_a = np.array([t[0] for t in T_indices])
    T_b = np.array([t[1] for t in T_indices])
    T_c = np.array([t[2] for t in T_indices])
    
    def make_residual_func(target_rank):
        u_size = n*m * target_rank
        v_size = m*p * target_rank
        w_size = p*n * target_rank
        
        def res(x):
            U = x[:u_size].reshape(n*m, target_rank)
            V = x[u_size:u_size+v_size].reshape(m*p, target_rank)
            W = x[u_size+v_size:].reshape(p*n, target_rank)
            
            U_a = U[T_a, :]
            V_b = V[T_b, :]
            W_c = W[T_c, :]
            
            vals = np.einsum('ij,ij,ij->i', U_a, V_b, W_c)
            return vals - 1.0
            
        return res
    
    def snap_and_validate(U, V, W, tol=1e-6):
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        U_a = U_snap[T_a, :]
        V_b = V_snap[T_b, :]
        W_c = W_snap[T_c, :]
        vals = np.einsum('ij,ij,ij->i', U_a, V_b, W_c)
        
        if not np.allclose(vals, 1.0, atol=tol):
            return False
        
        # Check non-targets are approximately 0
        np.random.seed(42)
        for _ in range(200):
            a = np.random.randint(0, n*m)
            b = np.random.randint(0, m*p)
            c = np.random.randint(0, p*n)
            if (a, b, c) in T_indices:
                continue
            val = np.dot(U_snap[a, :], V_snap[b, :] * W_snap[c, :])
            if abs(val) > tol:
                return False
        return True
    
    # Strategy: Find rank-11 decomposition for 2x3x3, then use two copies for 2x3x6
    n2, m2, p2 = 2, 3, 3
    rank_sub = 11
    
    T2_indices = []
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T2_indices.append((i*m2+j, j*p2+k, k*n2+i))
                
    T2_a = np.array([t[0] for t in T2_indices])
    T2_b = np.array([t[1] for t in T2_indices])
    T2_c = np.array([t[2] for t in T2_indices])
    
    def make_residual_func_2x3x3(target_rank):
        u_size = n2*m2 * target_rank
        v_size = m2*p2 * target_rank
        w_size = p2*n2 * target_rank
        
        def res(x):
            U = x[:u_size].reshape(n2*m2, target_rank)
            V = x[u_size:u_size+v_size].reshape(m2*p2, target_rank)
            W = x[u_size+v_size:].reshape(p2*n2, target_rank)
            
            U_a = U[T2_a, :]
            V_b = V[T2_b, :]
            W_c = W[T2_c, :]
            
            vals = np.einsum('ij,ij,ij->i', U_a, V_b, W_c)
            return vals - 1.0
            
        return res
    
    def snap_and_validate_2x3x3(U, V, W, tol=1e-6):
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        U_a = U_snap[T2_a, :]
        V_b = V_snap[T2_b, :]
        W_c = W_snap[T2_c, :]
        vals = np.einsum('ij,ij,ij->i', U_a, V_b, W_c)
        
        if not np.allclose(vals, 1.0, atol=tol):
            return False
        
        np.random.seed(42)
        for _ in range(200):
            a = np.random.randint(0, n2*m2)
            b = np.random.randint(0, m2*p2)
            c = np.random.randint(0, p2*n2)
            if (a, b, c) in T2_indices:
                continue
            val = np.dot(U_snap[a, :], V_snap[b, :] * W_snap[c, :])
            if abs(val) > tol:
                return False
        return True
    
    np.random.seed(42)
    
    U_sub = None
    V_sub = None
    W_sub = None
    
    res_func_sub = make_residual_func_2x3x3(rank_sub)
    u_size_sub = n2*m2 * rank_sub
    v_size_sub = m2*p2 * rank_sub
    w_size_sub = p2*n2 * rank_sub
    
    found_sub = False
    for restart in range(50):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        if restart == 0:
            U_init = np.zeros((n2*m2, rank_sub))
            V_init = np.zeros((m2*p2, rank_sub))
            W_init = np.zeros((p2*n2, rank_sub))
            col = 0
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        if col >= rank_sub:
                            break
                        U_init[i*m2+j, col] = 1.0
                        V_init[j*p2+k, col] = 1.0
                        W_init[k*n2+i, col] = 1.0
                        col += 1
                    if col >= rank_sub:
                        break
                if col >= rank_sub:
                    break
        else:
            U_init = np.random.randn(n2*m2, rank_sub) * 0.5
            V_init = np.random.randn(m2*p2, rank_sub) * 0.5
            W_init = np.random.randn(p2*n2, rank_sub) * 0.5
        
        x0 = np.concatenate([U_init.ravel(), V_init.ravel(), W_init.ravel()])
        
        try:
            result = least_squares(res_func_sub, x0, method='lm', max_nfev=2000)
            
            U_opt = result.x[:u_size_sub].reshape(n2*m2, rank_sub)
            V_opt = result.x[u_size_sub:u_size_sub+v_size_sub].reshape(m2*p2, rank_sub)
            W_opt = result.x[u_size_sub+v_size_sub:].reshape(p2*n2, rank_sub)
            
            if snap_and_validate_2x3x3(U_opt, V_opt, W_opt):
                U_sub = U_opt
                V_sub = V_opt
                W_sub = W_opt
                found_sub = True
                break
        except Exception:
            continue
    
    if found_sub and time.time() - start_time < DEADLINE - 10:
        target_rank = 2 * rank_sub  # 22
        
        U_full = np.zeros((n*m, target_rank), dtype=float)
        V_full = np.zeros((m*p, target_rank), dtype=float)
        W_full = np.zeros((p*n, target_rank), dtype=float)
        
        for r in range(rank_sub):
            U_full[:, r] = U_sub[:, r]
            for j in range(m2):
                for k in range(p2):
                    V_full[j*p + k, r] = V_sub[j*p2 + k, r]
            for k in range(p2):
                for i in range(n2):
                    W_full[k*n + i, r] = W_sub[k*n2 + i, r]
        
        for r in range(rank_sub):
            r2 = rank_sub + r
            U_full[:, r2] = U_sub[:, r]
            for j in range(m2):
                for k in range(p2):
                    V_full[j*p + (k + p2), r2] = V_sub[j*p2 + k, r]
            for k in range(p2):
                for i in range(n2):
                    W_full[(k + p2)*n + i, r2] = W_sub[k*n2 + i, r]
        
        if snap_and_validate(U_full, V_full, W_full):
            best_rank = target_rank
            best_U = U_full
            best_V = V_full
            best_W = W_full
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
