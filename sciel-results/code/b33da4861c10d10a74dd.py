import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    start_time = time.time()
    timeout = 140.0

    def check_solution(U, V, W, T):
        Uf = np.asarray(U, dtype=np.float64)
        Vf = np.asarray(V, dtype=np.float64)
        Wf = np.asarray(W, dtype=np.float64)
        T_recon = np.einsum('uc,vc,wc->uvw', Uf, Vf, Wf)
        if np.allclose(T_recon, T, atol=1e-9):
            return True
        return False

    def find_strassen_333():
        """
        Find a rank-23 decomposition of the 3x3x3 matrix multiplication tensor.
        Returns U, V, W if found, else None.
        """
        n_s, m_s, p_s = 3, 3, 3
        dim_us = n_s * m_s
        dim_vs = m_s * p_s
        dim_ws = p_s * n_s
        
        # Target tensor for 3x3x3
        T_s = np.zeros((dim_us, dim_vs, dim_ws), dtype=np.float64)
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    u_idx = i * m_s + j
                    v_idx = j * p_s + k
                    w_idx = k * n_s + i
                    T_s[u_idx, v_idx, w_idx] = 1.0
        
        rank_s = 23
        size_U = dim_us * rank_s
        size_V = dim_vs * rank_s
        size_W = dim_ws * rank_s
        total_size = size_U + size_V + size_W
        
        def objective_and_grad(x):
            U_flat = x[:size_U]
            V_flat = x[size_U:size_U + size_V]
            W_flat = x[size_U + size_V:]
            
            U = U_flat.reshape(dim_us, rank_s)
            V = V_flat.reshape(dim_vs, rank_s)
            W = W_flat.reshape(dim_ws, rank_s)
            
            T_rec = np.einsum('uc,vc,wc->uvw', U, V, W)
            err = T_rec - T_s
            loss = np.sum(err ** 2)
            
            grad_U = 2 * np.einsum('uvw,vc,wc->uc', err, V, W)
            grad_V = 2 * np.einsum('uvw,uc,wc->vc', err, U, W)
            grad_W = 2 * np.einsum('uvw,uc,vc->wc', err, U, V)
            
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            return loss, grad

        # Schoolbook initialization for 3x3x3 (rank 27)
        def schoolbook_333():
            U = np.zeros((dim_us, 27))
            V = np.zeros((dim_vs, 27))
            W = np.zeros((dim_ws, 27))
            col = 0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        U[i*3+j, col] = 1
                        V[j*3+k, col] = 1
                        W[k*3+i, col] = 1
                        col += 1
            return U, V, W

        # Try multiple strategies
        best_loss = np.inf
        best_x = None
        
        # Strategy 1: Random initialization with many seeds
        for seed in range(30):
            if time.time() - start_time > timeout - 15:
                break
            
            np.random.seed(seed)
            x0 = np.random.randn(total_size) * 0.5
            
            try:
                res = minimize(
                    objective_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6}
                )
            except Exception:
                continue
            
            if res.fun < best_loss:
                best_loss = res.fun
                best_x = res.x.copy()
            
            if res.fun < 1e-8:
                U_flat = res.x[:size_U]
                V_flat = res.x[size_U:size_U + size_V]
                W_flat = res.x[size_U + size_V:]
                
                U = U_flat.reshape(dim_us, rank_s)
                V = V_flat.reshape(dim_vs, rank_s)
                W = W_flat.reshape(dim_ws, rank_s)
                
                # Snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                if check_solution(U_snap, V_snap, W_snap, T_s):
                    return U_snap, V_snap, W_snap
        
        # Strategy 2: Start from schoolbook and perturb
        if time.time() - start_time < timeout - 15:
            U_sb, V_sb, W_sb = schoolbook_333()
            # Take first 23 columns and perturb
            for seed in range(20):
                if time.time() - start_time > timeout - 15:
                    break
                
                np.random.seed(1000 + seed)
                U_p = U_sb[:, :23].copy() + np.random.randn(dim_us, 23) * 0.3
                V_p = V_sb[:, :23].copy() + np.random.randn(dim_vs, 23) * 0.3
                W_p = W_sb[:, :23].copy() + np.random.randn(dim_ws, 23) * 0.3
                
                x0 = np.concatenate([U_p.flatten(), V_p.flatten(), W_p.flatten()])
                
                try:
                    res = minimize(
                        objective_and_grad,
                        x0,
                        method='L-BFGS-B',
                        jac=True,
                        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6}
                    )
                except Exception:
                    continue
                
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_x = res.x.copy()
                
                if res.fun < 1e-8:
                    U_flat = res.x[:size_U]
                    V_flat = res.x[size_U:size_U + size_V]
                    W_flat = res.x[size_U + size_V:]
                    
                    U = U_flat.reshape(dim_us, rank_s)
                    V = V_flat.reshape(dim_vs, rank_s)
                    W = W_flat.reshape(dim_ws, rank_s)
                    
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    if check_solution(U_snap, V_snap, W_snap, T_s):
                        return U_snap, V_snap, W_snap
        
        # Final polish: if we got close, try to snap
        if best_x is not None and best_loss < 1.0:
            U_flat = best_x[:size_U]
            V_flat = best_x[size_U:size_U + size_V]
            W_flat = best_x[size_U + size_V:]
            
            U = U_flat.reshape(dim_us, rank_s)
            V = V_flat.reshape(dim_vs, rank_s)
            W = W_flat.reshape(dim_ws, rank_s)
            
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            if check_solution(U_snap, V_snap, W_snap, T_s):
                return U_snap, V_snap, W_snap
        
        return None

    # Target Tensor T for 3x4x6
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    # Schoolbook fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    # Try to find Strassen 333
    strassen_333 = find_strassen_333()
    
    if strassen_333 is not None and time.time() - start_time < timeout - 20:
        U_s, V_s, W_s = strassen_333
        rank_s = 23
        
        # Construct 64-rank solution
        total_rank = rank_s + rank_s + 18
        U = np.zeros((dim_u, total_rank))
        V = np.zeros((dim_v, total_rank))
        W = np.zeros((dim_w, total_rank))
        
        # Block 1: A1 B1a (j in 0..2, k in 0..2)
        # Map local (i, j, k) to global
        # u = 4i + j, v = 6j + k, w = 3k + i
        for c in range(rank_s):
            for u_loc in range(9):
                val = U_s[u_loc, c]
                if val != 0:
                    i = u_loc // 3
                    j = u_loc % 3
                    u_glob = 4 * i + j
                    U[u_glob, c] += val
            
            for v_loc in range(9):
                val = V_s[v_loc, c]
                if val != 0:
                    j = v_loc // 3
                    k = v_loc % 3
                    v_glob = 6 * j + k
                    V[v_glob, c] += val
            
            for w_loc in range(9):
                val = W_s[w_loc, c]
                if val != 0:
                    k = w_loc // 3
                    i = w_loc % 3
                    w_glob = 3 * k + i
                    W[w_glob, c] += val
        
        # Block 2: A1 B1b (j in 0..2, k in 3..5)
        # Map local (i, j, k') to global
        # u = 4i + j, v = 6j + (k' + 3), w = 3(k' + 3) + i
        offset = rank_s
        for c in range(rank_s):
            for u_loc in range(9):
                val = U_s[u_loc, c]
                if val != 0:
                    i = u_loc // 3
                    j = u_loc % 3
                    u_glob = 4 * i + j
                    U[u_glob, offset + c] += val
            
            for v_loc in range(9):
                val = V_s[v_loc, c]
                if val != 0:
                    j = v_loc // 3
                    k_prime = v_loc % 3
                    v_glob = 6 * j + (k_prime + 3)
                    V[v_glob, offset + c] += val
            
            for w_loc in range(9):
                val = W_s[w_loc, c]
                if val != 0:
                    k_prime = w_loc // 3
                    i = w_loc % 3
                    w_glob = 3 * (k_prime + 3) + i
                    W[w_glob, offset + c] += val
        
        # Block 3: A2 B2 (j=3, k in 0..5)
        # u = 4i + 3, v = 6*3 + k = 18 + k, w = 3k + i
        offset2 = 2 * rank_s
        col = offset2
        for i in range(3):
            for k in range(6):
                u_glob = 4 * i + 3
                v_glob = 18 + k
                w_glob = 3 * k + i
                U[u_glob, col] = 1
                V[v_glob, col] = 1
                W[w_glob, col] = 1
                col += 1
        
        # Verify
        if check_solution(U, V, W, T):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback to schoolbook
    U, V, W = get_schoolbook()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
