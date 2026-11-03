import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = 3, 3, 7
    
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
                
    def verify_exact(U, V, W):
        r = U.shape[1]
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        U_r = U_rounded.reshape(n, m, r)
        V_r = V_rounded.reshape(m, p, r)
        W_r = W_rounded.reshape(p, n, r)
        
        T_recon_r = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        
        if np.max(np.abs(T_recon_r - T)) < 1e-9:
            return U_rounded, V_rounded, W_rounded
        else:
            return None

    def run_als_vectorized(rank, max_iters=100, seed=0):
        np.random.seed(seed)
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        scale = (1.0 / rank) ** (1/3)
        U *= scale
        V *= scale
        W *= scale
        
        i_idx, j_idx, k_idx, c_idx = np.indices((n, m, p, rank))
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Update U
            V_gathered = V[v_idx, c_idx]
            W_gathered = W[w_idx, c_idx]
            M_tensor = V_gathered * W_gathered
            M_stack = M_tensor.reshape(n*m, p, rank)
            RHS = np.ones((n*m, p, 1))
            try:
                U = np.linalg.lstsq(M_stack, RHS, rcond=None)[0]
            except:
                pass
                
            # Update V
            U_gathered = U[u_idx, c_idx]
            W_gathered = W[w_idx, c_idx]
            B_tensor = U_gathered * W_gathered
            B_tensor = B_tensor.transpose(1, 2, 0, 3)
            B_stack = B_tensor.reshape(m*p, n, rank)
            RHS_V = np.ones((m*p, n, 1))
            try:
                V = np.linalg.lstsq(B_stack, RHS_V, rcond=None)[0]
            except:
                pass
                
            # Update W
            U_gathered = U[u_idx, c_idx]
            V_gathered = V[v_idx, c_idx]
            C_tensor = U_gathered * V_gathered
            C_tensor = C_tensor.transpose(2, 0, 1, 3)
            C_stack = C_tensor.reshape(p*n, m, rank)
            RHS_W = np.ones((p*n, m, 1))
            try:
                W = np.linalg.lstsq(C_stack, RHS_W, rcond=None)[0]
            except:
                pass
                
        result = verify_exact(U, V, W)
        if result:
            return result
        return None

    # Schoolbook
    U0 = np.zeros((n*m, 63))
    V0 = np.zeros((m*p, 63))
    W0 = np.zeros((p*n, 63))
    for idx in range(63):
        i = idx // (m*p)
        j = (idx // p) % m
        k = idx % p
        U0[i*m+j, idx] = 1
        V0[j*p+k, idx] = 1
        W0[k*n+i, idx] = 1
        
    best_U, best_V, best_W = U0, V0, W0
    best_rank = 63
    
    for rank in range(62, 30, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        for seed in range(5):
            if time.time() > deadline - 10:
                break
                
            result = run_als_vectorized(rank, max_iters=50, seed=seed)
            if result:
                U_res, V_res, W_res = result
                best_U, best_V, best_W = U_res, V_res, W_res
                best_rank = rank
                found = True
                break
        
    U_list = [[float(best_U[i, c]) for c in range(best_rank)] for i in range(n*m)]
    V_list = [[float(best_V[i, c]) for c in range(best_rank)] for i in range(m*p)]
    W_list = [[float(best_W[i, c]) for c in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
