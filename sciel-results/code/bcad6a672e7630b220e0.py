import time
import sys
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    n, m, p = DIMENSIONS  # 2, 3, 10
    
    def try_find_decomposition(rank, max_restarts=10):
        # Tensor J_{2,3,2}
        T_flat = np.ones(12)
        
        for restart in range(max_restarts):
            if time.time() > deadline - 20: # Keep 20s for construction
                return None
                
            n_params = 6*rank + 6*rank + 4*rank
            # Random init
            x0 = np.random.randn(n_params) * 0.5
            
            def objective(x):
                U = x[:6*rank].reshape(6, rank)
                V = x[6*rank:12*rank].reshape(6, rank)
                W = x[12*rank:].reshape(4, rank)
                
                T = np.zeros((2, 3, 2))
                for r in range(rank):
                    u_r = U[:, r]
                    v_r = V[:, r]
                    w_r = W[:, r]
                    for i in range(2):
                        for j in range(3):
                            for k in range(2):
                                T[i, j, k] += u_r[i*3+j] * v_r[j*2+k] * w_r[k*2+i]
                return np.sum((T.flatten() - T_flat)**2)
            
            # Use BFGS
            try:
                res = minimize(objective, x0, method='BFGS', options={'maxiter': 1000, 'disp': False})
            except Exception:
                continue
                
            if res.fun < 1e-6:
                x_opt = res.x
                U = x_opt[:6*rank].reshape(6, rank)
                V = x_opt[6*rank:12*rank].reshape(6, rank)
                W = x_opt[12*rank:].reshape(4, rank)
                
                # Round to nearest 0.5
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify
                T = np.zeros((2, 3, 2))
                for r in range(rank):
                    u_r = U_r[:, r]
                    v_r = V_r[:, r]
                    w_r = W_r[:, r]
                    for i in range(2):
                        for j in range(3):
                            for k in range(2):
                                T[i, j, k] += u_r[i*3+j] * v_r[j*2+k] * w_r[k*2+i]
                
                if np.allclose(T.flatten(), T_flat, atol=1e-9):
                    return U_r, V_r, W_r
        return None

    # Try rank 10
    U10, V10, W10 = try_find_decomposition(10)
    
    if U10 is not None:
        rank_block = 10
        U_block, V_block, W_block = U10, V10, W10
    else:
        # Fallback to rank 11
        rank_block = 11
        U_block = np.zeros((6, 11))
        V_block = np.zeros((6, 11))
        W_block = np.zeros((4, 11))
        
        # Strassen 7
        U_block[0, 0] += 1; U_block[4, 0] += 1
        V_block[0, 0] += 1; V_block[5, 0] += 1
        W_block[0, 0] += 1; W_block[3, 0] += 1
        
        U_block[3, 1] += 1; U_block[4, 1] += 1
        V_block[0, 1] += 1
        W_block[1, 1] += 1; W_block[3, 1] -= 1
        
        U_block[0, 2] += 1
        V_block[1, 2] += 1; V_block[5, 2] -= 1
        W_block[2, 2] += 1; W_block[3, 2] += 1
        
        U_block[4, 3] += 1
        V_block[4, 3] += 1; V_block[0, 3] -= 1
        W_block[0, 3] += 1; W_block[1, 3] += 1
        
        U_block[0, 4] += 1; U_block[1, 4] += 1
        V_block[5, 4] += 1
        W_block[0, 4] -= 1; W_block[2, 4] += 1
        
        U_block[3, 5] += 1; U_block[0, 5] -= 1
        V_block[0, 5] += 1; V_block[1, 5] += 1
        W_block[3, 5] += 1
        
        U_block[1, 6] += 1; U_block[4, 6] -= 1
        V_block[4, 6] += 1; V_block[5, 6] += 1
        W_block[0, 6] += 1
        
        # Schoolbook 4
        U_block[2, 7] += 1
        V_block[4, 7] += 1
        W_block[0, 7] += 1
        
        U_block[2, 8] += 1
        V_block[5, 8] += 1
        W_block[2, 8] += 1
        
        U_block[5, 9] += 1
        V_block[4, 9] += 1
        W_block[1, 9] += 1
        
        U_block[5, 10] += 1
        V_block[5, 10] += 1
        W_block[3, 10] += 1

    # Construct full 2x3x10 decomposition
    total_rank = 5 * rank_block
    U = np.zeros((6, total_rank))
    V = np.zeros((30, total_rank))
    W = np.zeros((20, total_rank))
    
    for t in range(5):
        k0 = 2 * t
        U[:, t*rank_block:(t+1)*rank_block] = U_block
        
        for j in range(3):
            for k_local in range(2):
                k_global = k0 + k_local
                v_idx = j * 10 + k_global
                v_block_idx = j * 2 + k_local
                V[v_idx, t*rank_block:(t+1)*rank_block] = V_block[v_block_idx]
                
        for k_local in range(2):
            k_global = k0 + k_local
            for i in range(2):
                w_idx = k_global * 2 + i
                w_block_idx = k_local * 2 + i
                W[w_idx, t*rank_block:(t+1)*rank_block] = W_block[w_block_idx]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
