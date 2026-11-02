import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 2, 4, 6
    
    def get_rank_42_solution():
        rank_total = 42
        U = [[0] * rank_total for _ in range(n * m)]
        V = [[0] * rank_total for _ in range(m * p)]
        W = [[0] * rank_total for _ in range(p * n)]

        strassen_U = [
            {0: 1, 3: 1}, {2: 1, 3: 1}, {0: 1}, {3: 1},
            {0: 1, 1: 1}, {0: -1, 2: 1}, {1: 1, 3: -1}
        ]
        strassen_V = [
            {0: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}, {2: 1, 0: -1},
            {3: 1}, {0: 1, 1: 1}, {2: 1, 3: 1}
        ]
        strassen_W = [
            {0: 1, 3: 1}, {2: 1, 3: -1}, {1: 1, 3: 1}, {0: 1, 2: 1},
            {0: -1, 1: 1}, {3: 1}, {0: 1}
        ]

        col_offset = 0
        for j_block in range(3):
            for side in ['L', 'R']:
                if side == 'L':
                    u_col_offset = 0
                    v_row_offset = 0
                else:
                    u_col_offset = 2
                    v_row_offset = 2

                for prod_idx in range(7):
                    col = col_offset + prod_idx

                    for local_idx, sign in strassen_U[prod_idx].items():
                        if local_idx == 0: u_idx = u_col_offset
                        elif local_idx == 1: u_idx = u_col_offset + 1
                        elif local_idx == 2: u_idx = 4 + u_col_offset
                        else: u_idx = 4 + u_col_offset + 1
                        U[u_idx][col] += sign

                    for local_idx, sign in strassen_V[prod_idx].items():
                        if local_idx == 0: v_idx = v_row_offset * 6 + 2 * j_block
                        elif local_idx == 1: v_idx = v_row_offset * 6 + 2 * j_block + 1
                        elif local_idx == 2: v_idx = (v_row_offset + 1) * 6 + 2 * j_block
                        else: v_idx = (v_row_offset + 1) * 6 + 2 * j_block + 1
                        V[v_idx][col] += sign

                    for local_idx, sign in strassen_W[prod_idx].items():
                        if local_idx == 0: w_idx = 2 * (2 * j_block)
                        elif local_idx == 1: w_idx = 2 * (2 * j_block + 1)
                        elif local_idx == 2: w_idx = 2 * (2 * j_block) + 1
                        else: w_idx = 2 * (2 * j_block + 1) + 1
                        W[w_idx][col] += sign

                col_offset += 7

        return {"U": U, "V": V, "W": W}

    def find_abcd_decomp(rank_target=12):
        T = np.zeros((8, 8, 4))
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    a_idx = i * 2 + j
                    b_idx = j * 2 + k
                    e_idx = i * 2 + k
                    T[a_idx, b_idx, e_idx] = 1.0
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    c_idx = 4 + i * 2 + j
                    d_idx = 4 + j * 2 + k
                    e_idx = i * 2 + k
                    T[c_idx, d_idx, e_idx] = 1.0

        rank = rank_target
        
        def unpack(params):
            U = params[:8*rank].reshape(8, rank)
            V = params[8*rank:8*rank+8*rank].reshape(8, rank)
            W = params[8*rank+8*rank:].reshape(4, rank)
            return U, V, W

        def loss(params):
            U, V, W = unpack(params)
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.sum((T_recon - T)**2)
            return err

        def try_decomposition(U0, V0, W0):
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            res = minimize(loss, x0, method='BFGS', 
                          options={'maxiter': 1000, 'gtol': 1e-8})
            
            if res.fun > 1e-6:
                return None
            
            U, V, W = unpack(res.x)
            
            res2 = minimize(loss, res.x, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            if res2.fun < res.fun:
                U, V, W = unpack(res2.x)
            
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
            if np.allclose(T_recon, T, atol=1e-9):
                return U_snap, V_snap, W_snap
            
            U_snap_i = np.round(U)
            V_snap_i = np.round(V)
            W_snap_i = np.round(W)
            T_recon_i = np.einsum('ar,br,cr->abc', U_snap_i, V_snap_i, W_snap_i)
            if np.allclose(T_recon_i, T, atol=1e-9):
                return U_snap_i, V_snap_i, W_snap_i
            
            return None

        best_result = None
        for seed in range(50):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(seed)
            U0 = np.random.randn(8, rank) * 0.1
            V0 = np.random.randn(8, rank) * 0.1
            W0 = np.random.randn(4, rank) * 0.1
            
            result = try_decomposition(U0, V0, W0)
            if result is not None:
                best_result = result
                break
        
        return best_result

    if time.time() < deadline - 15:
        decomp = find_abcd_decomp(12)
        if decomp is not None:
            U_sub, V_sub, W_sub = decomp
            rank_sub = U_sub.shape[1]
            total_rank = 3 * rank_sub
            
            U = [[0] * total_rank for _ in range(n * m)]
            V = [[0] * total_rank for _ in range(m * p)]
            W = [[0] * total_rank for _ in range(p * n)]
            
            for col_pair in range(3):
                col_offset = col_pair * rank_sub
                
                for r in range(rank_sub):
                    for i in range(2):
                        for j in range(2):
                            u_idx_full = i * 4 + j
                            U[u_idx_full][col_offset + r] += U_sub[i*2+j, r]
                            
                    for i in range(2):
                        for j in range(2):
                            u_idx_full = i * 4 + (2 + j)
                            U[u_idx_full][col_offset + r] += U_sub[4 + i*2 + j, r]
                            
                    for j in range(2):
                        for k in range(2):
                            v_idx_full = j * 6 + (2 * col_pair + k)
                            V[v_idx_full][col_offset + r] += V_sub[j*2+k, r]
                            
                    for j in range(2):
                        for k in range(2):
                            v_idx_full = (2 + j) * 6 + (2 * col_pair + k)
                            V[v_idx_full][col_offset + r] += V_sub[4 + j*2 + k, r]
                            
                    for i in range(2):
                        for k in range(2):
                            w_idx_full = (2 * col_pair + k) * 2 + i
                            W[w_idx_full][col_offset + r] += W_sub[i*2+k, r]
            
            return {"U": U, "V": V, "W": W}

    return get_rank_42_solution()
