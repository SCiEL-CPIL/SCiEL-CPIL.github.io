# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def construct():
    deadline = time.time() + 110  # Safety margin
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor_entries.append((u_idx, v_idx, w_idx))
    
    num_entries = len(tensor_entries)
    
    # Fallback
    best_rank = n * m * p
    U_sb = [[0] * best_rank for _ in range(nm)]
    V_sb = [[0] * best_rank for _ in range(mp)]
    W_sb = [[0] * best_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    np.random.seed(42)
    
    u_indices = np.array([e[0] for e in tensor_entries])
    v_indices = np.array([e[1] for e in tensor_entries])
    w_indices = np.array([e[2] for e in tensor_entries])
    
    def solve_for_W(U_curr, V_curr, target_rank):
        P = U_curr[u_indices, :] * V_curr[v_indices, :]
        A_mat = np.zeros((num_entries, pn * target_rank))
        row_idx = np.arange(num_entries)[:, None] * np.ones((1, target_rank))
        col_idx = (w_indices[:, None] * target_rank + np.arange(target_rank)[None, :])
        A_mat[row_idx, col_idx] = P
        W_flat, _, _, _ = np.linalg.lstsq(A_mat, np.ones(num_entries), rcond=None)
        W_curr = W_flat.reshape(pn, target_rank)
        W_curr = np.round(W_curr * 2) / 2
        return W_curr

    def solve_for_U(V_curr, W_curr, target_rank):
        P = V_curr[v_indices, :] * W_curr[w_indices, :]
        A_mat = np.zeros((num_entries, nm * target_rank))
        row_idx = np.arange(num_entries)[:, None] * np.ones((1, target_rank))
        col_idx = (u_indices[:, None] * target_rank + np.arange(target_rank)[None, :])
        A_mat[row_idx, col_idx] = P
        U_flat, _, _, _ = np.linalg.lstsq(A_mat, np.ones(num_entries), rcond=None)
        U_curr = U_flat.reshape(nm, target_rank)
        U_curr = np.round(U_curr * 2) / 2
        return U_curr

    def solve_for_V(U_curr, W_curr, target_rank):
        P = U_curr[u_indices, :] * W_curr[w_indices, :]
        A_mat = np.zeros((num_entries, mp * target_rank))
        row_idx = np.arange(num_entries)[:, None] * np.ones((1, target_rank))
        col_idx = (v_indices[:, None] * target_rank + np.arange(target_rank)[None, :])
        A_mat[row_idx, col_idx] = P
        V_flat, _, _, _ = np.linalg.lstsq(A_mat, np.ones(num_entries), rcond=None)
        V_curr = V_flat.reshape(mp, target_rank)
        V_curr = np.round(V_curr * 2) / 2
        return V_curr

    def compute_error(U_curr, V_curr, W_curr, target_rank):
        P1 = U_curr[u_indices, :] * V_curr[v_indices, :]
        P2 = P1 * W_curr[w_indices, :]
        vals = np.sum(P2, axis=1)
        return np.sum(np.abs(vals - 1.0))

    # Try ranks 18 to 24 (18 is likely optimal)
    for target_rank in range(18, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        found = False
        num_restarts = 100
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            U_curr = np.random.choice([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], size=(nm, target_rank))
            V_curr = np.random.choice([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], size=(mp, target_rank))
            W_curr = np.random.choice([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], size=(pn, target_rank))
            
            prev_error = float('inf')
            for iteration in range(200):
                if time.time() > deadline - 3:
                    break
                
                W_curr = solve_for_W(U_curr, V_curr, target_rank)
                U_curr = solve_for_U(V_curr, W_curr, target_rank)
                V_curr = solve_for_V(U_curr, W_curr, target_rank)
                
                error = compute_error(U_curr, V_curr, W_curr, target_rank)
                
                if error < 1e-9:
                    best_rank = target_rank
                    best_result = {
                        "U": [[float(x) for x in U_curr[i]] for i in range(nm)],
                        "V": [[float(x) for x in V_curr[i]] for i in range(mp)],
                        "W": [[float(x) for x in W_curr[i]] for i in range(pn)]
                    }
                    found = True
                    break
                
                if error >= prev_error and iteration > 20:
                    break
                prev_error = error
            
            if found:
                break
        
        if found:
            break
            
    return best_result
# EVOLVE-BLOCK-END
