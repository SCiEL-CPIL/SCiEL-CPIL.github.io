import numpy as np
import time
import random

def construct():
    start_time = time.time()
    n, m, p = 3, 4, 7
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build the tensor T
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for ia in range(n):
        for ja in range(m):
            for jb in range(m):
                for kb in range(p):
                    for ic in range(n):
                        for kc in range(p):
                            if ia == ic and ja == jb and kb == kc:
                                u_idx = ia * m + ja
                                v_idx = jb * p + kb
                                w_idx = kc * n + ic
                                T[u_idx, v_idx, w_idx] = 1.0
    
    T_flat = T.flatten()
    
    # Initialize with schoolbook
    rank = n * m * p
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Helper function to check if a set of columns works
    def check_solution(cols):
        if not cols:
            return False, None
        r = len(cols)
        # Construct A_sys for these columns
        # A_sys shape: (dim_u * dim_v, dim_w * r)
        A_sys = np.zeros((dim_u * dim_v, dim_w * r))
        for idx, s in enumerate(cols):
            outer_uv = np.outer(U[:, s], V[:, s])
            outer_uv_flat = outer_uv.flatten()
            A_sys[:, idx*dim_w : (idx+1)*dim_w] = np.tile(outer_uv_flat[:, np.newaxis], (1, dim_w))
        
        # Solve
        W_flat, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, T_flat, rcond=None)
        
        # Check residual
        if np.linalg.norm(A_sys @ W_flat - T_flat) > 1e-9:
            return False, None
        
        # Check rationality
        W_sol = W_flat.reshape((dim_w, r))
        W_half = W_sol * 2
        W_round = np.round(W_half)
        if np.linalg.norm(W_half - W_round) > 1e-9:
            return False, None
        
        return True, W_sol
    
    # Greedy removal with shuffling
    active_cols = list(range(rank))
    best_rank = rank
    
    # Try multiple random orders
    for _ in range(10):
        if time.time() - start_time > 100:
            break
        
        current_cols = list(active_cols)
        random.shuffle(current_cols)
        
        changed = True
        while changed:
            changed = False
            for s in current_cols:
                if time.time() - start_time > 100:
                    break
                if s not in active_cols:
                    continue
                # Try removing s
                new_cols = [c for c in active_cols if c != s]
                res = check_solution(new_cols)
                if res[0]:
                    # Remove s
                    active_cols = new_cols
                    current_cols = [c for c in current_cols if c != s]
                    changed = True
                    if len(active_cols) < best_rank:
                        best_rank = len(active_cols)
        
        # If we found a better solution, break
        if best_rank < rank:
            break
    
    # If we found a solution, return it
    if best_rank < rank:
        res = check_solution(active_cols)
        if res[0]:
            U_final = U[:, active_cols]
            V_final = V[:, active_cols]
            W_final = res[1]
            return {
                "U": U_final.tolist(),
                "V": V_final.tolist(),
                "W": W_final.tolist()
            }
    
    # Fallback
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
