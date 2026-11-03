"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n*m):
        for j in range(m*p):
            for k in range(p*n):
                row_a, col_a = divmod(i, m)
                col_a2, row_b = divmod(j, p)
                col_b, row_a2 = divmod(k, n)
                if col_a == col_a2 and row_a == row_a2:
                    T[i, j, k] = 1.0
    
    def verify_np(U, V, W):
        for i in range(n*m):
            for j in range(m*p):
                for k in range(p*n):
                    val = np.dot(U[i, :], V[j, :] * W[k, :])
                    row_a, col_a = divmod(i, m)
                    col_a2, row_b = divmod(j, p)
                    col_b, row_a2 = divmod(k, n)
                    if col_a == col_a2 and row_a == row_a2:
                        if abs(val - 1.0) > 1e-6:
                            return False
                    else:
                        if abs(val) > 1e-6:
                            return False
        return True
    
    def try_rank(rank, seed):
        np.random.seed(seed)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        for it in range(100):
            if time.time() > deadline:
                break
            
            C = np.zeros((rank, m*p * p*n))
            for r in range(rank):
                C[r, :] = np.kron(V[:, r], W[:, r])
            T_flat = T.reshape(n*m, m*p * p*n)
            U_new, _, _, _ = np.linalg.lstsq(C.T, T_flat.T, rcond=None)
            U_new = U_new.T
            
            D = np.zeros((rank, n*m * p*n))
            for r in range(rank):
                D[r, :] = np.kron(U[:, r], W[:, r])
            T_flat2 = T.reshape(m*p, n*m * p*n)
            V_new, _, _, _ = np.linalg.lstsq(D.T, T_flat2.T, rcond=None)
            V_new = V_new.T
            
            E = np.zeros((rank, n*m * m*p))
            for r in range(rank):
                E[r, :] = np.kron(U[:, r], V[:, r])
            T_flat3 = T.reshape(p*n, n*m * m*p)
            W_new, _, _, _ = np.linalg.lstsq(E.T, T_flat3.T, rcond=None)
            W_new = W_new.T
            
            U, V, W = U_new, V_new, W_new
        
        all_entries = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        rounded = np.round(all_entries)
        round_err = np.sum((all_entries - rounded)**2)
        
        if round_err < 1e-4:
            U_r = np.round(U)
            V_r = np.round(V)
            W_r = np.round(W)
            if verify_np(U_r, V_r, W_r):
                return U_r, V_r, W_r
        return None
    
    for rank in [35, 34, 33, 32]:
        if time.time() > deadline:
            break
        for seed in range(30):
            if time.time() > deadline:
                break
            result = try_rank(rank, seed)
            if result is not None:
                U, V, W = result
                return {
                    "U": U.astype(int).tolist(),
                    "V": V.astype(int).tolist(),
                    "W": W.astype(int).tolist()
                }
    
    # Fallback: rank-36 solution
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_rank1_term(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank:
            return
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
        col += 1
    
    def add_strassen_2x2x2(a_i0, a_j0, b_i0, b_j0, c_i0, c_k0):
        def a_idx(li, lj):
            return (a_i0 + li) * m + (a_j0 + lj)
        def b_idx(bi, bj):
            return (b_i0 + bi) * p + (b_j0 + bj)
        def c_idx(ci, ck):
            return (c_k0 + ck) * n + (c_i0 + ci)
        
        u = {a_idx(0,0): 1, a_idx(1,1): 1}
        v = {b_idx(0,0): 1, b_idx(1,1): 1}
        w = {c_idx(0,0): 1, c_idx(1,1): 1}
        add_rank1_term(u, v, w)
        
        u = {a_idx(1,0): 1, a_idx(1,1): 1}
        v = {b_idx(0,0): 1}
        w = {c_idx(1,0): 1, c_idx(1,1): -1}
        add_rank1_term(u, v, w)
        
        u = {a_idx(0,0): 1}
        v = {b_idx(0,1): 1, b_idx(1,1): -1}
        w = {c_idx(0,1): 1, c_idx(1,1): 1}
        add_rank1_term(u, v, w)
        
        u = {a_idx(1,1): 1}
        v = {b_idx(1,0): 1, b_idx(0,0): -1}
        w = {c_idx(0,0): 1, c_idx(1,0): 1}
        add_rank1_term(u, v, w)
        
        u = {a_idx(0,0): 1, a_idx(0,1): 1}
        v = {b_idx(1,1): 1}
        w = {c_idx(0,0): -1, c_idx(0,1): 1}
        add_rank1_term(u, v, w)
        
        u = {a_idx(1,0): 1, a_idx(0,0): -1}
        v = {b_idx(0,0): 1, b_idx(0,1): 1}
        w = {c_idx(1,1): 1}
        add_rank1_term(u, v, w)
        
        u = {a_idx(0,1): 1, a_idx(1,1): -1}
        v = {b_idx(1,0): 1, b_idx(1,1): 1}
        w = {c_idx(0,0): 1}
        add_rank1_term(u, v, w)
    
    def add_schoolbook_2x2x1(a_i0, a_j0, b_i0, b_j0, c_i0, c_k0):
        def a_idx(li, lj):
            return (a_i0 + li) * m + (a_j0 + lj)
        def b_idx(bi):
            return (b_i0 + bi) * p + b_j0
        def c_idx(ci):
            return c_k0 * n + (c_i0 + ci)
        
        for ci in range(2):
            for aj in range(2):
                u = {a_idx(ci, aj): 1}
                v = {b_idx(aj): 1}
                w = {c_idx(ci): 1}
                add_rank1_term(u, v, w)
    
    def add_2x2x5(a_i0, a_j0, b_i0, b_j0, c_i0, c_k0):
        add_strassen_2x2x2(a_i0, a_j0, b_i0, b_j0, c_i0, c_k0)
        add_strassen_2x2x2(a_i0, a_j0, b_i0, b_j0 + 2, c_i0, c_k0 + 2)
        add_schoolbook_2x2x1(a_i0, a_j0, b_i0, b_j0 + 4, c_i0, c_k0 + 4)
    
    add_2x2x5(0, 0, 0, 0, 0, 0)
    add_2x2x5(0, 2, 2, 0, 0, 0)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
