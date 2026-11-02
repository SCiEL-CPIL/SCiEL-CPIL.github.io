import numpy as np
import time

def construct():
    n, m, p = 2, 2, 12
    start_time = time.time()
    deadline = start_time + 140
    
    # Build tensor T of shape (4, 24, 24)
    T = np.zeros((4, 24, 24))
    for a in range(2):
        for b in range(2):
            for c in range(12):
                T[a*2+b, b*12+c, c*2+a] = 1.0
    
    # Strassen-based rank-42 solution
    def strassen_solution():
        rank = 42
        U = np.zeros((4, rank))
        V = np.zeros((24, rank))
        W = np.zeros((24, rank))
        
        strassen_coeffs = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
        ]
        
        r = 0
        for b in range(6):
            for s in range(7):
                u, v, w = strassen_coeffs[s]
                for i_loc in range(2):
                    for j_loc in range(2):
                        val = u[i_loc * 2 + j_loc]
                        if val != 0:
                            U[i_loc * 2 + j_loc, r] = val
                for j_loc in range(2):
                    for k_loc in range(2):
                        val = v[j_loc * 2 + k_loc]
                        if val != 0:
                            V[j_loc * 12 + (2 * b + k_loc), r] = val
                for k_loc in range(2):
                    for i_loc in range(2):
                        val = w[k_loc * 2 + i_loc]
                        if val != 0:
                            W[(2 * b + k_loc) * 2 + i_loc, r] = val
                r += 1
        
        return U, V, W
    
    U42, V42, W42 = strassen_solution()
    
    # Try to find a lower-rank decomposition using ALS
    def als_decompose(target, r, max_iter=25, seed=0):
        np.random.seed(seed)
        U = np.random.randn(4, r) * 0.5
        V = np.random.randn(24, r) * 0.5
        W = np.random.randn(24, r) * 0.5
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                return None
            
            # Update U
            for i in range(4):
                A_mat = np.array([np.outer(V[:, rr], W[:, rr]).flatten() for rr in range(r)]).T
                b_vec = target[i].flatten()
                try:
                    U[i], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    return None
            
            # Update V
            for j in range(24):
                A_mat = np.array([np.outer(U[:, rr], W[:, rr]).flatten() for rr in range(r)]).T
                b_vec = target[:, j, :].flatten()
                try:
                    V[j], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    return None
            
            # Update W
            for k in range(24):
                A_mat = np.array([np.outer(U[:, rr], V[:, rr]).flatten() for rr in range(r)]).T
                b_vec = target[:, :, k].flatten()
                try:
                    W[k], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    return None
            
            result = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.linalg.norm(result - target) / np.linalg.norm(target)
            if err < 1e-10:
                return (U, V, W)
        
        result = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = np.linalg.norm(result - target) / np.linalg.norm(target)
        if err < 1e-8:
            return (U, V, W)
        return None
    
    # Try rank 36 with a few restarts
    for seed in range(3):
        if time.time() > deadline - 20:
            break
        result = als_decompose(T, 36, max_iter=20, seed=seed)
        if result is not None:
            U_r, V_r, W_r = result
            # Try to round to half-integers
            for mult in [2, 1]:
                U_rnd = np.round(U_r * mult) / mult
                V_rnd = np.round(V_r * mult) / mult
                W_rnd = np.round(W_r * mult) / mult
                res = np.einsum('ir,jr,kr->ijk', U_rnd, V_rnd, W_rnd)
                if np.allclose(res, T, atol=1e-10):
                    return {"U": U_rnd.tolist(), "V": V_rnd.tolist(), "W": W_rnd.tolist()}
            # Use as-is if close enough
            res = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
            if np.allclose(res, T, atol=1e-8):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Fallback to rank 42
    return {"U": U42.tolist(), "V": V42.tolist(), "W": W42.tolist()}
