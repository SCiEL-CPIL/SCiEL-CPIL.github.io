"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140  # Safety margin before 150s kill

    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1

    def _verify(U, V, W):
        """Verify decomposition: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def _snap_to_half_int(arr):
        """Snap values to nearest half-integer"""
        return np.round(2 * arr) / 2.0

    def _residuals(x, R):
        num_u = n * m
        num_v = m * p
        num_w = p * n
        
        U = x[:num_u*R].reshape(num_u, R)
        V = x[num_u*R : num_u*R + num_v*R].reshape(num_v, R)
        W = x[num_u*R + num_v*R : num_u*R + num_v*R + num_w*R].reshape(num_w, R)
        
        res = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    res[idx] = np.sum(U[u_idx] * V[v_idx] * W[w_idx]) - 1.0
                    idx += 1
        return res

    def _solve_rank(R, max_iters=50, n_restarts=10):
        num_u = n * m
        num_v = m * p
        num_w = p * n
        total_vars = (num_u + num_v + num_w) * R
        
        for restart in range(n_restarts):
            if time.time() - start_time > time_limit:
                return None

            # Random initialization with small floats
            U_init = np.random.uniform(-1, 1, size=(num_u, R))
            V_init = np.random.uniform(-1, 1, size=(num_v, R))
            W_init = np.random.uniform(-1, 1, size=(num_w, R))
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                from scipy.optimize import least_squares
                res = least_squares(_residuals, x0, args=(R,), method='trf', max_nfev=max_iters*10)
                
                if res.success or res.cost < 1e-6:
                    x_opt = res.x
                    U_opt = x_opt[:num_u*R].reshape(num_u, R)
                    V_opt = x_opt[num_u*R : num_u*R + num_v*R].reshape(num_v, R)
                    W_opt = x_opt[num_u*R + num_v*R : num_u*R + num_v*R + num_w*R].reshape(num_w, R)
                    
                    # Snap to half-integers
                    U_snap = _snap_to_half_int(U_opt)
                    V_snap = _snap_to_half_int(V_opt)
                    W_snap = _snap_to_half_int(W_opt)
                    
                    if _verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist()):
                        return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
                        
                    # Try integer snap
                    U_int = np.round(U_opt)
                    V_int = np.round(V_opt)
                    W_int = np.round(W_opt)
                    if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist()):
                        return U_int.tolist(), V_int.tolist(), W_int.tolist()
                        
            except Exception:
                pass
                
        return None

    best_rank = rank_schoolbook
    best_result = (U_sb, V_sb, W_sb)
    
    # Try ranks from 18 up to schoolbook-1
    # 18 is the known minimal rank for 2x3x4
    for r in range(18, rank_schoolbook):
        if time.time() - start_time > time_limit:
            break
            
        result = _solve_rank(r, max_iters=50, n_restarts=10)
        if result is not None:
            best_rank = r
            best_result = result
            # We want minimal rank, so if we find 18, we can stop
            if r == 18:
                break
        
    if best_rank < rank_schoolbook:
        U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
