import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x6 matrix multiplication.
    Attempts to find a rank 20 decomposition by perturbing a rank 21 solution.
    Falls back to rank 21 if rank 20 is not found within the time limit.
    """
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    # Build the rank 21 Strassen solution
    rank_21 = 21
    U_21 = np.zeros((nm, rank_21))
    V_21 = np.zeros((mp, rank_21))
    W_21 = np.zeros((pn, rank_21))
    
    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]
    
    def map_a(local_idx):
        return local_idx
        
    def map_b(local_idx, g):
        k0 = 2 * g
        k1 = 2 * g + 1
        if local_idx == 0: return 0 * 6 + k0
        elif local_idx == 1: return 0 * 6 + k1
        elif local_idx == 2: return 1 * 6 + k0
        else: return 1 * 6 + k1

    def map_c(local_idx, g):
        k0 = 2 * g
        k1 = 2 * g + 1
        if local_idx == 0: return k0 * 2 + 0
        elif local_idx == 1: return k1 * 2 + 0
        elif local_idx == 2: return k0 * 2 + 1
        else: return k1 * 2 + 1

    for g in range(3):
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            for local_idx in range(4):
                u_val = step['u'][local_idx]
                if u_val != 0:
                    U_21[map_a(local_idx), r] = u_val
                v_val = step['v'][local_idx]
                if v_val != 0:
                    V_21[map_b(local_idx, g), r] = v_val
                w_val = step['w'][local_idx]
                if w_val != 0:
                    W_21[map_c(local_idx, g), r] = w_val

    # Expected tensor
    T_exp = np.zeros((nm, mp, pn))
    for i in range(2):
        for j in range(2):
            for k in range(6):
                a = i * 2 + j
                b = j * 6 + k
                c = k * 2 + i
                T_exp[a, b, c] = 1.0

    def residual(params):
        # params is flattened U, V, W
        # U: 4*rank, V: 12*rank, W: 12*rank
        # Total size: (4+12+12)*rank = 28*rank
        rank = len(params) // 28
        U = params[:4*rank].reshape(4, rank)
        V = params[4*rank:4*rank+12*rank].reshape(12, rank)
        W = params[4*rank+12*rank:].reshape(12, rank)
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        return (T_calc - T_exp).flatten()

    # Try to find rank 20
    start_time = time.time()
    deadline = start_time + 100.0 # Leave 20s margin
    
    best_result = None
    best_res_norm = np.inf
    
    np.random.seed(42)
    
    # Build multiple starting points by removing different columns
    starting_points = []
    
    # Remove last column (original approach)
    U_init = U_21[:, :20]
    V_init = V_21[:, :20]
    W_init = W_21[:, :20]
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    starting_points.append(x0)
    
    # Remove other columns
    for col_to_remove in [0, 3, 7, 10, 14, 17, 20]:
        if col_to_remove == 20:
            continue
        mask = np.ones(21, dtype=bool)
        mask[col_to_remove] = False
        U_r = U_21[:, mask]
        V_r = V_21[:, mask]
        W_r = W_21[:, mask]
        x_r = np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()])
        starting_points.append(x_r)
    
    # Add noisy variants
    for x_sp in list(starting_points):
        x_noisy = x_sp + np.random.normal(0, 0.05, x_sp.shape)
        starting_points.append(x_noisy)
    
    # Try each starting point with different methods
    for sp_idx, x_attempt in enumerate(starting_points):
        if time.time() > deadline:
            break
        
        for method in ['trf', 'lm']:
            if time.time() > deadline:
                break
            
            try:
                if method == 'lm':
                    res = least_squares(residual, x_attempt, method='lm', max_nfev=10000, xtol=1e-10, ftol=1e-10, gtol=1e-10)
                else:
                    res = least_squares(residual, x_attempt, method='trf', max_nfev=5000, xtol=1e-8, ftol=1e-8)
                
                res_norm = np.linalg.norm(res.fun)
                
                if res_norm < best_res_norm:
                    best_res_norm = res_norm
                    best_result = res.x.copy()
                    
                # Check if we found a good solution
                if res_norm < 1e-8:
                    # Polish pass with tighter tolerances
                    if time.time() < deadline - 5.0:
                        try:
                            res_polish = least_squares(residual, best_result, method='lm', max_nfev=5000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
                            res_norm_polish = np.linalg.norm(res_polish.fun)
                            if res_norm_polish < best_res_norm:
                                best_res_norm = res_norm_polish
                                best_result = res_polish.x.copy()
                        except Exception:
                            pass
                    break
                    
            except Exception:
                continue
        
        if best_res_norm < 1e-8:
            break

    if best_result is not None and best_res_norm < 1e-6:
        # Reconstruct U, V, W from best_result
        rank = 20
        U = best_result[:4*rank].reshape(4, rank)
        V = best_result[4*rank:4*rank+12*rank].reshape(12, rank)
        W = best_result[4*rank+12*rank:].reshape(12, rank)
        
        # Final verification
        T_check = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(T_check, T_exp, atol=1e-5):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to rank 21
    return {"U": U_21.tolist(), "V": V_21.tolist(), "W": W_21.tolist()}
