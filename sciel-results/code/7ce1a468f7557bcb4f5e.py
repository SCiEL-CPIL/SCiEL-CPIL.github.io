import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    n, m, p = 2, 2, 6
    nm, mp, pn = 4, 12, 12
    
    # Target Tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # C[i, k] += A[i, j] * B[j, k]
                # Index in U: i*m + j
                # Index in V: j*p + k
                # Index in W: k*n + i
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
                
    T_flat = T.flatten()
    
    def objective(params, R):
        # params shape: (28 * R,)
        U = params[:nm * R].reshape(nm, R)
        V = params[nm * R : nm * R + mp * R].reshape(mp, R)
        W = params[nm * R + mp * R :].reshape(pn, R)
        
        # Compute T_recon
        # T_recon[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # Use einsum
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return (T_recon.flatten() - T_flat)
        
    # Rank 21 fallback (Strassen)
    def get_strassen_21():
        U = np.zeros((nm, 21))
        V = np.zeros((mp, 21))
        W = np.zeros((pn, 21))
        
        strassen_steps = [
            {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
            {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
            {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
            {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
            {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
            {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
            {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
        ]
        
        def map_a(local_idx): return local_idx
        def map_b(local_idx, g):
            k0 = 2 * g; k1 = 2 * g + 1
            if local_idx == 0: return k0
            elif local_idx == 1: return k1
            elif local_idx == 2: return 6 + k0
            else: return 6 + k1
        def map_c(local_idx, g):
            k0 = 2 * g; k1 = 2 * g + 1
            if local_idx == 0: return k0 * 2
            elif local_idx == 1: return k1 * 2
            elif local_idx == 2: return k0 * 2 + 1
            else: return k1 * 2 + 1

        for g in range(3):
            for s in range(7):
                r = 7 * g + s
                step = strassen_steps[s]
                for li in range(4):
                    if step['u'][li] != 0:
                        U[map_a(li), r] = step['u'][li]
                    if step['v'][li] != 0:
                        V[map_b(li, g), r] = step['v'][li]
                    if step['w'][li] != 0:
                        W[map_c(li, g), r] = step['w'][li]
        return U, V, W

    start_time = time.time()
    deadline = start_time + 100.0 # 100s for search
    
    best_result = None
    best_rank = 21
    best_U, best_V, best_W = get_strassen_21()
    
    # Try to find rank 20
    # Use multiple random restarts
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        R = 20
        # Random initialization
        # Scale to be reasonable
        U_init = np.random.randn(nm, R) * 0.5
        V_init = np.random.randn(mp, R) * 0.5
        W_init = np.random.randn(pn, R) * 0.5
        
        # Alternatively, initialize from Strassen 21 by dropping one column?
        # Or perturbing?
        # Random is risky but might find a global min if landscape is benign.
        # Let's try a "warm start" from Strassen 21 by merging two columns?
        # No, let's stick to random + least_squares.
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            res = least_squares(objective, params_init, args=(R,), method='trf', max_nfev=2000)
            if res.success or res.cost < 1e-10:
                if res.cost < 1e-10:
                    U_sol = res.x[:nm * R].reshape(nm, R)
                    V_sol = res.x[nm * R : nm * R + mp * R].reshape(mp, R)
                    W_sol = res.x[nm * R + mp * R :].reshape(pn, R)
                    return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
        except Exception:
            pass
            
    # If rank 20 not found, return rank 21
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
