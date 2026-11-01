"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

import time

# EVOLVE-BLOCK-START

def _build_strassen_21():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for b in range(3):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        w_c00 = k0 * n + 0
        w_c10 = k0 * n + 1
        w_c01 = k1 * n + 0
        w_c11 = k1 * n + 1
        r = b * 7 + 0
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r = b * 7 + 1
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r = b * 7 + 2
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r = b * 7 + 3
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r = b * 7 + 4
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r = b * 7 + 5
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r = b * 7 + 6
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    for i in range(n):
        for k in range(m):
            for j in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+k][r] * V[k*p+j][r] * W[j*n+i][r]
                if s != 1:
                    return False
    return True

def _find_rank_20(n, m, p, deadline):
    import numpy as np
    import scipy.optimize as opt
    
    rank_target = 20
    nU = n * m
    nV = m * p
    nW = p * n
    
    def objective(params):
        U = params[0:rank_target*nU].reshape(nU, rank_target)
        V = params[rank_target*nU:rank_target*nU+rank_target*nV].reshape(nV, rank_target)
        W = params[rank_target*nU+rank_target*nV:rank_target*nU+rank_target*nV+rank_target*nW].reshape(nW, rank_target)
        
        errors = []
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    s = 0
                    for r in range(rank_target):
                        s += U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                    errors.append(s - 1)
        return np.sum(np.array(errors)**2)
    
    U0, V0, W0 = _build_strassen_21()
    
    best_params = None
    best_err = float('inf')
    
    for drop_idx in range(21):
        if time.time() > deadline:
            break
            
        params_init = []
        for r in range(rank_target):
            actual_r = r if r < drop_idx else r + 1
            for a in range(nU):
                params_init.append(U0[a][actual_r])
            for b in range(nV):
                params_init.append(V0[b][actual_r])
            for c in range(nW):
                params_init.append(W0[c][actual_r])
        
        params_init = np.array(params_init)
        
        try:
            res = opt.minimize(objective, params_init, method='L-BFGS-B', options={'maxiter': 100})
            if res.fun < best_err:
                best_err = res.fun
                best_params = res.x
                if best_err < 1e-10:
                    break
        except Exception:
            continue
            
    if best_params is not None and best_err < 1e-10:
        U = best_params[0:rank_target*nU].reshape(nU, rank_target).tolist()
        V = best_params[rank_target*nU:rank_target*nU+rank_target*nV].reshape(nV, rank_target).tolist()
        W = best_params[rank_target*nU+rank_target*nV:rank_target*nU+rank_target*nV+rank_target*nW].reshape(nW, rank_target).tolist()
        
        # Round to nearest 0.5
        U = [[round(val * 2) / 2 for val in row] for row in U]
        V = [[round(val * 2) / 2 for val in row] for row in V]
        W = [[round(val * 2) / 2 for val in row] for row in W]
        
        if _verify(U, V, W, n, m, p, rank_target):
            return U, V, W
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    U, V, W = _build_strassen_21()
    
    result = _find_rank_20(n, m, p, deadline)
    if result is not None:
        new_U, new_V, new_W = result
        return {"U": new_U, "V": new_V, "W": new_W}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
