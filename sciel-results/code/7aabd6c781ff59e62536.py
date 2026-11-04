import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 4]

def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for r in range(n):
        for c in range(m):
            for k in range(p):
                i = r * m + c
                j = c * p + k
                kk = k * n + r
                T[i, j, kk] = 1.0
    return T

def _strassen_decomposition(n, m, p):
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    patterns = [
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
    ]
    
    blocks = [
        (0, 0, 0, 0),
        (2, 2, 0, 0),
        (0, 0, 2, 2),
        (2, 2, 2, 2),
    ]
    
    col_idx = 0
    for a_off, b_r_off, b_c_off, c_off in blocks:
        for pat in patterns:
            for local_i, coeff in pat['a'].items():
                r = local_i // 2
                c = local_i % 2
                global_i = r * m + (a_off + c)
                U[global_i, col_idx] = coeff
            for local_j, coeff in pat['b'].items():
                r = local_j // 2
                c = local_j % 2
                global_j = (b_r_off + r) * p + (b_c_off + c)
                V[global_j, col_idx] = coeff
            for local_k, coeff in pat['c'].items():
                r = local_k // 2
                c = local_k % 2
                global_k = c_off + c
                global_i = r
                W[global_k * n + global_i, col_idx] = coeff
            col_idx += 1
            
    return U, V, W

def _try_lower_rank(T, n, m, p, target_rank, U_init, V_init, W_init, seed=0, max_nfev=5000):
    np.random.seed(seed)
    dim_u, dim_v, dim_w = n * m, m * p, p * n
    
    U = U_init[:, :target_rank].copy()
    V = V_init[:, :target_rank].copy()
    W = W_init[:, :target_rank].copy()
    U += 0.01 * np.random.randn(dim_u, target_rank)
    V += 0.01 * np.random.randn(dim_v, target_rank)
    W += 0.01 * np.random.randn(dim_w, target_rank)
    
    def residuals(x):
        U_r = x[:dim_u * target_rank].reshape(dim_u, target_rank)
        V_r = x[dim_u * target_rank : dim_u * target_rank + dim_v * target_rank].reshape(dim_v, target_rank)
        W_r = x[dim_u * target_rank + dim_v * target_rank :].reshape(dim_w, target_rank)
        T_calc = np.einsum('it,jt,kt->ijk', U_r, V_r, W_r)
        return (T_calc - T).flatten()
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        result = least_squares(residuals, x0, method='lm', max_nfev=max_nfev)
        if result.cost < 1e-14:
            U_sol = result.x[:dim_u * target_rank].reshape(dim_u, target_rank)
            V_sol = result.x[dim_u * target_rank : dim_u * target_rank + dim_v * target_rank].reshape(dim_v, target_rank)
            W_sol = result.x[dim_u * target_rank + dim_v * target_rank :].reshape(dim_w, target_rank)
            T_check = np.einsum('it,jt,kt->ijk', U_sol, V_sol, W_sol)
            if np.allclose(T_check, T, atol=1e-9):
                return U_sol, V_sol, W_sol
    except Exception:
        pass
    return None

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    n, m, p = DIMENSIONS
    
    U28, V28, W28 = _strassen_decomposition(n, m, p)
    T = _build_tensor(n, m, p)
    T_check = np.einsum('it,jt,kt->ijk', U28, V28, W28)
    
    if not np.allclose(T_check, T, atol=1e-9):
        U32 = np.zeros((n*m, n*m*p))
        V32 = np.zeros((m*p, n*m*p))
        W32 = np.zeros((p*n, n*m*p))
        idx = 0
        for r in range(n):
            for c in range(m):
                for k in range(p):
                    i = r*m + c
                    j = c*p + k
                    kk = k*n + r
                    U32[i, idx] = 1.0
                    V32[j, idx] = 1.0
                    W32[kk, idx] = 1.0
                    idx += 1
        return {"U": U32.tolist(), "V": V32.tolist(), "W": W32.tolist()}
    
    best_U, best_V, best_W = U28, V28, W28
    
    for target_rank in [27, 26, 25]:
        if time.time() - start_time > DEADLINE - 30:
            break
        found = False
        for seed in range(5):
            if time.time() - start_time > DEADLINE - 15:
                break
            result = _try_lower_rank(T, n, m, p, target_rank, U28, V28, W28, seed=seed, max_nfev=3000)
            if result is not None:
                best_U, best_V, best_W = result
                found = True
                break
        if found:
            continue
        else:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank used:", len(res['U'][0]))
