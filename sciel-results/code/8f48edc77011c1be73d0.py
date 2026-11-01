import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    n, m, p = 2, 3, 9
    
    # Try to find rank 15 decomposition for 2x3x3
    U_sub, V_sub, W_sub = _find_rank_15_233(deadline - (time.time() - start_time) - 5.0)
    
    if U_sub is not None:
        # Construct 2x3x9 from 3 blocks
        U_total, V_total, W_total = _build_239_from_233(U_sub, V_sub, W_sub)
        if _verify_239(U_total, V_total, W_total):
            return {"U": U_total, "V": V_total, "W": W_total}
            
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def _find_rank_15_233(time_limit):
    # Tensor for 2x3x3
    # U: 6, V: 9, W: 6
    # T[u,v,w] = 1 if u=(i,j), v=(j,k), w=(k,i)
    # u = 3i + j, v = 3j + k, w = 2k + i
    
    T = np.zeros((6, 9, 6))
    for i in range(2):
        for j in range(3):
            for k in range(3):
                u = 3*i + j
                v = 3*j + k
                w = 2*k + i
                T[u, v, w] = 1.0
                
    rank = 15
    # Random init
    best_result = None
    best_err = float('inf')
    
    # Use a few restarts
    for restart in range(10):
        if time.time() > start_time + time_limit: break
        U = np.random.randn(6, rank)
        V = np.random.randn(9, rank)
        W = np.random.randn(6, rank)
        
        def func(params):
            U_flat = params[:6*rank]
            V_flat = params[6*rank:6*rank+9*rank]
            W_flat = params[6*rank+9*rank:]
            U = U_flat.reshape(6, rank)
            V = V_flat.reshape(9, rank)
            W = W_flat.reshape(6, rank)
            # Compute tensor
            T_est = np.einsum('ur,vr,wr->uvw', U, V, W)
            return (T_est - T).flatten()
            
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        try:
            res = least_squares(func, x0, method='trf', max_nfev=1000)
            err = np.sum(res.fun**2)
            if err < best_err:
                best_err = err
                best_result = res.x
                if err < 1e-8:
                    break
        except Exception:
            continue
            
    if best_result is None or best_err > 1e-6:
        return None, None, None
        
    U_flat = best_result[:6*rank]
    V_flat = best_result[6*rank:6*rank+9*rank]
    W_flat = best_result[6*rank+9*rank:]
    U = U_flat.reshape(6, rank)
    V = V_flat.reshape(9, rank)
    W = W_flat.reshape(6, rank)
    
    # Round to integers or half-integers
    U_int = np.round(U * 2) / 2
    V_int = np.round(V * 2) / 2
    W_int = np.round(W * 2) / 2
    
    # Verify
    T_est = np.einsum('ur,vr,wr->uvw', U_int, V_int, W_int)
    if np.allclose(T_est, T, atol=1e-9):
        return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
    # Try rounding to integers
    U_int2 = np.round(U)
    V_int2 = np.round(V)
    W_int2 = np.round(W)
    T_est2 = np.einsum('ur,vr,wr->uvw', U_int2, V_int2, W_int2)
    if np.allclose(T_est2, T, atol=1e-9):
        return U_int2.tolist(), V_int2.tolist(), W_int2.tolist()
        
    return None, None, None

def _build_239_from_233(U_sub, V_sub, W_sub):
    # U_sub: 6x15, V_sub: 9x15, W_sub: 6x15
    # Total rank 45
    U_total = np.zeros((6, 45))
    V_total = np.zeros((9, 45))
    W_total = np.zeros((18, 45)) # W is p*n = 9*2 = 18
    
    for block in range(3):
        col_start = block * 15
        col_end = (block + 1) * 15
        
        # U is same for all blocks
        U_total[:, col_start:col_end] = U_sub
        
        # V: rows are 3j + k.
        # For block b, k = 3*b + k_sub.
        # v = 3j + (3*b + k_sub) = 3j + 3b + k_sub.
        # V_sub rows are 3j + k_sub.
        # So V_total[v, col] = V_sub[3j + k_sub, col_sub]
        for j in range(3):
            for k_sub in range(3):
                v = 3*j + 3*block + k_sub
                v_sub = 3*j + k_sub
                V_total[v, col_start:col_end] = V_sub[v_sub, :]
                
        # W: rows are 2k + i.
        # For block b, k = 3*b + k_sub.
        # w = 2*(3*b + k_sub) + i = 6b + 2k_sub + i.
        # W_sub rows are 2k_sub + i.
        # So W_total[w, col] = W_sub[2k_sub + i, col_sub]
        for i in range(2):
            for k_sub in range(3):
                w = 2*(3*block + k_sub) + i
                w_sub = 2*k_sub + i
                W_total[w, col_start:col_end] = W_sub[w_sub, :]
                
    return U_total.tolist(), V_total.tolist(), W_total.tolist()

def _verify_239(U, V, W):
    # U: 6xR, V: 9xR, W: 18xR
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    T_est = np.einsum('ur,vr,wr->uvw', U, V, W)
    
    T = np.zeros((6, 9, 18))
    for i in range(2):
        for j in range(3):
            for k in range(9):
                u = 3*i + j
                v = 3*j + k
                w = 2*k + i
                T[u, v, w] = 1.0
                
    return np.allclose(T_est, T, atol=1e-9)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
