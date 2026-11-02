import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 15]

def _build_strassen_2x2x2():
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _get_target_tensor(n, m, p):
    """
    Returns a flattened array of the target tensor T.
    Index order: u, v, w.
    u in [0, n*m), v in [0, m*p), w in [0, p*n).
    T[u,v,w] = 1 if u=(i,j), v=(j,k), w=(k,i) else 0.
    Note: w index is k*n+i.
    """
    size_u = n * m
    size_v = m * p
    size_w = p * n
    T = np.zeros((size_u, size_v, size_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _residuals_223(U, V, W, T_flat):
    """
    Calculate residuals for M_{2,2,3}.
    U: 4xR, V: 6xR, W: 6xR.
    T_flat: flattened target tensor of size 4*6*6.
    """
    n, m, p = 2, 2, 3
    R = U.shape[1]
    # Compute tensor product sum
    # T_comp[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
    # Efficient way: 
    # Reshape U, V, W to (size, R)
    # Use einsum or broadcasting
    # T_comp = np.einsum('ur,vr,wr->uvw', U, V, W)
    T_comp = np.einsum('ur,vr,wr->uvw', U, V, W)
    T_comp_flat = T_comp.flatten()
    return T_comp_flat - T_flat

def _find_rank10_223(deadline):
    """
    Find a rank-10 decomposition for M_{2,2,3}.
    """
    n, m, p = 2, 2, 3
    T = _get_target_tensor(n, m, p)
    T_flat = T.flatten()
    
    rank = 10
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    def objective(params):
        U = params[:size_u*rank].reshape(size_u, rank)
        V = params[size_u*rank : size_u*rank + size_v*rank].reshape(size_v, rank)
        W = params[size_u*rank + size_v*rank :].reshape(size_w, rank)
        res = _residuals_223(U, V, W, T_flat)
        return np.sum(res**2)
        
    # Initial guess: Rank 11 solution
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
    
    U11 = np.zeros((4, 11))
    V11 = np.zeros((6, 11))
    W11 = np.zeros((6, 11))
    
    U11[:, 0:7] = U_str
    for j in range(2):
        for k_local in range(2):
            k_global = k_local # 0, 1
            V11[j*3+k_global, 0:7] = V_str[j*2+k_local, :]
    for k_local in range(2):
        k_global = k_local
        for i in range(2):
            W11[k_global*2+i, 0:7] = W_str[k_local*2+i, :]
            
    U11[:, 7:11] = U_sb
    for j in range(2):
        V11[j*3+2, 7:11] = V_sb[j, :]
    for i in range(2):
        W11[2*2+i, 7:11] = W_sb[i, :]
        
    # Verify Rank 11
    res11 = _residuals_223(U11, V11, W11, T_flat)
    if np.max(np.abs(res11)) > 1e-8:
        return None

    # Try to optimize Rank 10
    best_params = None
    best_obj = np.inf
    
    # Seed 0: Rank 11 with last column zeroed
    x0_list = []
    x0_0 = np.zeros(160)
    x0_0[:40] = U11[:, 0:10].flatten()
    x0_0[40:100] = V11[:, 0:10].flatten()
    x0_0[100:160] = W11[:, 0:10].flatten()
    x0_list.append(x0_0)
    
    # Random seeds
    for seed in range(1, 20):
        if time.time() > deadline - 5:
            break
        np.random.seed(seed)
        x0_list.append(np.random.randn(160) * 0.5)
        
    for x0 in x0_list:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
            if best_obj < 1e-10:
                break
        except Exception:
            continue
            
    if best_params is None or best_obj > 1e-8:
        return None
        
    U = best_params[:40].reshape(4, 10)
    V = best_params[40:100].reshape(6, 10)
    W = best_params[100:160].reshape(6, 10)
    
    # Snap to half-integers
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    res_snap = _residuals_223(U_snap, V_snap, W_snap, T_flat)
    if np.max(np.abs(res_snap)) < 1e-6:
        return U_snap, V_snap, W_snap
        
    # Check raw
    res_raw = _residuals_223(U, V, W, T_flat)
    if np.max(np.abs(res_raw)) < 1e-6:
        if np.allclose(U*2, np.round(U*2), atol=1e-6) and \
           np.allclose(V*2, np.round(V*2), atol=1e-6) and \
           np.allclose(W*2, np.round(W*2), atol=1e-6):
            return U_snap, V_snap, W_snap
            
    return None

def _build_rank50_from_223(U223, V223, W223):
    n, m, p = DIMENSIONS
    total_rank = 50
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    for b in range(5):
        start_k = 3 * b
        col_offset = 10 * b
        U[:, col_offset:col_offset+10] = U223
        for j in range(m):
            for k_local in range(3):
                k_global = start_k + k_local
                V[j * p + k_global, col_offset:col_offset+10] = V223[j * 3 + k_local, :]
        for k_local in range(3):
            k_global = start_k + k_local
            for i in range(n):
                W[k_global * n + i, col_offset:col_offset+10] = W223[k_local * 2 + i, :]

    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    rank = U.shape[1]
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    T = _get_target_tensor(n, m, p)
    T_flat = T.flatten()
    T_comp = np.einsum('ur,vr,wr->uvw', U, V, W)
    T_comp_flat = T_comp.flatten()
    if np.max(np.abs(T_comp_flat - T_flat)) > tol:
        return False
    return True

def _build_rank53():
    n, m, p = DIMENSIONS
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)

    total_rank = 7 * 7 + 4
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    col_offset = 0
    for b in range(7):
        start_k = 2 * b
        U[:, col_offset:col_offset+7] = U_str
        for j in range(m):
            for k_local in range(2):
                k_global = start_k + k_local
                V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
        for k_local in range(2):
            k_global = start_k + k_local
            for i in range(n):
                W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
        col_offset += 7

    U[:, col_offset:col_offset+4] = U_sb
    for j in range(m):
        V[j * p + 14, col_offset:col_offset+4] = V_sb[j, :]
    for i in range(n):
        W[14 * n + i, col_offset:col_offset+4] = W_sb[i, :]

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Try Rank 50
    U223, V223, W223 = _find_rank10_223(deadline)
    if U223 is not None:
        U50, V50, W50 = _build_rank50_from_223(U223, V223, W223)
        if _verify_decomposition(U50, V50, W50, n, m, p):
            return {"U": U50.tolist(), "V": V50.tolist(), "W": W50.tolist()}

    # Fallback Rank 53
    U53, V53, W53 = _build_rank53()
    if _verify_decomposition(U53, V53, W53, n, m, p):
        return {"U": U53.tolist(), "V": V53.tolist(), "W": W53.tolist()}
        
    # Fallback Schoolbook
    Usb, Vsb, Wsb = _build_schoolbook(n, m, p)
    return {"U": Usb.tolist(), "V": Vsb.tolist(), "W": Wsb.tolist()}
