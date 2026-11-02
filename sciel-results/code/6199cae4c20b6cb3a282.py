import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 9]

def _build_rank32():
    n, m, p = DIMENSIONS
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[2, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
        col += 1
        U[0, col] = 1
        V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[3, col] = 1
        V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
        col += 1
        U[0, col] = 1; U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
        col += 1
        U[2, col] = 1; U[0, col] = -1
        V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        U[1, col] = 1; U[3, col] = -1
        V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0, col] = 1; V[0*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[1, col] = 1; V[1*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[2, col] = 1; V[0*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1
        U[3, col] = 1; V[1*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)
    
    return U, V, W

def _compute_residual(U, V, W, n, m, p):
    rank = U.shape[1]
    M = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    T_true = np.zeros((n * m, m * p, p * n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_true[i, j, k] = 1.0
    
    residual = M - T_true
    return np.max(np.abs(residual))

def _try_rank31_vectorized(U0, V0, W0, n, m, p, time_budget):
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n
    rank = 32
    
    # Precompute true tensor entries
    tensor_i = []
    tensor_j = []
    tensor_k = []
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                tensor_i.append(i)
                tensor_j.append(j)
                tensor_k.append(k)
    
    tensor_i = np.array(tensor_i)
    tensor_j = np.array(tensor_j)
    tensor_k = np.array(tensor_k)
    num_tensor = len(tensor_i)
    
    # Precompute non-tensor samples
    np.random.seed(42)
    non_tensor_i = np.random.randint(0, nm, 50)
    non_tensor_j = np.random.randint(0, mp, 50)
    non_tensor_k = np.random.randint(0, pn, 50)
    
    # Filter to keep only non-tensor entries
    valid_mask = np.ones(50, dtype=bool)
    for idx in range(50):
        i, j, k = non_tensor_i[idx], non_tensor_j[idx], non_tensor_k[idx]
        a_from_i = i // m
        b_from_i = i % m
        b_from_j = j // p
        c_from_j = j % p
        c_from_k = k // n
        a_from_k = k % n
        if b_from_i == b_from_j and c_from_j == c_from_k and a_from_i == a_from_k:
            valid_mask[idx] = False
    
    non_tensor_i = non_tensor_i[valid_mask]
    non_tensor_j = non_tensor_j[valid_mask]
    non_tensor_k = non_tensor_k[valid_mask]
    num_non_tensor = len(non_tensor_i)
    
    for target_col in range(rank):
        if time.time() - start_time > time_budget * 0.7:
            break
        
        U = U0.copy()
        V = V0.copy()
        W = W0.copy()
        
        np.random.seed(target_col * 7 + 3)
        U[:, target_col] += 0.05 * np.random.randn(nm)
        V[:, target_col] += 0.05 * np.random.randn(mp)
        W[:, target_col] += 0.05 * np.random.randn(pn)
        
        def objective(params):
            U_flat = params[:nm * rank]
            V_flat = params[nm * rank : nm * rank + mp * rank]
            W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
            
            U = U_flat.reshape(nm, rank)
            V = V_flat.reshape(mp, rank)
            W = W_flat.reshape(pn, rank)
            
            # Compute tensor residuals vectorized
            U_t = U[tensor_i]  # (num_tensor, rank)
            V_t = V[tensor_j]  # (num_tensor, rank)
            W_t = W[tensor_k]  # (num_tensor, rank)
            
            tensor_vals = np.einsum('ir,jr,kr->r', U_t, V_t, W_t)
            tensor_res = tensor_vals - 1.0
            
            # Compute non-tensor residuals
            U_nt = U[non_tensor_i]
            V_nt = V[non_tensor_j]
            W_nt = W[non_tensor_k]
            
            non_tensor_vals = np.einsum('ir,jr,kr->r', U_nt, V_nt, W_nt)
            
            # Combine
            res_sq = np.sum(tensor_res ** 2) + np.sum(non_tensor_vals ** 2)
            
            # Column penalty
            col_penalty = np.sum(U[:, target_col]**2) + np.sum(V[:, target_col]**2) + np.sum(W[:, target_col]**2)
            
            return res_sq + 5.0 * col_penalty
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B',
                              options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12})
        except Exception:
            continue
        
        U_flat = result.x[:nm * rank]
        V_flat = result.x[nm * rank : nm * rank + mp * rank]
        W_flat = result.x[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
        U = U_flat.reshape(nm, rank)
        V = V_flat.reshape(mp, rank)
        W = W_flat.reshape(pn, rank)
        
        col_norm = np.sqrt(np.sum(U[:, target_col]**2) + np.sum(V[:, target_col]**2) + np.sum(W[:, target_col]**2))
        
        if col_norm < 0.005:
            U31 = np.delete(U, target_col, axis=1)
            V31 = np.delete(V, target_col, axis=1)
            W31 = np.delete(W, target_col, axis=1)
            score31 = _compute_residual(U31, V31, W31, n, m, p)
            if score31 < 1e-8:
                return U31, V31, W31, score31
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    n, m, p = DIMENSIONS
    
    U, V, W = _build_rank32()
    
    def to_list(arr):
        return arr.tolist()
    
    remaining_time = deadline - time.time()
    if remaining_time > 25:
        search_budget = min(remaining_time - 20, 80)
        result = _try_rank31_vectorized(U, V, W, n, m, p, search_budget)
        if result is not None:
            U31, V31, W31, score = result
            if score < 1e-8:
                return {"U": to_list(U31), "V": to_list(V31), "W": to_list(W31)}
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
