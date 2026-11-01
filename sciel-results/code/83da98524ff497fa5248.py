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

def _soft_max_abs(residual, beta=100.0):
    abs_res = np.abs(residual)
    max_val = np.max(abs_res)
    if max_val < 1e-15:
        return 0.0
    scaled = beta * abs_res
    shifted = scaled - np.max(scaled)
    log_sum_exp = np.max(scaled) + np.log(np.sum(np.exp(shifted)))
    return log_sum_exp / beta

def _try_rank31_search(U0, V0, W0, n, m, p, time_budget):
    start_time = time.time()
    
    nm, mp, pn = n * m, m * p, p * n
    rank = 32
    
    T_true = np.zeros((nm, mp, pn))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_true[i, j, k] = 1.0
    
    x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    # Precompute tensor entries for efficient residual computation
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_entries.append((a, b, c))
    
    tensor_a = np.array([e[0] for e in tensor_entries])
    tensor_b = np.array([e[1] for e in tensor_entries])
    tensor_c = np.array([e[2] for e in tensor_entries])
    
    for target_col in range(rank):
        if time.time() - start_time > time_budget * 0.75:
            break
        
        def objective_epigraph(params, lam=1.0, beta=80.0):
            U_flat = params[:nm * rank]
            V_flat = params[nm * rank : nm * rank + mp * rank]
            W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
            
            U = U_flat.reshape(nm, rank)
            V = V_flat.reshape(mp, rank)
            W = W_flat.reshape(pn, rank)
            
            # Compute only the tensor entries we care about
            residuals = np.zeros(len(tensor_entries))
            for idx in range(len(tensor_entries)):
                a, b, c = tensor_a[idx], tensor_b[idx], tensor_c[idx]
                val = np.dot(np.dot(U[a], V[b]), W[c])
                residuals[idx] = val - 1.0
            
            # Also check that non-tensor entries are approximately zero
            # Sample a few non-tensor entries
            non_tensor_count = 0
            non_tensor_residuals = []
            for _ in range(20):
                a = np.random.randint(0, nm)
                b = np.random.randint(0, mp)
                c = np.random.randint(0, pn)
                i_from_a = a // m
                j_from_a = a % m
                j_from_b = b // p
                k_from_b = b % p
                k_from_c = c // n
                i_from_c = c % n
                if not (j_from_a == j_from_b and k_from_b == k_from_c and i_from_a == i_from_c):
                    val = np.dot(np.dot(U[a], V[b]), W[c])
                    non_tensor_residuals.append(val)
                    non_tensor_count += 1
            
            if non_tensor_count > 0:
                non_tensor_res = np.array(non_tensor_residuals)
                all_res = np.concatenate([residuals, non_tensor_res])
            else:
                all_res = residuals
            
            smooth_max = _soft_max_abs(all_res, beta=beta)
            
            col_penalty = np.sum(U[:, target_col]**2) + np.sum(V[:, target_col]**2) + np.sum(W[:, target_col]**2)
            
            return smooth_max + lam * col_penalty
        
        for lam in [0.01, 0.05, 0.1, 0.5, 1.0]:
            if time.time() - start_time > time_budget * 0.85:
                break
            
            try:
                result = minimize(objective_epigraph, x0, args=(lam, 80.0), method='L-BFGS-B', 
                                  options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            except Exception:
                continue
            
            U_flat = result.x[:nm * rank]
            V_flat = result.x[nm * rank : nm * rank + mp * rank]
            W_flat = result.x[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
            U = U_flat.reshape(nm, rank)
            V = V_flat.reshape(mp, rank)
            W = W_flat.reshape(pn, rank)
            
            col_norm = np.sqrt(np.sum(U[:, target_col]**2) + np.sum(V[:, target_col]**2) + np.sum(W[:, target_col]**2))
            
            if col_norm < 0.02:
                U31 = np.delete(U, target_col, axis=1)
                V31 = np.delete(V, target_col, axis=1)
                W31 = np.delete(W, target_col, axis=1)
                score31 = _compute_residual(U31, V31, W31, n, m, p)
                if score31 < 1e-6:
                    return U31, V31, W31, score31
    
    return None

def _polish_rank31(U31, V31, W31, n, m, p, time_budget):
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n
    rank = 31
    
    T_true = np.zeros((nm, mp, pn))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_true[i, j, k] = 1.0
    
    def objective_polish(params):
        U_flat = params[:nm * rank]
        V_flat = params[nm * rank : nm * rank + mp * rank]
        W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
        
        U = U_flat.reshape(nm, rank)
        V = V_flat.reshape(mp, rank)
        W = W_flat.reshape(pn, rank)
        
        M = np.einsum('ir,jr,kr->ijk', U, V, W)
        residual = M - T_true
        return np.sum(residual**2)
    
    x0 = np.concatenate([U31.flatten(), V31.flatten(), W31.flatten()])
    
    try:
        result = minimize(objective_polish, x0, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-14})
    except Exception:
        return None
    
    U_flat = result.x[:nm * rank]
    V_flat = result.x[nm * rank : nm * rank + mp * rank]
    W_flat = result.x[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
    U = U_flat.reshape(nm, rank)
    V = V_flat.reshape(mp, rank)
    W = W_flat.reshape(pn, rank)
    
    score = _compute_residual(U, V, W, n, m, p)
    if score < 1e-10:
        return U, V, W, score
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    U, V, W = _build_rank32()
    
    def to_list(arr):
        return arr.tolist()
    
    remaining_time = deadline - time.time()
    if remaining_time > 30:
        search_budget = min(remaining_time - 25, 70)
        result = _try_rank31_search(U, V, W, n, m, p, search_budget)
        if result is not None:
            U31, V31, W31, score = result
            if score < 1e-6:
                remaining2 = deadline - time.time()
                if remaining2 > 15:
                    polished = _polish_rank31(U31, V31, W31, n, m, p, min(remaining2 - 10, 30))
                    if polished is not None:
                        U31p, V31p, W31p, scorep = polished
                        return {"U": to_list(U31p), "V": to_list(V31p), "W": to_list(W31p)}
                return {"U": to_list(U31), "V": to_list(V31), "W": to_list(W31)}
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
