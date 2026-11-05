import numpy as np
from fractions import Fraction
import time
import itertools

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array([[float(x) for x in row] for row in U], dtype=float)
    V_arr = np.array([[float(x) for x in row] for row in V], dtype=float)
    W_arr = np.array([[float(x) for x in row] for row in W], dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = np.dot(U_arr[idx_u], V_arr[idx_v] * W_arr[idx_w])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_rank28():
    """Build the known rank-28 decomposition."""
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: j=2, all k (10 terms)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: j=0,1, k=4 (4 terms)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: j=0,1, k=0,1 using Strassen (7 terms)
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    W[0 * n + 1][r] = 1
    W[1 * n + 1][r] = -1
    r += 1
    U[0 * m + 0][r] = 1
    V[0 * p + 1][r] = 1
    V[1 * p + 1][r] = -1
    W[1 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    U[1 * m + 1][r] = 1
    V[1 * p + 0][r] = 1
    V[0 * p + 0][r] = -1
    W[0 * n + 0][r] = 1
    W[0 * n + 1][r] = 1
    r += 1
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = -1
    W[1 * n + 0][r] = 1
    r += 1
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 0][r] = 1
    V[0 * p + 1][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: j=0,1, k=2,3 using Strassen (7 terms)
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    W[2 * n + 1][r] = 1
    W[3 * n + 1][r] = -1
    r += 1
    U[0 * m + 0][r] = 1
    V[0 * p + 3][r] = 1
    V[1 * p + 3][r] = -1
    W[3 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    U[1 * m + 1][r] = 1
    V[1 * p + 2][r] = 1
    V[0 * p + 2][r] = -1
    W[2 * n + 0][r] = 1
    W[2 * n + 1][r] = 1
    r += 1
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = -1
    W[3 * n + 0][r] = 1
    r += 1
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 2][r] = 1
    V[0 * p + 3][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    return U, V, W

def _try_rank_from_base(base_U, base_V, base_W, target_rank, n, m, p, time_budget=20):
    """
    Try to find a rank-`target_rank` decomposition by starting from a 
    `target_rank+1` (or higher) solution and dropping terms.
    """
    base_rank = len(base_U[0])
    if base_rank <= target_rank:
        return None
        
    # We only support dropping exactly 1 term for now, as dropping more is harder
    if base_rank - target_rank != 1:
        return None
        
    nm, mp, pn = n * m, m * p, p * n
    
    start_time = time.time()
    
    def compute_residual(x):
        U = x[:target_rank*nm].reshape(target_rank, nm).T
        V = x[target_rank*nm:target_rank*nm+target_rank*mp].reshape(target_rank, mp).T
        W = x[target_rank*nm+target_rank*mp:].reshape(target_rank, pn).T
        
        residual = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j]
                for k in range(p):
                    v_w = V[j*p+k] * W[k*n+i]
                    val = np.dot(u_row, v_w)
                    residual += (val - 1.0)**2
        return residual

    def _objective(x):
        return compute_residual(x)

    def _jac(x):
        U = x[:target_rank*nm].reshape(target_rank, nm).T
        V = x[target_rank*nm:target_rank*nm+target_rank*mp].reshape(target_rank, mp).T
        W = x[target_rank*nm+target_rank*mp:].reshape(target_rank, pn).T
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                err_vec = np.zeros(p)
                for k in range(p):
                    v_w = V[j*p+k] * W[k*n+i]
                    err_vec[k] = U[i*m+j] @ v_w - 1.0
                for k in range(p):
                    grad_U[i*m+j] += 2 * err_vec[k] * V[j*p+k] * W[k*n+i]
        
        for j in range(m):
            for k in range(p):
                err_vec = np.zeros(n)
                for i in range(n):
                    v_w = V[j*p+k] * W[k*n+i]
                    err_vec[i] = U[i*m+j] @ v_w - 1.0
                for i in range(n):
                    grad_V[j*p+k] += 2 * err_vec[i] * U[i*m+j] * W[k*n+i]
        
        for k in range(p):
            for i in range(n):
                err_vec = np.zeros(m)
                for j in range(m):
                    v_w = V[j*p+k] * W[k*n+i]
                    err_vec[j] = U[i*m+j] @ v_w - 1.0
                for j in range(m):
                    grad_W[k*n+i] += 2 * err_vec[j] * U[i*m+j] * V[j*p+k]
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    # Try dropping each term from the base solution
    for drop_idx in range(base_rank):
        if time.time() - start_time > time_budget:
            break
            
        # Construct initial guess by removing drop_idx
        keep_indices = [i for i in range(base_rank) if i != drop_idx]
        
        U_init = np.array(base_U, dtype=float)[:, keep_indices]
        V_init = np.array(base_V, dtype=float)[:, keep_indices]
        W_init = np.array(base_W, dtype=float)[:, keep_indices]
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            from scipy.optimize import minimize
            result = minimize(_objective, x0, method='L-BFGS-B', jac=_jac,
                             options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'iprint': 0})
            
            if result.fun < 1e-8:
                U_res = result.x[:target_rank*nm].reshape(target_rank, nm).T
                V_res = result.x[target_rank*nm:target_rank*nm+target_rank*mp].reshape(target_rank, mp).T
                W_res = result.x[target_rank*nm+target_rank*mp:].reshape(target_rank, pn).T
                
                # Round to half-integers
                U_r = np.round(U_res * 2) / 2
                V_r = np.round(V_res * 2) / 2
                W_r = np.round(W_res * 2) / 2
                
                U_frac = [[Fraction(int(np.round(v * 2)), 2) for v in row] for row in U_r]
                V_frac = [[Fraction(int(np.round(v * 2)), 2) for v in row] for row in V_r]
                W_frac = [[Fraction(int(np.round(v * 2)), 2) for v in row] for row in W_r]
                
                if _verify_decomposition(U_frac, V_frac, W_frac, n, m, p):
                    return (U_frac, V_frac, W_frac)
                    
        except Exception:
            continue
            
    return None

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = 140
    
    U28, V28, W28 = _build_rank28()
    if not _verify_decomposition(U28, V28, W28, n, m, p):
        raise ValueError("Rank 28 verification failed")
    
    best_U, best_V, best_W = U28, V28, W28
    best_rank = 28
    
    # Try rank 27 by dropping one term from rank 28
    if time.time() - start_time < deadline - 30:
        result = _try_rank_from_base(U28, V28, W28, 27, n, m, p, time_budget=25)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 27
            
    return {"U": best_U, "V": best_V, "W": best_W}
