import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [2, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 120  # Safety margin before 150s kill

def time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def verify_decomposition(U, V, W, n=2, m=4, p=4):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
    
    T_target = np.ones((n, m, p), dtype=float)
    max_err = np.max(np.abs(T_recon - T_target))
    return max_err < 1e-9, max_err

def subgroup_decomposition_attempt():
    """Try to build a decomposition using shared linear forms between the two rows of A.
    
    The idea: for each rank-1 term, allow it to contribute to both i=0 and i=1 outputs.
    We parameterize U such that U[0,:] and U[1,:] share structure.
    """
    n, m, p = 2, 4, 4
    # Try rank 24 with structured initialization
    target_rank = 24
    
    U = np.zeros((n*m, target_rank))
    V = np.zeros((m*p, target_rank))
    W = np.zeros((p*n, target_rank))
    
    # Initialize with Strassen-like structure but with shared terms
    U_s = np.array(strassen_decomposition()[0], dtype=float)
    V_s = np.array(strassen_decomposition()[1], dtype=float)
    W_s = np.array(strassen_decomposition()[2], dtype=float)
    
    # Start from truncated Strassen
    U[:, :target_rank] = U_s[:, :target_rank]
    V[:, :target_rank] = V_s[:, :target_rank]
    W[:, :target_rank] = W_s[:, :target_rank]
    
    # Optimize with L-BFGS-B
    size_U = n * m * target_rank
    size_V = m * p * target_rank
    size_W = p * n * target_rank
    
    def objective(x):
        U_l = x[:size_U].reshape(n*m, target_rank)
        V_l = x[size_U : size_U + size_V].reshape(m*p, target_rank)
        W_l = x[size_U + size_V : ].reshape(p*n, target_rank)
        T_recon = np.zeros((n, m, p), dtype=float)
        for r in range(target_rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += U_l[i*m+j, r] * V_l[j*p+k, r] * W_l[k*n+i, r]
        return np.sum((T_recon - 1.0)**2)
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        result = opt.minimize(objective, x0, method='L-BFGS-B',
                               options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
        err = np.sqrt(result.fun)
        if err < 1e-10:
            U_f = result.x[:size_U].reshape(n*m, target_rank)
            V_f = result.x[size_U : size_U + size_V].reshape(m*p, target_rank)
            W_f = result.x[size_U + size_V : ].reshape(p*n, target_rank)
            valid, _ = verify_decomposition(U_f.tolist(), V_f.tolist(), W_f.tolist(), n, m, p)
            if valid:
                return U_f.tolist(), V_f.tolist(), W_f.tolist()
    except Exception:
        pass
    
    return None

def strassen_decomposition():
    """Return the Strassen-based decomposition (rank 28) - clean version."""
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen(A_r0, A_c0, B_r0, B_c0, C_r0, C_c0, start_r):
        a11 = A_r0 * m + A_c0
        a12 = A_r0 * m + A_c0 + 1
        a21 = (A_r0 + 1) * m + A_c0
        a22 = (A_r0 + 1) * m + A_c0 + 1
        
        b11 = B_r0 * p + B_c0
        b12 = B_r0 * p + B_c0 + 1
        b21 = (B_r0 + 1) * p + B_c0
        b22 = (B_r0 + 1) * p + B_c0 + 1
        
        c11_idx = C_c0 * n + C_r0
        c12_idx = (C_c0 + 1) * n + C_r0
        c21_idx = C_c0 * n + (C_r0 + 1)
        c22_idx = (C_c0 + 1) * n + (C_r0 + 1)
        
        r = start_r
        
        # M1 = (A11 + A22)(B11 + B22)
        U[a11][r] += 1
        U[a22][r] += 1
        V[b11][r] += 1
        V[b22][r] += 1
        W[c11_idx][r] += 1
        W[c22_idx][r] += 1
        r += 1
        
        # M2 = (A21 + A22)B11
        U[a21][r] += 1
        U[a22][r] += 1
        V[b11][r] += 1
        W[c21_idx][r] += 1
        W[c22_idx][r] += -1
        r += 1
        
        # M3 = A11(B12 - B22)
        U[a11][r] += 1
        V[b12][r] += 1
        V[b22][r] += -1
        W[c12_idx][r] += 1
        W[c22_idx][r] += 1
        r += 1
        
        # M4 = A22(B21 - B11)
        U[a22][r] += 1
        V[b21][r] += 1
        V[b11][r] += -1
        W[c11_idx][r] += 1
        W[c21_idx][r] += 1
        r += 1
        
        # M5 = (A11 + A12)B22
        U[a11][r] += 1
        U[a12][r] += 1
        V[b22][r] += 1
        W[c11_idx][r] += -1
        W[c12_idx][r] += 1
        r += 1
        
        # M6 = (A21 - A11)(B11 + B12)
        U[a21][r] += 1
        U[a11][r] += -1
        V[b11][r] += 1
        V[b12][r] += 1
        W[c22_idx][r] += 1
        r += 1
        
        # M7 = (A12 - A22)(B21 + B22)
        U[a12][r] += 1
        U[a22][r] += -1
        V[b21][r] += 1
        V[b22][r] += 1
        W[c11_idx][r] += 1
        r += 1
        
        return r
    
    r = add_strassen(0, 0, 0, 0, 0, 0, 0)
    r = add_strassen(0, 2, 2, 0, 0, 0, r)
    r = add_strassen(0, 0, 0, 2, 0, 2, r)
    r = add_strassen(0, 2, 2, 2, 0, 2, r)
    
    assert r == 28, f"Expected rank 28, got {r}"
    return U, V, W

def residuals(x, n=2, m=4, p=4, rank=24):
    U = x[:n*m*rank].reshape(n*m, rank)
    V = x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
    W = x[n*m*rank + m*p*rank : ].reshape(p*n, rank)
    
    # Efficient tensor reconstruction using einsum
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    T_recon = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        # For each rank-1 term, add outer product contribution
        U_r = U[:, r]  # shape (n*m,)
        V_r = V[:, r]  # shape (m*p,)
        W_r = W[:, r]  # shape (p*n,)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
    
    return T_recon.flatten() - 1.0

def try_rank_reduction(target_rank, n=2, m=4, p=4, max_restarts=30):
    """Try to reduce rank from Strassen's 28 to target_rank using multiple strategies."""
    if time_up():
        return None
    
    size_U = n * m * target_rank
    size_V = m * p * target_rank
    size_W = p * n * target_rank
    
    U_s = np.array(strassen_decomposition()[0], dtype=float)
    V_s = np.array(strassen_decomposition()[1], dtype=float)
    W_s = np.array(strassen_decomposition()[2], dtype=float)
    
    best_result = None
    best_error = float('inf')
    
    for restart in range(max_restarts):
        if time_up():
            break
        
        if restart == 0:
            # Direct truncation
            U_init = U_s[:, :target_rank].copy()
            V_init = V_s[:, :target_rank].copy()
            W_init = W_s[:, :target_rank].copy()
        elif restart == 1:
            # Truncation with small perturbation
            U_init = U_s[:, :target_rank].copy() + np.random.randn(n*m, target_rank) * 0.01
            V_init = V_s[:, :target_rank].copy() + np.random.randn(m*p, target_rank) * 0.01
            W_init = W_s[:, :target_rank].copy() + np.random.randn(p*n, target_rank) * 0.01
        elif restart == 2:
            # Random initialization
            U_init = np.random.randn(n*m, target_rank) * 0.1
            V_init = np.random.randn(m*p, target_rank) * 0.1
            W_init = np.random.randn(p*n, target_rank) * 0.1
        elif restart == 3:
            # Scaled random
            U_init = np.random.randn(n*m, target_rank) * 0.5
            V_init = np.random.randn(m*p, target_rank) * 0.5
            W_init = np.random.randn(p*n, target_rank) * 0.5
        else:
            # Larger perturbation
            scale = 0.02 + 0.005 * restart
            U_init = U_s[:, :target_rank].copy() + np.random.randn(n*m, target_rank) * scale
            V_init = V_s[:, :target_rank].copy() + np.random.randn(m*p, target_rank) * scale
            W_init = W_s[:, :target_rank].copy() + np.random.randn(p*n, target_rank) * scale
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = opt.least_squares(residuals, x0, args=(n, m, p, target_rank), 
                                       method='lm', max_nfev=3000)
            err = np.linalg.norm(result.fun)
            
            if err < best_error:
                best_error = err
                best_result = result.x.copy()
                
            if err < 1e-10:
                U = best_result[:size_U].reshape(n*m, target_rank)
                V = best_result[size_U : size_U + size_V].reshape(m*p, target_rank)
                W = best_result[size_U + size_V : ].reshape(p*n, target_rank)
                
                valid, _ = verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p)
                if valid:
                    return U.tolist(), V.tolist(), W.tolist()
                    
        except Exception:
            pass
            
    return None

def try_rank_reduction_2(target_rank, n=2, m=4, p=4, max_restarts=50):
    """Alternative rank reduction with BFGS and structured initialization."""
    if time_up():
        return None
    
    size_U = n * m * target_rank
    size_V = m * p * target_rank
    size_W = p * n * target_rank
    
    U_s = np.array(strassen_decomposition()[0], dtype=float)
    V_s = np.array(strassen_decomposition()[1], dtype=float)
    W_s = np.array(strassen_decomposition()[2], dtype=float)
    
    best_result = None
    best_error = float('inf')
    
    def objective(x):
        U = x[:size_U].reshape(n*m, target_rank)
        V = x[size_U : size_U + size_V].reshape(m*p, target_rank)
        W = x[size_U + size_V : ].reshape(p*n, target_rank)
        T_recon = np.zeros((n, m, p), dtype=float)
        for r in range(target_rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        return np.sum((T_recon - 1.0)**2)
    
    for restart in range(max_restarts):
        if time_up():
            break
        
        # Various initialization strategies
        strategy = restart % 6
        if strategy == 0:
            # Truncation
            U_init = U_s[:, :target_rank].copy()
            V_init = V_s[:, :target_rank].copy()
            W_init = W_s[:, :target_rank].copy()
        elif strategy == 1:
            # Small noise on truncation
            noise = 0.01
            U_init = U_s[:, :target_rank].copy() + np.random.randn(n*m, target_rank) * noise
            V_init = V_s[:, :target_rank].copy() + np.random.randn(m*p, target_rank) * noise
            W_init = W_s[:, :target_rank].copy() + np.random.randn(p*n, target_rank) * noise
        elif strategy == 2:
            # Random
            U_init = np.random.randn(n*m, target_rank) * 0.3
            V_init = np.random.randn(m*p, target_rank) * 0.3
            W_init = np.random.randn(p*n, target_rank) * 0.3
        elif strategy == 3:
            # Structured: share columns between U rows
            U_init = np.random.randn(n*m, target_rank) * 0.2
            # Make U[0,:] and U[1,:] partially correlated
            for r in range(target_rank):
                U_init[1, r] = 0.5 * U_init[0, r] + 0.5 * np.random.randn()
            V_init = np.random.randn(m*p, target_rank) * 0.2
            W_init = np.random.randn(p*n, target_rank) * 0.2
        elif strategy == 4:
            # Medium noise
            noise = 0.05
            U_init = U_s[:, :target_rank].copy() + np.random.randn(n*m, target_rank) * noise
            V_init = V_s[:, :target_rank].copy() + np.random.randn(m*p, target_rank) * noise
            W_init = W_s[:, :target_rank].copy() + np.random.randn(p*n, target_rank) * noise
        else:
            # Sparse structured
            U_init = np.zeros((n*m, target_rank))
            V_init = np.zeros((m*p, target_rank))
            W_init = np.zeros((p*n, target_rank))
            for r in range(target_rank):
                # Each rank-1 term touches a few entries
                idx_u = np.random.choice(n*m, size=2, replace=False)
                idx_v = np.random.choice(m*p, size=3, replace=False)
                idx_w = np.random.choice(p*n, size=2, replace=False)
                U_init[idx_u[0], r] = np.random.choice([-1, 1])
                U_init[idx_u[1], r] = np.random.choice([-1, 1])
                V_init[idx_v[0], r] = np.random.choice([-1, 1])
                V_init[idx_v[1], r] = np.random.choice([-1, 1])
                V_init[idx_v[2], r] = np.random.choice([-1, 1])
                W_init[idx_w[0], r] = np.random.choice([-1, 1])
                W_init[idx_w[1], r] = np.random.choice([-1, 1])
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            # Use BFGS for better convergence
            result = opt.minimize(objective, x0, method='L-BFGS-B', 
                                   options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            err = np.sqrt(result.fun)
            
            if err < best_error:
                best_error = err
                best_result = result.x.copy()
                
            if err < 1e-10:
                U = best_result[:size_U].reshape(n*m, target_rank)
                V = best_result[size_U : size_U + size_V].reshape(m*p, target_rank)
                W = best_result[size_U + size_V : ].reshape(p*n, target_rank)
                
                valid, _ = verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p)
                if valid:
                    return U.tolist(), V.tolist(), W.tolist()
                    
        except Exception:
            pass
            
    return None

def construct():
    n, m, p = 2, 4, 4
    best_result = None
    best_rank = float('inf')
    
    # Start with Strassen
    U, V, W = strassen_decomposition()
    valid, err = verify_decomposition(U, V, W, n, m, p)
    if valid:
        best_rank = 28
        best_result = {"U": U, "V": V, "W": W}
    
    # Try ranks in order of increasing difficulty (closer to 28 first)
    # Allocate time budget: ~30s per rank attempt
    ranks_to_try = [27, 26, 25, 24, 23, 22, 21, 20]
    
    for target_r in ranks_to_try:
        if best_rank <= target_r or time_up():
            break
        
        # Check if we have enough time (at least 20s remaining)
        remaining = TIME_LIMIT - (time.time() - START_TIME)
        if remaining < 20:
            break
        
        # First try the subgroup approach for rank 24
        if target_r == 24 and not time_up():
            res = subgroup_decomposition_attempt()
            if res is not None:
                U2, V2, W2 = res
                valid, err = verify_decomposition(U2, V2, W2, n, m, p)
                if valid:
                    best_rank = target_r
                    best_result = {"U": U2, "V": V2, "W": W2}
                    continue
        
        for method in [try_rank_reduction, try_rank_reduction_2]:
            if time_up():
                break
            res = method(target_r, n, m, p)
            if res is not None:
                U2, V2, W2 = res
                valid, err = verify_decomposition(U2, V2, W2, n, m, p)
                if valid:
                    best_rank = target_r
                    best_result = {"U": U2, "V": V2, "W": W2}
                    break
    
    if best_result is None:
        # Fallback: naive decomposition
        rank = 32
        U = [[0.0] * rank for _ in range(n*m)]
        V = [[0.0] * rank for _ in range(m*p)]
        W = [[0.0] * rank for _ in range(p*n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1.0
                    V[j*p+k][idx] = 1.0
                    W[k*n+i][idx] = 1.0
                    idx += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Best rank: {len(result['U'][0])}")
    valid, err = verify_decomposition(result['U'], result['V'], result['W'])
    print(f"Valid: {valid}, Max error: {err}")
