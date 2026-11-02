import time
import sys
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [2, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 130  # Safety margin before 150s kill

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

def strassen_decomposition():
    """Return the Strassen-based decomposition (rank 28)."""
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    alpha = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (0,0): -1 },
        { (0,1): 1, (1,1): -1 }
    ]
    
    beta = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (0,1): 1, (1,1): -1 },
        { (1,0): 1, (0,0): -1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (1,1): 1 }
    ]
    
    w_coeff = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): -1 },
        { (0,1): 1, (1,1): 1 },
        { (0,0): 1, (1,0): 1 },
        { (0,0): -1, (0,1): 1 },
        { (1,1): 1 },
        { (0,0): 1 }
    ]
    
    for s in range(1, 8):
        for t in range(4):
            c = 4 * (s - 1) + t
            k_a = 0 if t in [0, 2] else 1
            r_b = 1 if t in [1, 3] else 0
            c_b = 1 if t in [2, 3] else 0
            k_c = 0 if t in [0, 1] else 1
            
            for (i, j), val in alpha[s].items():
                if val != 0:
                    u_idx = i * m + 2 * j + k_a
                    U[u_idx][c] += val
            
            for (i, j), val in beta[s].items():
                if val != 0:
                    v_row = 2 * i + r_b
                    v_col = 2 * j + c_b
                    v_idx = v_row * p + v_col
                    V[v_idx][c] += val
            
            for (i, j), val in w_coeff[s].items():
                if val != 0:
                    w_col = 2 * j + k_c
                    w_row = i
                    w_idx = w_col * n + w_row
                    W[w_idx][c] += val
    
    return U, V, W

def residuals_fast(x, n=2, m=4, p=4, rank=24):
    """Calculate residuals for the optimization problem, vectorized."""
    size_U = n * m * rank
    size_V = m * p * rank
    U = x[:size_U].reshape(n*m, rank)
    V = x[size_U : size_U + size_V].reshape(m*p, rank)
    W = x[size_U + size_V : ].reshape(p*n, rank)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    # Loop over rank is usually faster than complex einsum for small ranks if not careful,
    # but let's try to vectorize the rank sum.
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    # We can compute this by iterating over i, j, k and summing over r
    # Or iterating over r and adding outer products?
    # For small n,m,p, the python loop over i,j,k is 32 iterations.
    # The inner sum is vectorized over rank.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                
    return T_recon.flatten() - 1.0

def find_low_rank_decomposition(target_rank, n=2, m=4, p=4, max_restarts=50):
    """Use scipy.optimize.least_squares to find a decomposition of target_rank."""
    n, m, p = 2, 4, 4
    size_U = n * m * target_rank
    size_V = m * p * target_rank
    size_W = p * n * target_rank
    total_size = size_U + size_V + size_W
    
    best_result = None
    best_error = float('inf')
    
    # Pre-compute Strassen solution for initialization
    U_strassen, V_strassen, W_strassen = strassen_decomposition()
    U_s = np.array(U_strassen, dtype=float)
    V_s = np.array(V_strassen, dtype=float)
    W_s = np.array(W_strassen, dtype=float)
    
    for restart in range(max_restarts):
        if time_up():
            break
            
        # Initialization Strategy:
        # 1. Truncate Strassen (if rank < 28)
        # 2. Perturb Strassen
        # 3. Random Sparse
        
        if restart < 10:
            # Use Strassen-based initialization
            if target_rank <= 28:
                U_init = U_s[:, :target_rank].copy()
                V_init = V_s[:, :target_rank].copy()
                W_init = W_s[:, :target_rank].copy()
                # Add small noise to escape local minima
                noise_scale = 0.01 + 0.001 * restart
                U_init += np.random.randn(n*m, target_rank) * noise_scale
                V_init += np.random.randn(m*p, target_rank) * noise_scale
                W_init += np.random.randn(p*n, target_rank) * noise_scale
            else:
                U_init = np.hstack([U_s, np.random.randn(n*m, target_rank-28)*0.1])
                V_init = np.hstack([V_s, np.random.randn(m*p, target_rank-28)*0.1])
                W_init = np.hstack([W_s, np.random.randn(p*n, target_rank-28)*0.1])
        else:
            # Random Sparse Initialization
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            for r in range(target_rank):
                # Each rank component should be sparse to allow for linear independence
                # But not too sparse
                u_nonzero = np.random.choice(n*m, size=max(1, n*m//3), replace=False)
                v_nonzero = np.random.choice(m*p, size=max(1, m*p//3), replace=False)
                w_nonzero = np.random.choice(p*n, size=max(1, p*n//3), replace=False)
                U[u_nonzero, r] = np.random.choice([-1, 1], size=len(u_nonzero))
                V[v_nonzero, r] = np.random.choice([-1, 1], size=len(v_nonzero))
                W[w_nonzero, r] = np.random.choice([-1, 1], size=len(w_nonzero))
            U_init = U
            V_init = V
            W_init = W

        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            # Use 'trf' which is robust for large problems and handles bounds if needed (not needed here)
            # 'lm' is faster but can diverge. 'trf' is safer.
            result = opt.least_squares(residuals_fast, x0, args=(n, m, p, target_rank), method='trf', max_nfev=2000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            err = np.linalg.norm(result.fun)
            
            if err < best_error:
                best_error = err
                best_result = result.x
                
            if err < 1e-9:
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
    
    # Try ranks in increasing order
    for r in [24, 26, 27]:
        if best_rank <= r:
            continue
        if time_up():
            break
            
        res = find_low_rank_decomposition(r, n, m, p)
        if res is not None:
            U2, V2, W2 = res
            valid, err = verify_decomposition(U2, V2, W2, n, m, p)
            if valid:
                best_rank = r
                best_result = {"U": U2, "V": V2, "W": W2}
                break # Found the lowest possible in this list? No, 24 is lowest. If 24 fails, try 26.
                # Actually, if 24 is found, we break. If not, we continue to 26.
    
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
