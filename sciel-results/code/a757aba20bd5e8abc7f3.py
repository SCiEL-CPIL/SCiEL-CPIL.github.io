import time
import sys
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [2, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 140  # Safety margin before 150s kill

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

def residuals(x, n=2, m=4, p=4, rank=24):
    """Calculate residuals for the optimization problem."""
    U = x[:n*m*rank].reshape(n*m, rank)
    V = x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
    W = x[n*m*rank + m*p*rank : ].reshape(p*n, rank)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
    
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
    
    for restart in range(max_restarts):
        if time_up():
            break
            
        # Random initialization
        x0 = np.random.randn(total_size) * 0.1
        
        try:
            result = opt.least_squares(residuals, x0, args=(n, m, p, target_rank), method='lm', max_nfev=10000)
            err = np.linalg.norm(result.fun)
            
            if err < best_error:
                best_error = err
                best_result = result.x
                
            if err < 1e-9:
                # Found a solution
                U = best_result[:size_U].reshape(n*m, target_rank)
                V = best_result[size_U : size_U + size_V].reshape(m*p, target_rank)
                W = best_result[size_U + size_V : ].reshape(p*n, target_rank)
                
                # Verify
                valid, _ = verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p)
                if valid:
                    return U.tolist(), V.tolist(), W.tolist()
                    
        except Exception as e:
            pass
            
    return None

def try_reduce_from_strassen(n=2, m=4, p=4, target_rank=27):
    """
    Try to find a decomposition of target_rank by starting from Strassen (rank 28)
    and optimizing a subset of variables after zeroing out one column.
    This is a heuristic to find rank 27, 26, 25, 24.
    """
    U_s, V_s, W_s = strassen_decomposition()
    U_s = np.array(U_s, dtype=float)
    V_s = np.array(V_s, dtype=float)
    W_s = np.array(W_s, dtype=float)
    rank_s = 28
    
    # We want to reduce rank by (rank_s - target_rank)
    # Let's try removing the last (rank_s - target_rank) columns
    # and optimizing the remaining ones.
    # This is a strong constraint: the removed columns must be linearly dependent
    # on the remaining ones in the tensor sense.
    
    cols_to_keep = target_rank
    cols_to_remove = rank_s - target_rank
    
    # We will fix the first `cols_to_keep` columns to be optimized,
    # and the rest are set to 0.
    # Actually, it's better to try different subsets of columns to remove.
    # For simplicity, we try removing the last few.
    
    # Initialize x with the kept columns
    x0 = np.zeros((n*m + m*p + p*n) * rank_s)
    
    # Fill x0 with Strassen values for kept columns
    # U part
    for r in range(cols_to_keep):
        x0[r * (n*m + m*p + p*n) : r * (n*m + m*p + p*n) + n*m] = U_s[:, r]
        x0[r * (n*m + m*p + p*n) + n*m : r * (n*m + m*p + p*n) + n*m + m*p] = V_s[:, r]
        x0[r * (n*m + m*p + p+n) + n*m + m*p : (r+1) * (n*m + m*p + p*n)] = W_s[:, r]
        
    # Note: The residuals function expects x to be flattened U, V, W for a given rank.
    # My x0 construction above is interleaved per column, which is NOT what residuals expects.
    # residuals expects: [U_flat, V_flat, W_flat] where U_flat is n*m*rank.
    
    # Let's rebuild x0 correctly for the residuals function.
    x0 = np.zeros(n*m*target_rank + m*p*target_rank + p*n*target_rank)
    
    # We will use the first `target_rank` columns of Strassen as initial guess.
    x0[:n*m*target_rank] = U_s[:, :target_rank].flatten()
    x0[n*m*target_rank : n*m*target_rank + m*p*target_rank] = V_s[:, :target_rank].flatten()
    x0[n*m*target_rank + m*p*target_rank :] = W_s[:, :target_rank].flatten()
    
    # Now optimize
    try:
        result = opt.least_squares(residuals, x0, args=(n, m, p, target_rank), method='lm', max_nfev=5000)
        err = np.linalg.norm(result.fun)
        if err < 1e-9:
            U = result.x[:n*m*target_rank].reshape(n*m, target_rank)
            V = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W = result.x[n*m*target_rank + m*p*target_rank : ].reshape(p*n, target_rank)
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
    
    # Try to find lower ranks using the new heuristic
    for r in [27, 26, 25, 24]:
        if time_up():
            break
        res = try_reduce_from_strassen(n, m, p, r)
        if res is not None:
            U2, V2, W2 = res
            valid, err = verify_decomposition(U2, V2, W2, n, m, p)
            if valid:
                best_rank = r
                best_result = {"U": U2, "V": V2, "W": W2}
                # If we find 24, we can stop. If we find 27, we might still try 26, 25, 24?
                # Given time limits, if we find 27, it's a big improvement.
                # But let's try to go lower if time permits.
                if r == 24:
                    break
                # If we found 27, we can try 26, 25, 24 in sequence.
                
    # Fallback to random search if time permits and we haven't found 24
    if best_rank > 24 and not time_up():
        for r in [24]:
            if time_up():
                break
            res = find_low_rank_decomposition(r, n, m, p, max_restarts=10)
            if res is not None:
                U2, V2, W2 = res
                valid, err = verify_decomposition(U2, V2, W2, n, m, p)
                if valid:
                    best_rank = r
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
