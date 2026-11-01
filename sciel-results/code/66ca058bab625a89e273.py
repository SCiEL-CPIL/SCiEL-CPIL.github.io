import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 4]

def _strassen_2x2():
    """Return the 7 Strassen rank-1 terms for 2x2 matrix multiplication.
    Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    Each term: (u_coeffs, v_coeffs, w_coeffs) as lists of (local_idx, value)
    """
    return [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

def _build_rank28():
    """Build the rank-28 Strassen-based decomposition."""
    n, m, p = DIMENSIONS
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    strassen = _strassen_2x2()
    
    u_map_L = [0, 1, 4, 5]
    u_map_R = [2, 3, 6, 7]
    v_map_TL = [0, 1, 4, 5]
    v_map_TR = [2, 3, 6, 7]
    v_map_BL = [8, 9, 12, 13]
    v_map_BR = [10, 11, 14, 15]
    w_map_L = [0, 2, 1, 3]
    w_map_R = [4, 6, 5, 7]
    
    col = 0
    for u_map, v_map, w_map in [(u_map_L, v_map_TL, w_map_L),
                                  (u_map_R, v_map_BL, w_map_L),
                                  (u_map_L, v_map_TR, w_map_R),
                                  (u_map_R, v_map_BR, w_map_R)]:
        for u_t, v_t, w_t in strassen:
            for idx, val in u_t:
                U[u_map[idx], col] += val
            for idx, val in v_t:
                V[v_map[idx], col] += val
            for idx, val in w_t:
                W[w_map[idx], col] += val
            col += 1
    
    return U, V, W

def _verify(U, V, W, n=2, m=4, p=4, tol=1e-9):
    """Verify the tensor decomposition."""
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    
    Cf = np.zeros((n, p))
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    return np.allclose(C, Cf, atol=tol)

def _tensor_from_factors(U, V, W, n, m, p):
    """Compute tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]"""
    T = np.zeros((n, m, p), dtype=float)
    for c in range(U.shape[1]):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    return T

def _try_rank_reduction(U, V, W, target_rank, start_time, time_limit=145.0):
    """Try to reduce rank from current to target_rank.
    
    Strategy: For each set of columns to eliminate, try to absorb their
    tensor contribution into the remaining columns via least squares.
    """
    n, m, p = DIMENSIONS
    r = U.shape[1]
    if r <= target_rank:
        return None
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build target tensor
    T_target = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i, j, k] = 1.0 if j == k else 0.0
    
    # Try different subsets of columns to keep
    # Use random sampling for efficiency
    np.random.seed(42)
    
    max_attempts = 50
    attempts = 0
    
    while attempts < max_attempts and time.time() - start_time < time_limit - 10.0:
        attempts += 1
        
        # Randomly select which columns to keep
        keep_cols = np.random.choice(r, size=target_rank, replace=False)
        elim_cols = np.setdiff1d(np.arange(r), keep_cols)
        
        U_keep = U[:, keep_cols].copy()
        V_keep = V[:, keep_cols].copy()
        W_keep = W[:, keep_cols].copy()
        
        U_elim = U[:, elim_cols]
        V_elim = V[:, elim_cols]
        W_elim = W[:, elim_cols]
        
        # Compute residual tensor from eliminated columns
        T_resid = _tensor_from_factors(U_elim, V_elim, W_elim, n, m, p)
        
        # We need: tensor(U_keep, V_keep, W_keep) = T_target - T_resid
        T_needed = T_target - T_resid
        
        # Parameters: flatten U_keep, V_keep, W_keep
        n_params = U_keep.size + V_keep.size + W_keep.size
        params = np.concatenate([U_keep.ravel(), V_keep.ravel(), W_keep.ravel()])
        
        def residuals(params):
            Uf = params[:U_keep.size].reshape(U_keep.shape)
            Vf = params[U_keep.size:U_keep.size+V_keep.size].reshape(V_keep.shape)
            Wf = params[U_keep.size+V_keep.size:].reshape(W_keep.shape)
            Tt = _tensor_from_factors(Uf, Vf, Wf, n, m, p)
            return (Tt - T_needed).ravel()
        
        # Use least squares
        try:
            from scipy.optimize import least_squares
            result = least_squares(residuals, params, method='lm', 
                                    max_nfev=3000, xtol=1e-10)
            
            if result.success or np.max(np.abs(result.fun)) < 1e-7:
                Uf = result.x[:U_keep.size].reshape(U_keep.shape)
                Vf = result.x[U_keep.size:U_keep.size+V_keep.size].reshape(V_keep.shape)
                Wf = result.x[U_keep.size+V_keep.size:].reshape(W_keep.shape)
                
                # Verify
                if _verify(Uf, Vf, Wf, tol=1e-6):
                    return Uf, Vf, Wf
        except Exception:
            pass
    
    return None

def _try_ilp_reduction(U, V, W, start_time, time_limit=145.0):
    """Try to find a lower-rank decomposition using a structured approach.
    
    For 2x4x4, we can exploit the structure more carefully.
    The key insight: C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can try various block decompositions and optimize within them.
    """
    n, m, p = DIMENSIONS
    
    # Try a structured decomposition where we group columns by their
    # support patterns in U, V, W
    
    # For now, try direct optimization from scratch for various ranks
    T_target = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i, j, k] = 1.0 if j == k else 0.0
    
    # Try rank 24-27 with random initialization
    for target_rank in [27, 26, 25, 24]:
        if time.time() - start_time > time_limit - 15.0:
            break
        
        best_result = None
        best_residual = float('inf')
        
        n_restarts = 10
        for restart in range(n_restarts):
            if time.time() - start_time > time_limit - 15.0:
                break
            
            np.random.seed(42 + restart)
            
            # Random initialization with small values
            U_init = np.random.randn(n * m, target_rank) * 0.5
            V_init = np.random.randn(m * p, target_rank) * 0.5
            W_init = np.random.randn(p * n, target_rank) * 0.5
            
            params = np.concatenate([U_init.ravel(), V_init.ravel(), W_init.ravel()])
            
            def residuals(params):
                Uf = params[:U_init.size].reshape(U_init.shape)
                Vf = params[U_init.size:U_init.size+V_init.size].reshape(V_init.shape)
                Wf = params[U_init.size+V_init.size:].reshape(W_init.shape)
                Tt = _tensor_from_factors(Uf, Vf, Wf, n, m, p)
                return (Tt - T_target).ravel()
            
            try:
                from scipy.optimize import least_squares
                result = least_squares(residuals, params, method='lm', 
                                        max_nfev=2000, xtol=1e-10)
                
                if np.max(np.abs(result.fun)) < best_residual:
                    best_residual = np.max(np.abs(result.fun))
                    if best_residual < 1e-7:
                        Uf = result.x[:U_init.size].reshape(U_init.shape)
                        Vf = result.x[U_init.size:U_init.size+V_init.size].reshape(V_init.shape)
                        Wf = result.x[U_init.size+V_init.size:].reshape(W_init.shape)
                        
                        if _verify(Uf, Vf, Wf, tol=1e-6):
                            return Uf, Vf, Wf
                        else:
                            best_result = (Uf, Vf, Wf)
                        
                        if best_residual < 1e-8:
                            break
            except Exception:
                pass
        
        if best_result is not None and best_residual < 1e-6:
            Uf, Vf, Wf = best_result
            if _verify(Uf, Vf, Wf, tol=1e-5):
                return Uf, Vf, Wf
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 145.0  # safety margin
    
    # Start with rank-28 Strassen decomposition
    U, V, W = _build_rank28()
    
    # Verify it's correct
    if not _verify(U, V, W):
        # Fallback: shouldn't happen, but return what we have
        if time.time() - start_time > time_limit:
            raise TimeoutError("Construction took too long")
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = 28
    
    # Try to reduce rank using column elimination
    for target_rank in [27, 26, 25]:
        if time.time() - start_time > time_limit - 20.0:
            break
        
        result = _try_rank_reduction(U, V, W, target_rank, start_time, time_limit)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = target_rank
    
    # Try ILP-style structured optimization
    if time.time() - start_time < time_limit - 30.0:
        result = _try_ilp_reduction(U, V, W, start_time, time_limit)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = U.shape[1]
    
    if time.time() - start_time > time_limit:
        raise TimeoutError("Construction took too long")
    
    # Round to nearest half-integer if close
    def round_half(x):
        return np.round(x * 2) / 2
    
    # Check if we can round to integers/half-integers
    U_out = best_U
    V_out = best_V
    W_out = best_W
    
    # Try rounding
    U_r = round_half(U_out)
    V_r = round_half(V_out)
    W_r = round_half(W_out)
    
    if _verify(U_r, V_r, W_r, tol=1e-6):
        U_out, V_out, W_out = U_r, V_r, W_r
    
    # Convert to list
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    # Test with random matrices to ensure correctness
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
            
    if not np.allclose(C, Cf):
        print("Verification Failed")
        print(C)
        print(Cf)
    else:
        print("Verification Passed")
    print("Rank:", U.shape[1])
