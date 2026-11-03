import time
import numpy as np
import scipy.optimize as opt

_START_TIME = time.time()
_DEADLINE = 140  # Safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check if U,V,W form a valid decomposition with integer/half-integer entries."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
        
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check integer/half-integer constraint first (cheap)
    # Entries must be in 0.5 * Z
    U_h = U_arr * 2
    V_h = V_arr * 2
    W_h = W_arr * 2
    
    if not np.allclose(U_h, np.round(U_h), atol=1e-5):
        return False
    if not np.allclose(V_h, np.round(V_h), atol=1e-5):
        return False
    if not np.allclose(W_h, np.round(W_h), atol=1e-5):
        return False

    # Check exact decomposition
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # Indices: a = i*m+j, b = j*p+k, c = k*n+i
    # T is 1 if indices are consistent, 0 otherwise.
    
    # Efficient check using einsum
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Build target tensor for comparison
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                
    if np.max(np.abs(T_recon - T_target)) > tol:
        return False
        
    return True

def _solve_rank(rank, n, m, p, seed=42):
    """
    Attempt to find a rank-R decomposition using scipy.optimize.least_squares.
    """
    rng = np.random.RandomState(seed)
    
    # Flatten U, V, W into a single vector x
    # Sizes: n*m, m*p, p*n
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    x0 = rng.randn(size_u * rank + size_v * rank + size_w * rank)
    
    # Reshape x0 to U, V, W
    U0 = x0[:size_u * rank].reshape(size_u, rank)
    V0 = x0[size_u * rank : size_u * rank + size_v * rank].reshape(size_v, rank)
    W0 = x0[size_u * rank + size_v * rank :].reshape(size_w, rank)
    
    # Normalize columns to keep scale stable
    for r in range(rank):
        n_u = np.linalg.norm(U0[:, r])
        n_v = np.linalg.norm(V0[:, r])
        n_w = np.linalg.norm(W0[:, r])
        if n_u > 1e-8: U0[:, r] /= n_u
        if n_v > 1e-8: V0[:, r] /= n_v
        if n_w > 1e-8: W0[:, r] /= n_w

    def objective(x):
        U = x[:size_u * rank].reshape(size_u, rank)
        V = x[size_u * rank : size_u * rank + size_v * rank].reshape(size_v, rank)
        W = x[size_u * rank + size_v * rank :].reshape(size_w, rank)
        
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        # We only need to check the non-zero entries of the target tensor to save time?
        # No, we need the full error to drive the gradient.
        # But we can compute error only on the 24 non-zero entries if we unroll?
        # The target tensor is sparse.
        # T_target[i*m+j, j*p+k, k*n+i] = 1
        # All other entries are 0.
        
        # For speed, let's compute the error on the non-zero entries and the zero entries.
        # Actually, np.einsum is fast enough for these small sizes.
        # But we can be smarter.
        
        # Let's just use the full tensor difference for simplicity, it's small (24*12*8 = 2304 elements)
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0
        
        err = T_recon - T_target
        return err.flatten()

    # Use least_squares
    try:
        result = opt.least_squares(objective, x0, method='lm', max_nfev=1000)
    except Exception:
        # LM might fail if not full rank or due to numerical issues, try default
        try:
            result = opt.least_squares(objective, x0, max_nfev=1000)
        except Exception:
            return None

    x_opt = result.x
    U = x_opt[:size_u * rank].reshape(size_u, rank)
    V = x_opt[size_u * rank : size_u * rank + size_v * rank].reshape(size_v, rank)
    W = x_opt[size_u * rank + size_v * rank :].reshape(size_w, rank)
    
    # Round to nearest 0.5
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    # Verify
    if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
        
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = 2, 3, 4
    schoolbook_rank = n * m * p
    
    # The known rank is 23.
    # We will try 23 first with multiple seeds.
    # If that fails, we try 24 (which is guaranteed to work via schoolbook).
    
    ranks_to_try = [23]
    
    # Try rank 23 with many seeds
    num_seeds = 200
    for seed in range(num_seeds):
        if time.time() > _DEADLINE - 10:
            break
            
        res = _solve_rank(23, n, m, p, seed=seed)
        if res is not None:
            U, V, W = res
            # Double check verification
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
                
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
