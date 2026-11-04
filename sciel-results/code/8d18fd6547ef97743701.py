import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition is valid."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check shapes
    if U_arr.shape != (n * m, rank) or V_arr.shape != (m * p, rank) or W_arr.shape != (p * n, rank):
        return False
        
    # Check all (i,j,k) entries equal 1
    # Use einsum for efficiency: T_ijk = sum_r U_im_jr V_jp_kr W_kn_ir
    # We want T_ijk == 1 for all i,j,k
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # Compute tensor
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    if np.allclose(T, 1.0, atol=tol):
        return True
    return False

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _try_rank(n, m, p, rank, seed, deadline, max_iter=200):
    """
    Try to find a rank-r decomposition using iterative least squares.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize with small random values
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5
    
    T = np.ones((n, m, p))
    T_flat = T.flatten()
    
    # Precompute index mappings
    # For each (i,j,k), the row indices in U, V, W and the tensor index
    nmp = n * m * p
    u_rows = []
    v_rows = []
    w_rows = []
    t_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_rows.append(i * m + j)
                v_rows.append(j * p + k)
                w_rows.append(k * n + i)
                t_indices.append(i * m * p + j * p + k)
    
    u_rows = np.array(u_rows)
    v_rows = np.array(v_rows)
    w_rows = np.array(w_rows)
    t_indices = np.array(t_indices)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Update U: for each (i,j,k), T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Fix V and W, solve for U
        # Build the linear system: for each tensor entry t, sum_r U[u_row, r] * (V[v_row, r] * W[w_row, r]) = 1
        # This is a linear system in U_flat
        # M_U[t, (u_row * rank + r)] = V[v_row, r] * W[w_row, r]
        
        # Efficient construction using broadcasting
        # For each tensor entry t with (i,j,k):
        #   u_row = i*m+j, v_row = j*p+k, w_row = k*n+i
        #   For each r: coefficient of U[u_row, r] is V[v_row, r] * W[w_row, r]
        
        # Build M_U as sparse-like using advanced indexing
        # M_U has shape (n*m*p, n*m*rank)
        # Column (u_row * rank + r) gets value V[v_row, r] * W[w_row, r] for row t
        
        # Vectorized approach:
        # For each r, the contribution to M_U[:, u_row*rank+r] is V[v_row, r] * W[w_row, r]
        M_U = np.zeros((nmp, n * m * rank))
        for r in range(rank):
            coeff = V[v_rows, r] * W[w_rows, r]
            cols = u_rows * rank + r
            M_U[np.arange(nmp), cols] = coeff
        
        U_flat, _, _, _ = np.linalg.lstsq(M_U, T_flat, rcond=None)
        U = U_flat.reshape(n * m, rank)
        
        # Update V
        M_V = np.zeros((nmp, m * p * rank))
        for r in range(rank):
            coeff = U[u_rows, r] * W[w_rows, r]
            cols = v_rows * rank + r
            M_V[np.arange(nmp), cols] = coeff
        
        V_flat, _, _, _ = np.linalg.lstsq(M_V, T_flat, rcond=None)
        V = V_flat.reshape(m * p, rank)
        
        # Update W
        M_W = np.zeros((nmp, p * n * rank))
        for r in range(rank):
            coeff = U[u_rows, r] * V[v_rows, r]
            cols = w_rows * rank + r
            M_W[np.arange(nmp), cols] = coeff
        
        W_flat, _, _, _ = np.linalg.lstsq(M_W, T_flat, rcond=None)
        W = W_flat.reshape(p * n, rank)
        
        # Compute error
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        T_calc = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        err = np.linalg.norm(T_calc - T)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-8:
            break
    
    # Try to round to half-integers
    for scale in [2, 4, 8]:
        U_rounded = np.round(best_U * scale) / scale
        V_rounded = np.round(best_V * scale) / scale
        W_rounded = np.round(best_W * scale) / scale
        
        if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    # If rounding failed, check if the continuous solution is close enough
    if best_err < 1e-6:
        # Try to find a nearby half-integer solution
        # This is a hard problem, but let's try a simple approach
        U_h = np.round(best_U * 2) / 2
        V_h = np.round(best_V * 2) / 2
        W_h = np.round(best_W * 2) / 2
        
        # Check if this is valid
        if _verify(U_h.tolist(), V_h.tolist(), W_h.tolist(), n, m, p):
            return U_h.tolist(), V_h.tolist(), W_h.tolist()
    
    return None

def _structured_init(n, m, p, rank, seed):
    """
    Try structured initializations that might be closer to valid decompositions.
    """
    rng = np.random.RandomState(seed)
    
    # Strategy 1: Random
    U = rng.randn(n * m, rank) * 0.3
    V = rng.randn(m * p, rank) * 0.3
    W = rng.randn(p * n, rank) * 0.3
    return U, V, W

def _try_rank_fast(n, m, p, rank, seed, deadline, max_iter=50):
    """
    Faster rank-r decomposition using alternating least squares with
    better initialization and early stopping.
    """
    rng = np.random.RandomState(seed)
    
    # Better initialization: start with sparse-like structure
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    T = np.ones((n, m, p))
    T_flat = T.flatten()
    
    # Precompute index mappings
    nmp = n * m * p
    u_rows = np.array([i * m + j for i in range(n) for j in range(m) for k in range(p)])
    v_rows = np.array([j * p + k for i in range(n) for j in range(m) for k in range(p)])
    w_rows = np.array([k * n + i for i in range(n) for j in range(m) for k in range(p)])
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(max_iter):
        if time.time() > deadline - 10:
            break
        
        # Update U
        M_U = np.zeros((nmp, n * m * rank))
        for r in range(rank):
            coeff = V[v_rows, r] * W[w_rows, r]
            cols = u_rows * rank + r
            M_U[np.arange(nmp), cols] = coeff
        
        U_flat, _, _, _ = np.linalg.lstsq(M_U, T_flat, rcond=None)
        U = U_flat.reshape(n * m, rank)
        
        # Update V
        M_V = np.zeros((nmp, m * p * rank))
        for r in range(rank):
            coeff = U[u_rows, r] * W[w_rows, r]
            cols = v_rows * rank + r
            M_V[np.arange(nmp), cols] = coeff
        
        V_flat, _, _, _ = np.linalg.lstsq(M_V, T_flat, rcond=None)
        V = V_flat.reshape(m * p, rank)
        
        # Update W
        M_W = np.zeros((nmp, p * n * rank))
        for r in range(rank):
            coeff = U[u_rows, r] * V[v_rows, r]
            cols = w_rows * rank + r
            M_W[np.arange(nmp), cols] = coeff
        
        W_flat, _, _, _ = np.linalg.lstsq(M_W, T_flat, rcond=None)
        W = W_flat.reshape(p * n, rank)
        
        # Compute error
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        T_calc = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        err = np.linalg.norm(T_calc - T)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-10:
            break
    
    # Try to round to half-integers
    for scale in [2, 4, 8]:
        U_rounded = np.round(best_U * scale) / scale
        V_rounded = np.round(best_V * scale) / scale
        W_rounded = np.round(best_W * scale) / scale
        
        if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Schoolbook verification failed")
    
    # Lower bound: max(n*m, m*p, p*n) = 24
    # For 3x4x6 matrix multiplication, known bounds suggest rank around 50-60
    # Try ranks from 70 down to 25, but focus on most likely range first
    
    # Use multiple seeds
    seeds = list(range(20))
    
    # Try most promising ranks first: 55-65 range
    promising_ranks = list(range(55, 71)) + list(range(45, 55)) + list(range(25, 45))
    
    for r in promising_ranks:
        if r >= best_rank:
            continue
        if time.time() > deadline - 15:
            break
        
        found = False
        for seed in seeds:
            if time.time() > deadline - 5:
                break
            
            result = _try_rank_fast(n, m, p, r, seed, deadline, max_iter=30)
            if result is not None:
                U_new, V_new, W_new = result
                if _verify(U_new, V_new, W_new, n, m, p):
                    best_result = {"U": U_new, "V": V_new, "W": W_new}
                    best_rank = r
                    found = True
                    break
        
        if found:
            # Found a better rank, continue to try lower ranks
            pass
    
    return best_result
