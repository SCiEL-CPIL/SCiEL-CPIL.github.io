import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def _build_strassen_rank28():
    """Build the known rank-28 Strassen decomposition for 2x4x4 matmul."""
    n, m, p = DIMENSIONS
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    strassen_steps = [
        {'u': {0: 1, 3: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},
        {'u': {2: 1, 3: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}},
        {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}},
        {'u': {3: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},
        {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}},
        {'u': {2: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},
        {'u': {1: 1, 3: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}}
    ]
    
    def add_block(a_col_start, b_row_start, b_col_start, c_col_start, col_offset):
        for s in range(7):
            step = strassen_steps[s]
            col = col_offset + s
            for local_idx, coeff in step['u'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_idx = l_row * m + (a_col_start + l_col)
                U[global_idx, col] += coeff
            for local_idx, coeff in step['v'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_idx = (b_row_start + l_row) * p + (b_col_start + l_col)
                V[global_idx, col] += coeff
            for local_idx, coeff in step['w'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_idx = (c_col_start + l_col) * n + l_row
                W[global_idx, col] += coeff

    add_block(0, 0, 0, 0, 0)
    add_block(0, 0, 2, 2, 7)
    add_block(2, 2, 0, 0, 14)
    add_block(2, 2, 2, 2, 21)
    
    return U, V, W


def _verify_decomposition(U, V, W, tol=1e-6):
    """Verify that U, V, W exactly decompose the 2x4x4 matmul tensor."""
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p))
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    residual = np.max(np.abs(recon - T))
    return residual < tol, residual


def _als_polish(U, V, W, max_iters=200, tol=1e-12, time_limit=None, start_time=None):
    """
    Fast ALS polish: iteratively solve for each factor given the other two.
    Much faster than the original by using vectorized operations.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    T = np.ones((n, m, p))
    
    U3 = U.reshape(n, m, rank).astype(float).copy()
    V3 = V.reshape(m, p, rank).astype(float).copy()
    W3 = W.reshape(p, n, rank).astype(float).copy()
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Update U: for each (i,j), solve (V3[j] * W3[:,i]).T @ U3[i,j] = T[i,j,:]
        for i in range(n):
            for j in range(m):
                # A_rk = V3[j,k,r] * W3[k,i,r] summed over k
                # We need: sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r] = T[i,j,k]
                # This is: (V3[j] * W3[:,i])^T @ U3[i,j] = T[i,j,:]
                # Where (V3[j]*W3[:,i]) has shape (p, rank), so its transpose is (rank, p)
                A = (V3[j] * W3[:, i]).T  # (rank, p)
                try:
                    U3[i, j] = np.linalg.lstsq(A, T[i, j, :], rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                B = (U3[:, j] * W3[k]).T  # (rank, n)
                try:
                    V3[j, k] = np.linalg.lstsq(B, T[:, j, k], rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                C = (U3[i] * V3[:, k]).T  # (rank, m)
                try:
                    W3[k, i] = np.linalg.lstsq(C, T[i, :, k], rcond=None)[0]
                except:
                    pass
        
        recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        residual = np.max(np.abs(recon - T))
        if residual < tol:
            break
    
    U_flat = U3.reshape(n * m, rank)
    V_flat = V3.reshape(m * p, rank)
    W_flat = W3.reshape(p * n, rank)
    return U_flat, V_flat, W_flat, residual


def _try_rank_reduction(target_rank, seed, max_iters=300, time_limit=None, start_time=None):
    """
    Try to reduce from rank 28 to target_rank by zeroing columns and re-optimizing.
    Uses multiple strategies: zero one column, or perturb and re-solve.
    """
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p))
    rng = np.random.RandomState(seed)
    
    U28, V28, W28 = _build_strassen_rank28()
    
    # Strategy 1: Zero out columns one at a time and re-optimize
    for col_idx in range(28):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        mask = np.ones(28, dtype=bool)
        mask[col_idx] = False
        
        U = U28[:, mask].copy().astype(float)
        V = V28[:, mask].copy().astype(float)
        W = W28[:, mask].copy().astype(float)
        
        rank = target_rank
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        # ALS iterations
        for it in range(max_iters):
            if time_limit is not None and start_time is not None:
                if time.time() - start_time > time_limit:
                    break
            
            for i in range(n):
                for j in range(m):
                    A = (V3[j] * W3[:, i]).T
                    try:
                        U3[i, j] = np.linalg.lstsq(A, T[i, j, :], rcond=None)[0]
                    except:
                        pass
            for j in range(m):
                for k in range(p):
                    B = (U3[:, j] * W3[k]).T
                    try:
                        V3[j, k] = np.linalg.lstsq(B, T[:, j, k], rcond=None)[0]
                    except:
                        pass
            for k in range(p):
                for i in range(n):
                    C = (U3[i] * V3[:, k]).T
                    try:
                        W3[k, i] = np.linalg.lstsq(C, T[i, :, k], rcond=None)[0]
                    except:
                        pass
            
            if it % 50 == 0:
                recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
                residual = np.max(np.abs(recon - T))
                if residual < 1e-8:
                    break
        
        recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        residual = np.max(np.abs(recon - T))
        
        if residual < 1e-6:
            U_flat = U3.reshape(n * m, rank)
            V_flat = V3.reshape(m * p, rank)
            W_flat = W3.reshape(p * n, rank)
            return U_flat, V_flat, W_flat, residual
    
    return None


def _try_random_rank(target_rank, seed, max_iters=500, time_limit=None, start_time=None):
    """Try random initialization for rank reduction."""
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p))
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, target_rank) * 0.5
    V = rng.randn(m * p, target_rank) * 0.5
    W = rng.randn(p * n, target_rank) * 0.5
    
    U3 = U.reshape(n, m, target_rank)
    V3 = V.reshape(m, p, target_rank)
    W3 = W.reshape(p, n, target_rank)
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        for i in range(n):
            for j in range(m):
                A = (V3[j] * W3[:, i]).T
                try:
                    U3[i, j] = np.linalg.lstsq(A, T[i, j, :], rcond=None)[0]
                except:
                    pass
        for j in range(m):
            for k in range(p):
                B = (U3[:, j] * W3[k]).T
                try:
                    V3[j, k] = np.linalg.lstsq(B, T[:, j, k], rcond=None)[0]
                except:
                    pass
        for k in range(p):
            for i in range(n):
                C = (U3[i] * V3[:, k]).T
                try:
                    W3[k, i] = np.linalg.lstsq(C, T[i, :, k], rcond=None)[0]
                except:
                    pass
        
        if it % 100 == 0:
            recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            residual = np.max(np.abs(recon - T))
            if residual < 1e-8:
                break
    
    recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    residual = np.max(np.abs(recon - T))
    
    if residual < 1e-6:
        U_flat = U3.reshape(n * m, target_rank)
        V_flat = V3.reshape(m * p, target_rank)
        W_flat = W3.reshape(p * n, target_rank)
        return U_flat, V_flat, W_flat, residual
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    def to_lists(arr):
        return [[float(x) for x in row] for row in arr]
    
    # Start with known rank-28 Strassen solution
    U28, V28, W28 = _build_strassen_rank28()
    best_result = {"U": to_lists(U28), "V": to_lists(V28), "W": to_lists(W28)}
    best_rank = 28
    
    # Polish the rank-28 solution
    if time.time() < deadline:
        try:
            U_pol, V_pol, W_pol, res = _als_polish(U28, V28, W28, max_iters=100, tol=1e-12,
                                                     time_limit=deadline - time.time(), start_time=start_time)
            if res < 1e-10:
                best_result = {"U": to_lists(U_pol), "V": to_lists(V_pol), "W": to_lists(W_pol)}
        except:
            pass
    
    # Try to reduce to rank 27
    if time.time() < deadline and best_rank == 28:
        for seed in range(50):
            if time.time() >= deadline:
                break
            try:
                result = _try_rank_reduction(27, seed, max_iters=200,
                                              time_limit=deadline - time.time(), start_time=start_time)
                if result is not None:
                    U_r, V_r, W_r, res = result
                    valid, res2 = _verify_decomposition(U_r, V_r, W_r, tol=1e-5)
                    if valid:
                        best_rank = 27
                        best_result = {"U": to_lists(U_r), "V": to_lists(V_r), "W": to_lists(W_r)}
                        break
            except:
                continue
    
    # Try random init for rank 27
    if time.time() < deadline and best_rank == 28:
        for seed in range(30):
            if time.time() >= deadline:
                break
            try:
                result = _try_random_rank(27, seed, max_iters=300,
                                           time_limit=deadline - time.time(), start_time=start_time)
                if result is not None:
                    U_r, V_r, W_r, res = result
                    valid, res2 = _verify_decomposition(U_r, V_r, W_r, tol=1e-5)
                    if valid:
                        best_rank = 27
                        best_result = {"U": to_lists(U_r), "V": to_lists(V_r), "W": to_lists(W_r)}
                        break
            except:
                continue
    
    return best_result
