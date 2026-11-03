import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _build_strassen_rank28():
    """Build the known rank-28 Strassen decomposition."""
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
    """Verify that U, V, W exactly decompose the matmul tensor."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for j in range(m):
        T[:, j, j] = 1.0
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U3 = U_arr.reshape(n, m, -1)
    V3 = V_arr.reshape(m, p, -1)
    W3 = W_arr.reshape(p, n, -1)
    
    recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    residual = np.max(np.abs(recon - T))
    return residual < tol, residual


def _als_polish(U, V, W, max_iters=1000, tol=1e-12):
    """Polish a near-exact decomposition to exact using alternating least squares."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for j in range(m):
        T[:, j, j] = 1.0
    
    U3 = U.reshape(n, m, rank).copy()
    V3 = V.reshape(m, p, rank).copy()
    W3 = W.reshape(p, n, rank).copy()
    
    for it in range(max_iters):
        # Update U
        for i in range(n):
            for j in range(m):
                A = (V3[j] * W3[:, i]).T  # (rank, p)
                try:
                    U3[i, j] = np.linalg.lstsq(A, T[i, j], rcond=None)[0]
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


def _try_rank27_from_strassen(U28, V28, W28, max_trials=50, seed=0):
    """Try to find rank-27 by merging/perturbing Strassen columns."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for j in range(m):
        T[:, j, j] = 1.0
    
    rng = np.random.RandomState(seed)
    
    # Strategy 1: Try to express one column as a combination of others
    # and remove it
    for trial in range(max_trials):
        # Pick a column to try removing
        col_remove = rng.randint(0, 28)
        
        # Create rank-27 matrices by zeroing that column
        U27 = U28.copy()
        V27 = V28.copy()
        W27 = W28.copy()
        U27[:, col_remove] = 0
        V27[:, col_remove] = 0
        W27[:, col_remove] = 0
        
        # Remove the zero column
        mask = np.ones(28, dtype=bool)
        mask[col_remove] = False
        U27 = U27[:, mask]
        V27 = V27[:, mask]
        W27 = W27[:, mask]
        
        # Now try to recover with ALS
        U27, V27, W27, residual = _als_polish(U27, V27, W27, max_iters=500, tol=1e-10)
        
        if residual < 1e-8:
            valid, res = _verify_decomposition(U27, V27, W27, tol=1e-6)
            if valid:
                return U27, V27, W27
    
    # Strategy 2: Random perturbation of one column then polish
    for trial in range(max_trials):
        col_perturb = rng.randint(0, 28)
        scale = rng.uniform(0.1, 1.0)
        sign = rng.choice([-1, 1])
        
        U27 = U28.copy()
        V27 = V28.copy()
        W27 = W28.copy()
        
        # Perturb the column
        U27[:, col_perturb] += sign * scale * rng.randn(8)
        V27[:, col_perturb] += sign * scale * rng.randn(16)
        W27[:, col_perturb] += sign * scale * rng.randn(8)
        
        # Try to polish to rank 27 by seeing if we can zero one column
        # First, polish the perturbed rank-28
        U28p, V28p, W28p, res28 = _als_polish(U27, V27, W27, max_iters=200, tol=1e-10)
        
        # Now try removing each column
        for col_remove in range(28):
            U_test = U28p.copy()
            V_test = V28p.copy()
            W_test = W28p.copy()
            U_test[:, col_remove] = 0
            V_test[:, col_remove] = 0
            W_test[:, col_remove] = 0
            
            mask = np.ones(28, dtype=bool)
            mask[col_remove] = False
            U_test = U_test[:, mask]
            V_test = V_test[:, mask]
            W_test = W_test[:, mask]
            
            U_test, V_test, W_test, res_test = _als_polish(U_test, V_test, W_test, max_iters=300, tol=1e-10)
            
            if res_test < 1e-8:
                valid, _ = _verify_decomposition(U_test, V_test, W_test, tol=1e-6)
                if valid:
                    return U_test, V_test, W_test
    
    return None


def _direct_rank27_search(max_trials=30, seed=0):
    """Direct search for rank-27 decomposition."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for j in range(m):
        T[:, j, j] = 1.0
    
    rng = np.random.RandomState(seed)
    rank = 27
    
    for trial in range(max_trials):
        # Initialize with structure-aware random
        U = rng.randn(n * m, rank) * 0.5
        V = rng.randn(m * p, rank) * 0.5
        W = rng.randn(p * n, rank) * 0.5
        
        # Polish
        U, V, W, residual = _als_polish(U, V, W, max_iters=800, tol=1e-12)
        
        if residual < 1e-8:
            valid, _ = _verify_decomposition(U, V, W, tol=1e-6)
            if valid:
                return U, V, W
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with known rank-28 solution
    U28, V28, W28 = _build_strassen_rank28()
    
    def to_lists(arr):
        return [[float(x) for x in row] for row in arr]
    
    best_result = {"U": to_lists(U28), "V": to_lists(V28), "W": to_lists(W28)}
    best_rank = 28
    
    # Try to reduce to rank 27 from Strassen
    if time.time() < deadline:
        for seed in range(5):
            if time.time() >= deadline:
                break
            try:
                result = _try_rank27_from_strassen(U28, V28, W28, max_trials=20, seed=seed)
                if result is not None:
                    U27, V27, W27 = result
                    valid, res = _verify_decomposition(U27, V27, W27, tol=1e-6)
                    if valid:
                        best_rank = 27
                        best_result = {"U": to_lists(U27), "V": to_lists(V27), "W": to_lists(W27)}
                        break
            except Exception:
                continue
    
    # Try direct rank-27 search
    if time.time() < deadline and best_rank == 28:
        for seed in range(10):
            if time.time() >= deadline:
                break
            try:
                result = _direct_rank27_search(max_trials=5, seed=seed)
                if result is not None:
                    U27, V27, W27 = result
                    valid, res = _verify_decomposition(U27, V27, W27, tol=1e-6)
                    if valid:
                        best_rank = 27
                        best_result = {"U": to_lists(U27), "V": to_lists(V27), "W": to_lists(W27)}
                        break
            except Exception:
                continue
    
    return best_result
# EVOLVE-BLOCK-END
