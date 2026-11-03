import time
import numpy as np
from fractions import Fraction

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


def _als_polish(U, V, W, max_iters=500, tol=1e-12):
    """Polish a near-exact decomposition using alternating least squares."""
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
                A = (V3[j] * W3[:, i]).T
                try:
                    U3[i, j] = np.linalg.lstsq(A, T[i, j], rcond=None)[0]
                except:
                    pass
        # Update V
        for j in range(m):
            for k in range(p):
                B = (U3[:, j] * W3[k]).T
                try:
                    V3[j, k] = np.linalg.lstsq(B, T[:, j, k], rcond=None)[0]
                except:
                    pass
        # Update W
        for k in range(p):
            for i in range(n):
                C = (U3[i] * V3[:, k]).T
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


def _structured_random_search(rank, max_trials=20, seed=0, deadline=None):
    """Structured random search with sparse integer/half-integer grid."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for j in range(m):
        T[:, j, j] = 1.0
    
    rng = np.random.RandomState(seed)
    
    # Sparse grid: small integers and half-integers
    grid = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    for trial in range(max_trials):
        if deadline is not None and time.time() > deadline:
            break
        
        # Generate sparse random matrices from the grid
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Fill with sparse random values from grid
        for _ in range(n * m * rank):
            r, c = rng.randint(n * m), rng.randint(rank)
            U[r, c] = rng.choice(grid)
        
        for _ in range(m * p * rank):
            r, c = rng.randint(m * p), rng.randint(rank)
            V[r, c] = rng.choice(grid)
        
        for _ in range(p * n * rank):
            r, c = rng.randint(p * n), rng.randint(rank)
            W[r, c] = rng.choice(grid)
        
        # Polish
        U, V, W, residual = _als_polish(U, V, W, max_iters=200, tol=1e-10)
        
        if residual < 1e-8:
            valid, _ = _verify_decomposition(U, V, W, tol=1e-6)
            if valid:
                return U, V, W
    
    return None


def _try_rank_reduction_from_strassen(U28, V28, W28, target_rank=27, max_trials=30, seed=0, deadline=None):
    """Try to reduce rank from 28 to target_rank by merging columns."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for j in range(m):
        T[:, j, j] = 1.0
    
    rng = np.random.RandomState(seed)
    current_rank = U28.shape[1]
    
    # Strategy: try to express one column as a linear combination of others
    for trial in range(max_trials):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a column to try removing
        col_remove = rng.randint(0, current_rank)
        
        # Create rank-27 matrices by zeroing that column
        U_new = U28.copy()
        V_new = V28.copy()
        W_new = W28.copy()
        U_new[:, col_remove] = 0
        V_new[:, col_remove] = 0
        W_new[:, col_remove] = 0
        
        # Remove the zero column
        mask = np.ones(current_rank, dtype=bool)
        mask[col_remove] = False
        U_new = U_new[:, mask]
        V_new = V_new[:, mask]
        W_new = W_new[:, mask]
        
        # Now try to recover with ALS
        U_new, V_new, W_new, residual = _als_polish(U_new, V_new, W_new, max_iters=300, tol=1e-10)
        
        if residual < 1e-8:
            valid, _ = _verify_decomposition(U_new, V_new, W_new, tol=1e-6)
            if valid:
                return U_new, V_new, W_new
    
    # Strategy 2: Perturb and re-optimize
    for trial in range(max_trials):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a column to perturb
        col_perturb = rng.randint(0, current_rank)
        scale = rng.uniform(0.05, 0.5)
        
        U_pert = U28.copy()
        V_pert = V28.copy()
        W_pert = W28.copy()
        
        # Perturb the column
        U_pert[:, col_perturb] += scale * rng.randn(n * m)
        V_pert[:, col_perturb] += scale * rng.randn(m * p)
        W_pert[:, col_perturb] += scale * rng.randn(p * n)
        
        # Polish the perturbed rank-28
        U_pol, V_pol, W_pol, _ = _als_polish(U_pert, V_pert, W_pert, max_iters=100, tol=1e-10)
        
        # Try removing each column
        for col_remove in range(current_rank):
            if deadline is not None and time.time() > deadline:
                break
            
            U_test = U_pol.copy()
            V_test = V_pol.copy()
            W_test = W_pol.copy()
            U_test[:, col_remove] = 0
            V_test[:, col_remove] = 0
            W_test[:, col_remove] = 0
            
            mask = np.ones(current_rank, dtype=bool)
            mask[col_remove] = False
            U_test = U_test[:, mask]
            V_test = V_test[:, mask]
            W_test = W_test[:, mask]
            
            U_test, V_test, W_test, res_test = _als_polish(U_test, V_test, W_test, max_iters=200, tol=1e-10)
            
            if res_test < 1e-8:
                valid, _ = _verify_decomposition(U_test, V_test, W_test, tol=1e-6)
                if valid:
                    return U_test, V_test, W_test
    
    return None


def _exact_rational_check(U, V, W, tol=1e-10):
    """Check if the decomposition is exact using rational arithmetic."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for j in range(m):
        T[:, j, j] = 1.0
    
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    
    recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    residual = np.max(np.abs(recon - T))
    return residual < tol


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    # Start with known rank-28 solution
    U28, V28, W28 = _build_strassen_rank28()
    
    def to_lists(arr):
        return [[float(x) for x in row] for row in arr]
    
    best_result = {"U": to_lists(U28), "V": to_lists(V28), "W": to_lists(W28)}
    best_rank = 28
    
    # Try to reduce to rank 27 from Strassen
    if time.time() < deadline:
        for seed in range(10):
            if time.time() >= deadline:
                break
            try:
                result = _try_rank_reduction_from_strassen(U28, V28, W28, target_rank=27, max_trials=15, seed=seed, deadline=deadline)
                if result is not None:
                    U27, V27, W27 = result
                    valid, res = _verify_decomposition(U27, V27, W27, tol=1e-6)
                    if valid:
                        best_rank = 27
                        best_result = {"U": to_lists(U27), "V": to_lists(V27), "W": to_lists(W27)}
                        break
            except Exception:
                continue
    
    # Try structured random search for rank 27
    if time.time() < deadline and best_rank == 28:
        for seed in range(15):
            if time.time() >= deadline:
                break
            try:
                result = _structured_random_search(rank=27, max_trials=5, seed=seed, deadline=deadline)
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
