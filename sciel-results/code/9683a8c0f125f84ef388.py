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


def _verify_decomposition(U, V, W, tol=1e-8):
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


def _gradient_optimize(U_init, V_init, W_init, target_rank, max_iters=2000, lr_init=0.01, seed=42, deadline=None):
    """Gradient-based optimization to find a rank-target_rank decomposition."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for j in range(m):
        T[:, j, j] = 1.0
    
    rng = np.random.RandomState(seed)
    
    # Start from perturbed version of the rank-28 solution, truncated to target_rank
    U = U_init[:, :target_rank].copy()
    V = V_init[:, :target_rank].copy()
    W = W_init[:, :target_rank].copy()
    
    # Add small random perturbation to escape local minima
    U += 0.1 * rng.randn(*U.shape)
    V += 0.1 * rng.randn(*V.shape)
    W += 0.1 * rng.randn(*W.shape)
    
    lr = lr_init
    best_residual = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        U3 = U.reshape(n, m, -1)
        V3 = V.reshape(m, p, -1)
        W3 = W.reshape(p, n, -1)
        
        # Compute reconstruction
        recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        residual = recon - T
        loss = np.sum(residual**2)
        
        if loss < best_residual:
            best_residual = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_residual < 1e-20:
            break
        
        # Compute gradients
        # dL/dU[i,j,r] = 2 * sum_{k} residual[i,j,k] * V3[j,k,r] * W3[k,i,r]
        dU3 = 2.0 * residual * np.einsum('jkr,kir->ijr', V3, W3)
        dV3 = 2.0 * residual * np.einsum('ijr,kir->jkr', U3, W3)
        dW3 = 2.0 * residual * np.einsum('ijr,jkr->kir', U3, V3)
        
        dU = dU3.reshape(n*m, -1)
        dV = dV3.reshape(m*p, -1)
        dW = dW3.reshape(p*n, -1)
        
        # Adaptive learning rate: reduce if loss increases
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Decay learning rate
        lr *= 0.999
        
        # Occasionally restart with smaller perturbation if stuck
        if it > 0 and it % 200 == 0 and loss > best_residual * 1.5:
            U = best_U + 0.05 * rng.randn(*best_U.shape)
            V = best_V + 0.05 * rng.randn(*best_V.shape)
            W = best_W + 0.05 * rng.randn(*best_W.shape)
            lr = lr_init * 0.5
    
    return best_U, best_V, best_W, best_residual


def _try_rank_reduction(target_rank, U28, V28, W28, max_restarts=8, seed=0, deadline=None):
    """Try to find a rank-target_rank decomposition starting from rank-28."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for j in range(m):
        T[:, j, j] = 1.0
    
    for restart in range(max_restarts):
        if deadline is not None and time.time() > deadline:
            break
        
        U_r, V_r, W_r, residual = _gradient_optimize(
            U28, V28, W28, target_rank, 
            max_iters=1500, lr_init=0.005, 
            seed=seed + restart, deadline=deadline
        )
        
        if residual < 1e-15:
            # Verify with the actual matrices
            valid, res = _verify_decomposition(U_r, V_r, W_r, tol=1e-10)
            if valid:
                return U_r, V_r, W_r, target_rank
    
    return None, None, None, None


def _try_column_merge(U28, V28, W28, target_rank, seed=0, deadline=None):
    """Try to merge columns of the rank-28 solution to reduce rank."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for j in range(m):
        T[:, j, j] = 1.0
    
    rng = np.random.RandomState(seed)
    current_rank = 28
    
    for attempt in range(28 - target_rank):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try removing a random column and re-optimizing
        col_to_remove = rng.randint(0, current_rank)
        
        mask = np.ones(current_rank, dtype=bool)
        mask[col_to_remove] = False
        
        U_sub = U28[:, mask].copy()
        V_sub = V28[:, mask].copy()
        W_sub = W28[:, mask].copy()
        
        # Now optimize with gradient descent to recover the missing contribution
        U_opt, V_opt, W_opt, residual = _gradient_optimize(
            U28, V28, W28, current_rank - 1,
            max_iters=1000, lr_init=0.003,
            seed=seed + attempt, deadline=deadline
        )
        
        if residual < 1e-15:
            valid, res = _verify_decomposition(U_opt, V_opt, W_opt, tol=1e-10)
            if valid:
                return U_opt, V_opt, W_opt, current_rank - 1
        
        current_rank -= 1
        if current_rank < target_rank:
            break
    
    return None, None, None, None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin before 150s limit
    
    # Start with known rank-28 solution
    U28, V28, W28 = _build_strassen_rank28()
    
    def to_lists(arr):
        return [[float(x) for x in row] for row in arr]
    
    best_result = {"U": to_lists(U28), "V": to_lists(V28), "W": to_lists(W28)}
    best_rank = 28
    
    # Try to reduce to rank 27 using gradient optimization
    if time.time() < deadline:
        for seed in range(15):
            if time.time() >= deadline:
                break
            try:
                U_r, V_r, W_r, rank_r = _try_rank_reduction(27, U28, V28, W28, max_restarts=1, seed=seed, deadline=deadline)
                if U_r is not None:
                    best_rank = 27
                    best_result = {"U": to_lists(U_r), "V": to_lists(V_r), "W": to_lists(W_r)}
                    break
            except Exception:
                continue
    
    # Try column merge approach
    if time.time() < deadline and best_rank == 28:
        for seed in range(10):
            if time.time() >= deadline:
                break
            try:
                U_r, V_r, W_r, rank_r = _try_column_merge(U28, V28, W28, 27, seed=seed, deadline=deadline)
                if U_r is not None:
                    best_rank = 27
                    best_result = {"U": to_lists(U_r), "V": to_lists(V_r), "W": to_lists(W_r)}
                    break
            except Exception:
                continue
    
    # If we found rank 27, try to go to rank 26
    if best_rank == 27 and time.time() < deadline:
        U_cur = np.array(best_result["U"])
        V_cur = np.array(best_result["V"])
        W_cur = np.array(best_result["W"])
        for seed in range(10):
            if time.time() >= deadline:
                break
            try:
                U_r, V_r, W_r, rank_r = _try_rank_reduction(26, U_cur, V_cur, W_cur, max_restarts=1, seed=seed, deadline=deadline)
                if U_r is not None:
                    best_rank = 26
                    best_result = {"U": to_lists(U_r), "V": to_lists(V_r), "W": to_lists(W_r)}
                    break
            except Exception:
                continue
    
    return best_result
# EVOLVE-BLOCK-END
