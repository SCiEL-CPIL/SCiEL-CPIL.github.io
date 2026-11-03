import time
import numpy as np
from scipy.linalg import lstsq

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


def _als_decomposition(U_init, V_init, W_init, target_rank, max_iters=500, seed=42, deadline=None, tol=1e-12):
    """
    Vectorized Alternating Least Squares for tensor decomposition.
    """
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for j in range(m):
        T[:, j, j] = 1.0
    
    rng = np.random.RandomState(seed)
    
    U = U_init[:, :target_rank].copy().astype(float)
    V = V_init[:, :target_rank].copy().astype(float)
    W = W_init[:, :target_rank].copy().astype(float)
    
    U += 0.05 * rng.randn(*U.shape)
    V += 0.05 * rng.randn(*V.shape)
    W += 0.05 * rng.randn(*W.shape)
    
    best_residual = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        U3 = U.reshape(n, m, -1)
        V3 = V.reshape(m, p, -1)
        W3 = W.reshape(p, n, -1)
        
        recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        residual = recon - T
        loss = np.sum(residual**2)
        
        if loss < best_residual:
            best_residual = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_residual < tol:
            break
        
        # Vectorized ALS: Solve for U
        # For each (i,j): U[i*m+j,:] @ B_ij = T[i,j,:] where B_ij[r,k] = V[j*p+k,r]*W[k*n+i,r]
        # Vectorize: for all (i,j), build a batch of B matrices and solve
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # B_ij[r,k] = V[j*p+k, r] * W[k*n+i, r]
                V_j = V3[j, :, :]  # (p, rank)
                W_i = W3[:, i, :]  # (p, rank)
                B_ij = (V_j * W_i).T  # (rank, p)
                rhs = T[i, j, :]
                try:
                    sol, _, _, _ = lstsq(B_ij.T, rhs, rcond=None)
                    U_new[i * m + j, :] = sol
                except:
                    U_new[i * m + j, :] = U[i * m + j, :]
        U = U_new
        
        # Vectorized ALS: Solve for V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                # C_jk[r,i] = U[i*m+j, r] * W[k*n+i, r]
                U_j = U3[:, j, :]  # (n, rank)
                W_k = W3[k, :, :]  # (n, rank)
                C_jk = (U_j * W_k).T  # (rank, n)
                rhs = T[:, j, k]
                try:
                    sol, _, _, _ = lstsq(C_jk.T, rhs, rcond=None)
                    V_new[j * p + k, :] = sol
                except:
                    V_new[j * p + k, :] = V[j * p + k, :]
        V = V_new
        
        # Vectorized ALS: Solve for W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                # D_ki[r,j] = U[i*m+j, r] * V[j*p+k, r]
                U_i = U3[i, :, :]  # (m, rank)
                V_k = V3[:, k, :]  # (m, rank)
                D_ki = (U_i * V_k).T  # (rank, m)
                rhs = T[i, :, k]
                try:
                    sol, _, _, _ = lstsq(D_ki.T, rhs, rcond=None)
                    W_new[k * n + i, :] = sol
                except:
                    W_new[k * n + i, :] = W[k * n + i, :]
        W = W_new
    
    return best_U, best_V, best_W, best_residual


def _gradient_optimize(U_init, V_init, W_init, target_rank, max_iters=5000, lr_init=0.02, seed=42, deadline=None):
    """Adaptive gradient descent with momentum and better restarts."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for j in range(m):
        T[:, j, j] = 1.0
    
    rng = np.random.RandomState(seed)
    
    U = U_init[:, :target_rank].copy().astype(float)
    V = V_init[:, :target_rank].copy().astype(float)
    W = W_init[:, :target_rank].copy().astype(float)
    
    U += 0.1 * rng.randn(*U.shape)
    V += 0.1 * rng.randn(*V.shape)
    W += 0.1 * rng.randn(*W.shape)
    
    # Momentum buffers
    vU = np.zeros_like(U)
    vV = np.zeros_like(V)
    vW = np.zeros_like(W)
    beta1, beta2 = 0.9, 0.999
    mU, mV, mW = np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)
    vU2, vV2, vW2 = np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)
    
    lr = lr_init
    best_residual = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        U3 = U.reshape(n, m, -1)
        V3 = V.reshape(m, p, -1)
        W3 = W.reshape(p, n, -1)
        
        recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        residual = recon - T
        loss = np.sum(residual**2)
        
        if loss < best_residual:
            best_residual = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_residual < 1e-25:
            break
        
        dU3 = 2.0 * residual * np.einsum('jkr,kir->ijr', V3, W3)
        dV3 = 2.0 * residual * np.einsum('ijr,kir->jkr', U3, W3)
        dW3 = 2.0 * residual * np.einsum('ijr,jkr->kir', U3, V3)
        
        dU = dU3.reshape(n*m, -1)
        dV = dV3.reshape(m*p, -1)
        dW = dW3.reshape(p*n, -1)
        
        # Adam-style update
        mU = beta1 * mU + (1 - beta1) * dU
        mV = beta1 * mV + (1 - beta1) * dV
        mW = beta1 * mW + (1 - beta1) * dW
        vU2 = beta2 * vU2 + (1 - beta2) * (dU ** 2)
        vV2 = beta2 * vV2 + (1 - beta2) * (dV ** 2)
        vW2 = beta2 * vW2 + (1 - beta2) * (dW ** 2)
        
        eps = 1e-8
        U -= lr * mU / (np.sqrt(vU2) + eps)
        V -= lr * mV / (np.sqrt(vV2) + eps)
        W -= lr * mW / (np.sqrt(vW2) + eps)
        
        lr *= 0.999
        
        # More aggressive restart
        if it > 0 and it % 200 == 0 and loss > best_residual * 1.5:
            U = best_U + 0.02 * rng.randn(*best_U.shape)
            V = best_V + 0.02 * rng.randn(*best_V.shape)
            W = best_W + 0.02 * rng.randn(*best_W.shape)
            mU, mV, mW = np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)
            vU2, vV2, vW2 = np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)
            lr = lr_init * 0.3
    
    return best_U, best_V, best_W, best_residual


def _polish(U, V, W, max_iters=2000, deadline=None, tol=1e-15):
    """High-precision polish pass using vectorized ALS."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for j in range(m):
        T[:, j, j] = 1.0
    
    rank = U.shape[1]
    
    U = U.copy().astype(float)
    V = V.copy().astype(float)
    W = W.copy().astype(float)
    
    best_residual = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        U3 = U.reshape(n, m, -1)
        V3 = V.reshape(m, p, -1)
        W3 = W.reshape(p, n, -1)
        
        recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        residual = recon - T
        loss = np.sum(residual**2)
        
        if loss < best_residual:
            best_residual = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_residual < tol:
            break
        
        # Vectorized ALS for U
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                V_j = V3[j, :, :]
                W_i = W3[:, i, :]
                B_ij = (V_j * W_i).T
                rhs = T[i, j, :]
                try:
                    sol, _, _, _ = lstsq(B_ij.T, rhs, rcond=None)
                    U_new[i * m + j, :] = sol
                except:
                    U_new[i * m + j, :] = U[i * m + j, :]
        U = U_new
        
        # Vectorized ALS for V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                U_j = U3[:, j, :]
                W_k = W3[k, :, :]
                C_jk = (U_j * W_k).T
                rhs = T[:, j, k]
                try:
                    sol, _, _, _ = lstsq(C_jk.T, rhs, rcond=None)
                    V_new[j * p + k, :] = sol
                except:
                    V_new[j * p + k, :] = V[j * p + k, :]
        V = V_new
        
        # Vectorized ALS for W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                U_i = U3[i, :, :]
                V_k = V3[:, k, :]
                D_ki = (U_i * V_k).T
                rhs = T[i, :, k]
                try:
                    sol, _, _, _ = lstsq(D_ki.T, rhs, rcond=None)
                    W_new[k * n + i, :] = sol
                except:
                    W_new[k * n + i, :] = W[k * n + i, :]
        W = W_new
    
    return best_U, best_V, best_W, best_residual


def _try_rank_reduction(target_rank, U28, V28, W28, max_restarts=10, seed=0, deadline=None):
    """Try to find a rank-target_rank decomposition using ALS + gradient + polish."""
    n, m, p = DIMENSIONS
    
    for restart in range(max_restarts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Method 1: ALS
        U_r, V_r, W_r, residual = _als_decomposition(
            U28, V28, W28, target_rank,
            max_iters=300, seed=seed + restart, deadline=deadline
        )
        
        if residual < 1e-10:
            # Polish
            U_p, V_p, W_p, res_p = _polish(U_r, V_r, W_r, max_iters=500, deadline=deadline)
            if res_p < 1e-15:
                valid, _ = _verify_decomposition(U_p, V_p, W_p, tol=1e-10)
                if valid:
                    return U_p, V_p, W_p, target_rank
        
        # Method 2: Gradient
        U_g, V_g, W_g, res_g = _gradient_optimize(
            U28, V28, W28, target_rank,
            max_iters=2000, lr_init=0.005,
            seed=seed + restart + 100, deadline=deadline
        )
        
        if res_g < 1e-10:
            U_p, V_p, W_p, res_p = _polish(U_g, V_g, W_g, max_iters=500, deadline=deadline)
            if res_p < 1e-15:
                valid, _ = _verify_decomposition(U_p, V_p, W_p, tol=1e-10)
                if valid:
                    return U_p, V_p, W_p, target_rank
    
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
    
    # Try to reduce to rank 27 with more aggressive search
    if time.time() < deadline:
        for seed in range(30):
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
    
    # If we found rank 27, try to go to rank 26
    if best_rank == 27 and time.time() < deadline:
        U_cur = np.array(best_result["U"])
        V_cur = np.array(best_result["V"])
        W_cur = np.array(best_result["W"])
        for seed in range(20):
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
    
    # Final polish on the best result
    if time.time() < deadline:
        U_best = np.array(best_result["U"])
        V_best = np.array(best_result["V"])
        W_best = np.array(best_result["W"])
        U_pol, V_pol, W_pol, _ = _polish(U_best, V_best, W_best, max_iters=500, deadline=deadline)
        best_result = {"U": to_lists(U_pol), "V": to_lists(V_pol), "W": to_lists(W_pol)}
    
    return best_result
# EVOLVE-BLOCK-END
