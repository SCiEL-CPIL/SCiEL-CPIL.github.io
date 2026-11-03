import time
import numpy as np

START_TIME = time.time()
TIME_LIMIT = 130  # safety margin before 150s kill

def time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def verify_decomposition(U, V, W, n=2, m=4, p=4, tol=1e-9):
    """Verify tensor decomposition exactly."""
    rank = U.shape[1]
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                T_recon[i, j, k] = s
    T_target = np.ones((n, m, p), dtype=float)
    return np.max(np.abs(T_recon - T_target)) < tol

def construct_strassen():
    """Build rank-28 Strassen decomposition for 2x4 * 4x4."""
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen 2x2x2 steps: (u_dict, v_dict, w_dict)
    # Local indices 0..3 correspond to (row, col) in 2x2
    strassen_steps = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    def place_block(a_col_start, b_row_start, b_col_start, c_col_start, col_offset):
        for s in range(7):
            u_dict, v_dict, w_dict = strassen_steps[s]
            col = col_offset + s
            for li, coeff in u_dict.items():
                lr, lc = li // 2, li % 2
                U[lr * m + (a_col_start + lc), col] += coeff
            for li, coeff in v_dict.items():
                lr, lc = li // 2, li % 2
                V[(b_row_start + lr) * p + (b_col_start + lc), col] += coeff
            for li, coeff in w_dict.items():
                lr, lc = li // 2, li % 2
                W[(c_col_start + lc) * n + lr, col] += coeff

    # 4 blocks: A cols {0,1}x{2,3}, B rows {0,1}x{2,3}, B cols {0,1}x{2,3}
    place_block(0, 0, 0, 0, 0)
    place_block(0, 0, 2, 2, 7)
    place_block(2, 2, 0, 0, 14)
    place_block(2, 2, 2, 2, 21)
    return U, V, W

def tensor_loss(U, V, W, n=2, m=4, p=4):
    """Smooth L2 loss for the decomposition."""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
    T_target = np.ones((n, m, p), dtype=float)
    diff = T - T_target
    return np.sum(diff * diff)

def tensor_loss_grad(U, V, W, n=2, m=4, p=4):
    """Compute loss and gradients w.r.t. U, V, W."""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
    T_target = np.ones((n, m, p), dtype=float)
    E = T - T_target  # error
    loss = np.sum(E * E)
    
    gU = np.zeros_like(U)
    gV = np.zeros_like(V)
    gW = np.zeros_like(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                d = 2.0 * E[i, j, k]
                gU[i*m+j, :] += d * V[j*p+k, :] * W[k*n+i, :]
                gV[j*p+k, :] += d * U[i*m+j, :] * W[k*n+i, :]
                gW[k*n+i, :] += d * U[i*m+j, :] * V[j*p+k, :]
    return loss, gU, gV, gW

def optimize_decomposition(U0, V0, W0, n=2, m=4, p=4, max_iter=2000, lr=0.01, tol=1e-9, time_budget=20):
    """Gradient descent with adaptive learning rate to minimize loss."""
    U = U0.copy()
    V = V0.copy()
    W = W0.copy()
    start_t = time.time()
    current_loss = tensor_loss(U, V, W, n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_loss = current_loss
    lr_current = lr
    patience = 0
    
    for it in range(max_iter):
        if time.time() - start_t > time_budget:
            break
        loss, gU, gV, gW = tensor_loss_grad(U, V, W, n, m, p)
        
        # Adaptive learning rate
        grad_norm = np.sqrt(np.sum(gU**2) + np.sum(gV**2) + np.sum(gW**2))
        if grad_norm > 1e-12:
            # Scale lr based on loss/grad to keep step size reasonable
            target_step = 0.1 * np.sqrt(loss) if loss > 1e-6 else 0.01
            lr_current = min(lr_current * 2, max(1e-6, target_step / (grad_norm + 1e-12)))
        
        U -= lr_current * gU
        V -= lr_current * gV
        W -= lr_current * gW
        
        new_loss = tensor_loss(U, V, W, n, m, p)
        if new_loss < best_loss:
            best_loss = new_loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            patience = 0
            # Occasionally reduce lr
            if it % 200 == 0:
                lr_current *= 0.95
        else:
            patience += 1
            U = U0.copy() if patience > 50 else U  # don't fully reset, just continue
            if patience > 200:
                break
        
        if best_loss < tol:
            break
    
    return best_U, best_V, best_W, best_loss

def try_reduce_to_rank(U, V, W, target_rank, n=2, m=4, p=4, time_budget=25):
    """Try to find a valid decomposition of exactly target_rank."""
    start_t = time.time()
    current_rank = U.shape[1]
    if target_rank >= current_rank:
        return U, V, W, tensor_loss(U, V, W, n, m, p)
    
    # Strategy: start from current decomposition, zero out (current_rank - target_rank) columns,
    # then optimize.
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_loss = tensor_loss(U, V, W, n, m, p)
    
    # Try multiple random subsets of columns to keep
    np.random.seed(12345)
    n_tries = 8
    for t in range(n_tries):
        if time.time() - start_t > time_budget:
            break
        if target_rank >= current_rank:
            break
        
        # Keep target_rank columns
        keep = np.random.choice(current_rank, size=target_rank, replace=False)
        keep.sort()
        U_t = U[:, keep].copy()
        V_t = V[:, keep].copy()
        W_t = W[:, keep].copy()
        
        # Initialize: scale up to compensate for removed columns
        scale = current_rank / target_rank
        U_t *= scale
        V_t *= scale
        W_t *= scale
        
        U_o, V_o, W_o, loss_o = optimize_decomposition(
            U_t, V_t, W_t, n, m, p,
            max_iter=3000, lr=0.02, tol=1e-9,
            time_budget=max(1, time_budget - (time.time() - start_t))
        )
        if loss_o < best_loss:
            best_loss = loss_o
            best_U, best_V, best_W = U_o, V_o, W_o
        
        if best_loss < 1e-9:
            break
    
    return best_U, best_V, best_W, best_loss

def polish(U, V, W, n=2, m=4, p=4, time_budget=5):
    """High-precision polish: tighten to exact."""
    start_t = time.time()
    U_c, V_c, W_c = U.copy(), V.copy(), W.copy()
    for _ in range(50):
        if time.time() - start_t > time_budget:
            break
        loss = tensor_loss(U_c, V_c, W_c, n, m, p)
        if loss < 1e-14:
            break
        _, gU, gV, gW = tensor_loss_grad(U_c, V_c, W_c, n, m, p)
        grad_norm = np.sqrt(np.sum(gU**2) + np.sum(gV**2) + np.sum(gW**2))
        if grad_norm < 1e-15:
            break
        # Newton-like step with small lr
        lr = min(0.001, 0.01 * np.sqrt(loss) / (grad_norm + 1e-15))
        U_c -= lr * gU
        V_c -= lr * gV
        W_c -= lr * gW
    return U_c, V_c, W_c

def construct():
    n, m, p = 2, 4, 4
    # Start from Strassen rank 28
    U, V, W = construct_strassen()
    
    # Verify base
    if not verify_decomposition(U, V, W, n, m, p, tol=1e-8):
        raise ValueError("Base decomposition failed verification")
    
    # Try to reduce to rank 27
    U27, V27, W27, loss27 = try_reduce_to_rank(U, V, W, 27, n, m, p, time_budget=30)
    if loss27 < 1e-9:
        U27, V27, W27 = polish(U27, V27, W27, n, m, p, time_budget=5)
        if verify_decomposition(U27, V27, W27, n, m, p, tol=1e-9):
            return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
    
    # Try to reduce to rank 26
    if not time_up():
        U26, V26, W26, loss26 = try_reduce_to_rank(U, V, W, 26, n, m, p, time_budget=40)
        if loss26 < 1e-9:
            U26, V26, W26 = polish(U26, V26, W26, n, m, p, time_budget=5)
            if verify_decomposition(U26, V26, W26, n, m, p, tol=1e-9):
                return {"U": U26.tolist(), "V": V26.tolist(), "W": W26.tolist()}
    
    # Try to reduce to rank 25
    if not time_up():
        U25, V25, W25, loss25 = try_reduce_to_rank(U, V, W, 25, n, m, p, time_budget=30)
        if loss25 < 1e-9:
            U25, V25, W25 = polish(U25, V25, W25, n, m, p, time_budget=5)
            if verify_decomposition(U25, V25, W25, n, m, p, tol=1e-9):
                return {"U": U25.tolist(), "V": V25.tolist(), "W": W25.tolist()}
    
    # Fallback: return Strassen rank 28
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
