import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-9):
    """Check if U, V, W exactly decompose T."""
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    diff = T_recon - T
    return np.max(np.abs(diff)) < tol

def _snap_to_grid(arr, grid_step=0.5):
    """Snap values to nearest multiple of grid_step."""
    return np.round(arr / grid_step) * grid_step

def _local_search(U, V, W, T, grid_values, max_iters=20000, tol=1e-9, deadline=None):
    """Local search: flip individual entries to reduce error."""
    nm, mp, np_ = U.shape[0], V.shape[0], W.shape[0]
    r = U.shape[1]
    
    def compute_error():
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T_recon - T) ** 2)
    
    current_err = compute_error()
    if current_err < tol:
        return U, V, W, True
    
    stuck = 0
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        if stuck > 200:
            break
        
        # Pick random matrix, row, column
        mat_idx = np.random.randint(3)
        if mat_idx == 0:
            mat = U
            dim = nm
        elif mat_idx == 1:
            mat = V
            dim = mp
        else:
            mat = W
            dim = np_
        
        row = np.random.randint(dim)
        col = np.random.randint(r)
        
        old_val = mat[row, col]
        new_val = float(np.random.choice(grid_values))
        
        if new_val == old_val:
            continue
        
        mat[row, col] = new_val
        new_err = compute_error()
        
        if new_err < current_err:
            current_err = new_err
            stuck = 0
            if current_err < tol:
                return U, V, W, True
        else:
            mat[row, col] = old_val
            stuck += 1
    
    return U, V, W, current_err < tol

def _als_iteration(U, V, W, T, num_a, num_b, num_c, rank):
    """One full ALS iteration."""
    new_U = np.zeros_like(U)
    for a in range(num_a):
        M_a = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(num_b * num_c, rank)
        b_vec = T[a, :, :].reshape(-1)
        new_U[a, :] = np.linalg.lstsq(M_a, b_vec, rcond=None)[0]
    U = new_U
    
    new_V = np.zeros_like(V)
    for b in range(num_b):
        M_b = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(num_a * num_c, rank)
        b_vec = T[:, b, :].reshape(-1)
        new_V[b, :] = np.linalg.lstsq(M_b, b_vec, rcond=None)[0]
    V = new_V
    
    new_W = np.zeros_like(W)
    for c in range(num_c):
        M_c = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(num_a * num_b, rank)
        b_vec = T[:, :, c].reshape(-1)
        new_W[c, :] = np.linalg.lstsq(M_c, b_vec, rcond=None)[0]
    W = new_W
    
    return U, V, W

def _gradient_polish(U, V, W, T, num_a, num_b, num_c, rank, n_iter=100, lr=0.01):
    """Polish using gradient descent."""
    for it in range(n_iter):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_recon - T
        loss = np.sum(diff ** 2)
        if loss < 1e-15:
            break
        
        grad_U = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
        grad_V = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
        grad_W = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    T = _build_tensor()
    num_a = n * m  # 6
    num_b = m * p  # 21
    num_c = p * n  # 14
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((num_a, rank_sb))
    V_sb = np.zeros((num_b, rank_sb))
    W_sb = np.zeros((num_c, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank_sb
    best_result = (U_sb, V_sb, W_sb)
    
    # Grid values to try: integers and half-integers
    grid_int = [0, 1, -1]
    grid_half = [0, 0.5, -0.5, 1, -1]
    
    # Try ranks from a reasonable lower bound
    min_rank = 21  # max(num_a, num_b, num_c)
    max_rank_to_try = 42
    
    for rank in range(min_rank, max_rank_to_try + 1):
        if time.time() > deadline - 20:
            break
        
        found = False
        
        # Multiple restarts
        num_restarts = 8
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                break
            
            seed = 42 + restart * 7919
            np.random.seed(seed)
            
            # Initialize with random values in the grid
            U = np.random.choice(grid_half, size=(num_a, rank)).astype(np.float64)
            V = np.random.choice(grid_half, size=(num_b, rank)).astype(np.float64)
            W = np.random.choice(grid_half, size=(num_c, rank)).astype(np.float64)
            
            # ALS iterations to get close
            prev_loss = np.inf
            for it in range(80):
                if time.time() > deadline - 12:
                    break
                
                U, V, W = _als_iteration(U, V, W, T, num_a, num_b, num_c, rank)
                
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                loss = np.sum((T_recon - T) ** 2)
                
                if loss < 1e-12:
                    found = True
                    break
                
                if abs(prev_loss - loss) < 1e-14:
                    # Gradient polish
                    U, V, W = _gradient_polish(U, V, W, T, num_a, num_b, num_c, rank, n_iter=50, lr=0.005)
                    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                    loss = np.sum((T_recon - T) ** 2)
                    if loss < 1e-12:
                        found = True
                        break
                
                prev_loss = loss
            
            if found:
                best_rank = rank
                best_result = (U.copy(), V.copy(), W.copy())
                break
            
            # If ALS didn't find exact, try snapping to grid and local search
            U_snapped = _snap_to_grid(U, 0.5)
            V_snapped = _snap_to_grid(V, 0.5)
            W_snapped = _snap_to_grid(W, 0.5)
            
            # Verify snapped
            if _verify(U_snapped, V_snapped, W_snapped, T):
                best_rank = rank
                best_result = (U_snapped.copy(), V_snapped.copy(), W_snapped.copy())
                found = True
                break
            
            # Local search from snapped
            U_ls, V_ls, W_ls, success = _local_search(
                U_snapped.copy(), V_snapped.copy(), W_snapped.copy(),
                T, grid_half, max_iters=5000, tol=1e-9, deadline=deadline - 10
            )
            if success:
                best_rank = rank
                best_result = (U_ls.copy(), V_ls.copy(), W_ls.copy())
                found = True
                break
        
        if found:
            break
    
    U, V, W = best_result
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
