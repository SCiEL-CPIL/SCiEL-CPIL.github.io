"""Program for matrix multiplication tensor decomposition (3x4, 4x6)."""
import time
import numpy as np

# The dimensions: n x m * m x p = 3x4 * 4x6
n_dim = 3
m_dim = 4
p_dim = 6

def _verify_fast(U, V, W, tensor, tol=1e-9):
    """Fast verification using vectorization."""
    rank = U.shape[1]
    n, m, p = tensor.shape
    
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    try:
        reconstructed = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
        err = np.max(np.abs(reconstructed - tensor))
        return err < tol
    except Exception:
        return False

def _als_step_fast(U, V, W, tensor, n, m, p, rank):
    """Fast ALS step using vectorized operations."""
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # 1. Solve for U: for each (i,j), A = V_r[j] * W_r[:,i] (element-wise, shape p x rank)
    U_new = np.zeros((n, m, rank))
    for i in range(n):
        for j in range(m):
            A = V_r[j] * W_r[:, i]  # (p, rank)
            b = tensor[i, j, :]     # (p,)
            # Solve A @ x = b where x is (rank,)
            try:
                x = np.linalg.lstsq(A, b, rcond=None)[0]
                U_new[i, j, :] = x
            except np.linalg.LinAlgError:
                pass
    U = U_new.reshape(n * m, rank)
    U_r = U.reshape(n, m, rank)
    
    # 2. Solve for V: for each (j,k), A = U_r[:,j] * W_r[k] (element-wise, shape n x rank)
    V_new = np.zeros((m, p, rank))
    for j in range(m):
        for k in range(p):
            A = U_r[:, j] * W_r[k]  # (n, rank)
            b = tensor[:, j, k]     # (n,)
            try:
                x = np.linalg.lstsq(A, b, rcond=None)[0]
                V_new[j, k, :] = x
            except np.linalg.LinAlgError:
                pass
    V = V_new.reshape(m * p, rank)
    V_r = V.reshape(m, p, rank)
    
    # 3. Solve for W: for each (k,i), A = U_r[i] * V_r[:,k] (element-wise, shape m x rank)
    W_new = np.zeros((p, n, rank))
    for k in range(p):
        for i in range(n):
            A = U_r[i] * V_r[:, k]  # (m, rank)
            b = tensor[i, :, k]     # (m,)
            try:
                x = np.linalg.lstsq(A, b, rcond=None)[0]
                W_new[k, i, :] = x
            except np.linalg.LinAlgError:
                pass
    W = W_new.reshape(p * n, rank)
    
    return U, V, W

def _compute_error(U, V, W, tensor, n, m, p, rank):
    """Compute reconstruction error."""
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    reconstructed = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
    return np.max(np.abs(reconstructed - tensor))

def _polish(U, V, W, tensor, n, m, p, rank, max_iters=100, tol=1e-13):
    """Polish the solution to higher precision."""
    prev_err = np.inf
    for _ in range(max_iters):
        U, V, W = _als_step_fast(U, V, W, tensor, n, m, p, rank)
        err = _compute_error(U, V, W, tensor, n, m, p, rank)
        if err < tol:
            return U, V, W, True
        if abs(err - prev_err) < 1e-15:
            break
        prev_err = err
    return U, V, W, False

def _structured_init(n, m, p, rank, rng):
    """Generate structured initializations that may converge faster."""
    inits = []
    
    # Init 1: All-ones scaled
    U = np.ones((n * m, rank))
    V = np.ones((m * p, rank))
    W = np.ones((p * n, rank))
    inits.append((U, V, W))
    
    # Init 2: Random but with some structure
    for _ in range(3):
        U = rng.standard_normal((n * m, rank))
        V = rng.standard_normal((m * p, rank))
        W = rng.standard_normal((p * n, rank))
        # Normalize columns to prevent numerical issues
        for mat in [U, V, W]:
            norms = np.linalg.norm(mat, axis=0, keepdims=True)
            norms[norms < 1e-10] = 1.0
            mat /= norms
        inits.append((U, V, W))
    
    # Init 3: Sparse-like structured
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for r in range(rank):
        # Assign each factor to cover different slices
        idx_u = np.random.choice(n * m, size=max(1, n * m // rank + 1), replace=False)
        idx_v = np.random.choice(m * p, size=max(1, m * p // rank + 1), replace=False)
        idx_w = np.random.choice(p * n, size=max(1, p * n // rank + 1), replace=False)
        U[idx_u, r] = rng.standard_normal(len(idx_u))
        V[idx_v, r] = rng.standard_normal(len(idx_v))
        W[idx_w, r] = rng.standard_normal(len(idx_w))
    inits.append((U, V, W))
    
    return inits

def construct():
    deadline = 145  # seconds, safe margin before 150s limit
    start_time = time.time()
    
    n = n_dim
    m = m_dim
    p = p_dim
    
    # Create the standard tensor: tensor[i,j,k] = 1 for all i,j,k
    tensor = np.ones((n, m, p), dtype=np.float64)
    
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    U_fb = np.zeros((n * m, rank_schoolbook), dtype=np.float64)
    V_fb = np.zeros((m * p, rank_schoolbook), dtype=np.float64)
    W_fb = np.zeros((p * n, rank_schoolbook), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1
    best_rank = rank_schoolbook
    best_U = U_fb
    best_V = V_fb
    best_W = W_fb
    
    rng = np.random.default_rng(42)
    
    # Search ranks from low to high
    # For 3x4x6, known lower bounds suggest rank >= 14-15
    # Try from 14 upward
    ranks_to_try = list(range(14, rank_schoolbook + 1))
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 8:
            break
        
        # Generate initializations
        inits = _structured_init(n, m, p, target_rank, rng)
        
        found = False
        for U0, V0, W0 in inits:
            if time.time() > deadline - 4:
                break
            
            U, V, W = U0.copy(), V0.copy(), W0.copy()
            
            # Run ALS iterations
            success = False
            for it in range(80):
                if time.time() > deadline - 2:
                    break
                
                U, V, W = _als_step_fast(U, V, W, tensor, n, m, p, target_rank)
                err = _compute_error(U, V, W, tensor, n, m, p, target_rank)
                
                if err < 1e-9:
                    # Polish for higher precision
                    U, V, W, polished = _polish(U, V, W, tensor, n, m, p, target_rank)
                    if polished or _verify_fast(U, V, W, tensor):
                        best_rank = target_rank
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        success = True
                        break
            
            if success:
                found = True
                break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
