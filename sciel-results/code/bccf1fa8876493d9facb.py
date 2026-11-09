"""Rewritten program for 3x5x7 matrix multiplication tensor rank minimization."""
import numpy as np
import time

DIMENSIONS = [3, 5, 7]
N, M, P = DIMENSIONS
NM, MP, PN = N*M, M*P, P*N
TOTAL = N*M*P

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k]) * W[k*n+i]  # This is wrong, need element-wise
                # Actually: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-8):
    """Faster vectorized verification."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            u_row = U[i*m+j]  # (rank,)
            for k in range(p):
                val = np.sum(u_row * V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _round_to_halfint(arr):
    """Round array entries to nearest half-integer."""
    return np.round(arr * 2.0) / 2.0

def _is_halfint(arr, tol=1e-8):
    """Check if all entries are within tol of half-integers."""
    doubled = arr * 2.0
    nearest = np.round(doubled)
    return np.max(np.abs(doubled - nearest)) < tol

def _structured_init(r, n, m, p, seed=0):
    """Create a structured initial guess for rank-r decomposition.
    
    Strategy: Use the schoolbook decomposition as a base and try to merge terms.
    For rank r < n*m*p, we need to find a way to share factors.
    
    A common approach: initialize with random values scaled appropriately.
    """
    rng = np.random.RandomState(seed)
    
    # Scale: each term contributes roughly 1/r to the sum, so values ~ 1/sqrt(r)
    scale = 1.0 / np.sqrt(r)
    U = rng.randn(n*m, r) * scale
    V = rng.randn(m*p, r) * scale
    W = rng.randn(p*n, r) * scale
    
    return U, V, W

def _structured_init_from_schoolbook(r, n, m, p, seed=0):
    """Initialize from schoolbook by merging random groups of terms.
    
    If r < n*m*p, we can't use schoolbook directly. Instead, use it as
    a target and start from a lower-rank random init.
    """
    rng = np.random.RandomState(seed)
    scale = 1.0 / np.sqrt(max(r, 1))
    U = rng.randn(n*m, r) * scale
    V = rng.randn(m*p, r) * scale
    W = rng.randn(p*n, r) * scale
    return U, V, W

def _loss_and_grads(U, V, W, n, m, p):
    """Compute loss and gradients for the decomposition.
    
    Loss = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
    """
    rank = U.shape[1]
    loss = 0.0
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            u_row = U[i*m+j]  # (rank,)
            for k in range(p):
                v_row = V[j*p+k]  # (rank,)
                w_row = W[k*n+i]  # (rank,)
                # val = sum_c u_row[c] * v_row[c] * w_row[c]
                val = np.sum(u_row * v_row * w_row)
                err = val - 1.0
                loss += err * err
                
                # Gradient w.r.t. U[i*m+j, c] = err * V[j*p+k, c] * W[k*n+i, c]
                grad_U[i*m+j] += err * v_row * w_row
                # Gradient w.r.t. V[j*p+k, c] = err * U[i*m+j, c] * W[k*n+i, c]
                grad_V[j*p+k] += err * u_row * w_row
                # Gradient w.r.t. W[k*n+i, c] = err * U[i*m+j, c] * V[j*p+k, c]
                grad_W[k*n+i] += err * u_row * v_row
    
    return loss, grad_U, grad_V, grad_W

def _loss_and_grads_fast(U, V, W, n, m, p):
    """Compute loss and gradients using vectorized operations where possible."""
    rank = U.shape[1]
    loss = 0.0
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    # Precompute for each (j, k): V[j*p+k]
    # For each (i, k): W[k*n+i]
    
    for i in range(n):
        for k in range(p):
            w_row = W[k*n+i]  # (rank,)
            for j in range(m):
                u_row = U[i*m+j]  # (rank,)
                v_row = V[j*p+k]  # (rank,)
                val = np.sum(u_row * v_row * w_row)
                err = val - 1.0
                loss += err * err
                
                uvw = v_row * w_row  # (rank,)
                uwr = u_row * w_row
                uvv = u_row * v_row
                
                grad_U[i*m+j] += err * uvw
                grad_V[j*p+k] += err * uwr
                grad_W[k*n+i] += err * uvv
    
    return loss, grad_U, grad_V, grad_W

def _optimize(U, V, W, n, m, p, max_iters=5000, lr_init=0.05, tol=1e-10, time_limit=None):
    """Optimize the decomposition using gradient descent with adaptive learning rate."""
    rank = U.shape[1]
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = lr_init
    prev_loss = float('inf')
    
    for it in range(max_iters):
        if time_limit is not None and time.time() > time_limit:
            break
        
        loss, gU, gV, gW = _loss_and_grads_fast(U, V, W, n, m, p)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < tol:
            break
        
        # Adaptive learning rate
        if loss > prev_loss:
            lr *= 0.5
        elif it > 0 and loss < prev_loss * 0.9:
            lr = min(lr * 1.1, 0.5)
        prev_loss = loss
        
        # Gradient step
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
        
        # Periodic check for half-integer convergence
        if it % 500 == 0 and best_loss < 1e-4:
            U_h = _round_to_halfint(best_U)
            V_h = _round_to_halfint(best_V)
            W_h = _round_to_halfint(best_W)
            if _verify_fast(U_h, V_h, W_h, n, m, p, tol=1e-6):
                return U_h, V_h, W_h, best_loss
    
    # Final check
    if best_loss < 1e-4:
        U_h = _round_to_halfint(best_U)
        V_h = _round_to_halfint(best_V)
        W_h = _round_to_halfint(best_W)
        if _verify_fast(U_h, V_h, W_h, n, m, p, tol=1e-6):
            return U_h, V_h, W_h, best_loss
    
    return best_U, best_V, best_W, best_loss

def _try_rank(r, n, m, p, max_seeds=10, max_iters=3000, time_limit=None):
    """Try to find a rank-r decomposition. Returns (U, V, W) or None."""
    if time_limit is not None and time.time() > time_limit:
        return None
    
    for seed in range(max_seeds):
        if time_limit is not None and time.time() > time_limit:
            return None
        
        U, V, W = _structured_init(r, n, m, p, seed=seed)
        U, V, W, loss = _optimize(U, V, W, n, m, p, max_iters=max_iters, time_limit=time_limit)
        
        if loss < 1e-8:
            # Try rounding
            U_h = _round_to_halfint(U)
            V_h = _round_to_halfint(V)
            W_h = _round_to_halfint(W)
            if _verify_fast(U_h, V_h, W_h, n, m, p, tol=1e-6):
                return U_h, V_h, W_h
        
        # Check if already close to half-integers
        if _is_halfint(U) and _is_halfint(V) and _is_halfint(W):
            if _verify_fast(U, V, W, n, m, p, tol=1e-6):
                return U, V, W
    
    return None

def _try_rank_with_polish(r, n, m, p, max_seeds=5, max_iters=2000, time_limit=None):
    """Try rank-r with a polish pass after initial optimization."""
    if time_limit is not None and time.time() > time_limit:
        return None
    
    for seed in range(max_seeds):
        if time_limit is not None and time.time() > time_limit:
            return None
        
        U, V, W = _structured_init(r, n, m, p, seed=seed)
        U, V, W, loss = _optimize(U, V, W, n, m, p, max_iters=max_iters, time_limit=time_limit)
        
        if loss < 1e-4:
            # Polish with smaller learning rate
            U, V, W, loss2 = _optimize(U, V, W, n, m, p, max_iters=2000, lr_init=0.005, tol=1e-12, time_limit=time_limit)
            
            if loss2 < 1e-8:
                U_h = _round_to_halfint(U)
                V_h = _round_to_halfint(V)
                W_h = _round_to_halfint(W)
                if _verify_fast(U_h, V_h, W_h, n, m, p, tol=1e-6):
                    return U_h, V_h, W_h
    
    return None

def construct():
    n, m, p = N, M, P
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Schoolbook decomposition (rank = n*m*p = 105)
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    
    # Known lower bounds:
    # rank >= max(n*m, m*p, n*p) = max(15, 35, 21) = 35
    # For small matrices, better bounds exist but let's try from a reasonable range.
    
    # Strategy: try ranks from a moderate upper bound down to the lower bound.
    # Start with ranks that are more likely to be achievable.
    
    # Try various ranks in a strategic order
    # First try some intermediate ranks, then go lower
    candidate_ranks = []
    
    # Start from a reasonable upper bound and go down
    # For 3x5x7, let's try from 70 down to 35
    for r in range(70, 34, -1):
        candidate_ranks.append(r)
    
    # Also try some specific values that might work well
    # Add some in between if not already there
    for r in [60, 55, 50, 45, 40]:
        if r not in candidate_ranks:
            candidate_ranks.append(r)
    
    # Sort in descending order (try higher ranks first as they're easier)
    candidate_ranks = sorted(set(candidate_ranks), reverse=True)
    
    for r in candidate_ranks:
        if time.time() > deadline - 15:
            break
        
        if r >= best_rank:
            continue
        
        # Try this rank
        result = _try_rank(r, n, m, p, max_seeds=8, max_iters=2000, time_limit=deadline - 10)
        
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
        
        # If we found a good rank, try to go lower
        if r <= 45 and time.time() > deadline - 30:
            break
    
    return best_result
