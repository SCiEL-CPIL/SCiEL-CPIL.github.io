import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _build_target(n, m, p):
    """Build the target tensor T where T[i*m+j, j*p+k, k*n+i] = 1"""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _reconstruct(U, V, W):
    """Reconstruct tensor from factor matrices."""
    return np.einsum('ar,br,cr->abc', U, V, W)

def _error_and_grad(U, V, W, T):
    """Compute error and gradients for tensor decomposition."""
    T_rec = _reconstruct(U, V, W)
    diff = T_rec - T
    err = np.sum(diff ** 2)
    
    grad_U = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
    grad_V = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
    grad_W = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
    
    return err, grad_U, grad_V, grad_W

def _try_rank(U, V, W, T, n, m, p, max_steps=300, lr_init=0.01, tol=1e-12):
    """Try to optimize a rank-r decomposition to zero error."""
    r = U.shape[1]
    lr = lr_init
    min_lr = 1e-8
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    patience = 0
    
    for step in range(max_steps):
        err, gU, gV, gW = _error_and_grad(U, V, W, T)
        
        if err < tol:
            return U, V, W, err
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            patience = 0
        else:
            patience += 1
            if patience > 50:
                break
        
        norm_g = np.sqrt(np.sum(gU**2) + np.sum(gV**2) + np.sum(gW**2))
        if norm_g > 1e-10:
            step_size = lr / (1.0 + norm_g * 0.05)
        else:
            step_size = lr
        
        U -= step_size * gU
        V -= step_size * gV
        W -= step_size * gW
        
        lr = max(min_lr, lr * 0.995)
    
    return best_U, best_V, best_W, best_err

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook baseline: rank = n*m*p = 54
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = 0.0
    
    # Precompute target tensor
    T = _build_target(n, m, p)
    
    # Try various ranks below schoolbook
    # Known: for 3x3x3, rank is 23. For 3x3x6, try from 27 upward
    ranks_to_try = list(range(27, 54))
    
    for r in ranks_to_try:
        if time.time() > deadline - 20:
            break
        
        # Multiple random restarts with structured initialization
        max_trials = 8 if r <= 40 else 5
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(12345 + r * 7919 + trial * 104729)
            
            # Structured initialization: use the schoolbook pattern as a guide
            # Initialize with small random perturbations around a structured base
            U_r = np.random.randn(n * m, r) * 0.3
            V_r = np.random.randn(m * p, r) * 0.3
            W_r = np.random.randn(p * n, r) * 0.3
            
            # Also try a "partially structured" init: for the first n*m*p entries,
            # seed some rank-1 terms
            if r >= n * m * p:
                pass  # not applicable since r < 54
            else:
                # Add some deterministic structure
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx = i * m * p + j * p + k
                            if idx < r:
                                U_r[i * m + j, idx] += 1.0
                                V_r[j * p + k, idx] += 1.0
                                W_r[k * n + i, idx] += 1.0
            
            U_opt, V_opt, W_opt, err_opt = _try_rank(U_r, V_r, W_r, T, n, m, p)
            
            if err_opt < best_err:
                best_err = err_opt
                if err_opt < 1e-10:
                    best_rank = r
                    best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
                    break  # Found exact solution for this rank, move to next rank
                else:
                    best_rank = r
                    best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
        
        if best_rank < rank_school and best_err < 1e-10:
            # Found an exact decomposition, try even lower ranks
            continue
        elif best_err < 1e-6:
            # Close to exact, keep going but be careful
            continue
        else:
            # Not making progress, might want to stop early
            if r > 40:
                break
    
    if best_rank < rank_school and best_err < 1e-10:
        return {'U': best_U.tolist(), 'V': best_V.tolist(), 'W': best_W.tolist()}
    else:
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
