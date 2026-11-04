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

def _error_and_grad(U, V, W, T, n, m, p):
    """Compute error and gradients for tensor decomposition.
    T_reconstructed = sum_r U[:,r] * V[:,r] * W[:,r]
    Error = sum of (T_recon - T_target)^2 over all entries
    """
    nm, mp, pn = n*m, m*p, p*n
    r = U.shape[1]
    
    # Build reconstructed tensor
    T_rec = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Error
    err = np.sum((T_rec - T) ** 2)
    
    # Gradients: d(err)/dU[a,r] = 2 * sum_{b,c} (T_rec[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
    # = 2 * (T_rec - T) contracted with V and W
    diff = T_rec - T  # shape (nm, mp, pn)
    
    grad_U = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
    grad_V = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
    grad_W = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
    
    return err, grad_U, grad_V, grad_W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook baseline
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = 0.0
    
    # Precompute target tensor
    T = _build_target(n, m, p)
    
    # Try to find lower rank decompositions using gradient-based optimization
    # Known lower bounds: For 3x3x3, rank is 23 (Strassen-like). For 3x3x6, we can try various ranks
    for r in range(23, 54):
        if time.time() > deadline - 15:
            break
        
        # Multiple random restarts
        max_trials = 3 if r > 40 else 5
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + r * 100 + trial)
            U_r = np.random.randn(n * m, r) * 0.5
            V_r = np.random.randn(m * p, r) * 0.5
            W_r = np.random.randn(p * n, r) * 0.5
            
            # Gradient descent with adaptive learning rate
            lr = 0.01
            min_lr = 1e-6
            patience = 0
            best_local_err = float('inf')
            
            for step in range(200):
                if time.time() > deadline - 8:
                    break
                
                err, gU, gV, gW = _error_and_grad(U_r, V_r, W_r, T, n, m, p)
                
                if err < 1e-10:
                    if err < best_err:
                        best_err = err
                        best_rank = r
                        best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    break
                
                if err < best_local_err:
                    best_local_err = err
                    patience = 0
                    if err < best_err:
                        best_err = err
                        best_rank = r
                        best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                else:
                    patience += 1
                    if patience > 30:
                        break
                
                # Adaptive learning rate
                norm_g = np.sqrt(np.sum(gU**2) + np.sum(gV**2) + np.sum(gW**2))
                if norm_g > 1e-8:
                    step_size = lr / (1.0 + norm_g * 0.1)
                else:
                    step_size = lr
                
                U_r -= step_size * gU
                V_r -= step_size * gW
                W_r -= step_size * gW
                
                # Reduce learning rate slowly
                lr = max(min_lr, lr * 0.99)
    
    if best_rank < rank:
        return {'U': best_U.tolist(), 'V': best_V.tolist(), 'W': best_W.tolist()}
    else:
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
