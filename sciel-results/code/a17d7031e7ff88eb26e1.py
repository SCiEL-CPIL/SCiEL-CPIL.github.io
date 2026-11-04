"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x6 matrix.
    Direct optimization on the full tensor with structured initialization.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_fb = np.zeros((n*m, rank_schoolbook), dtype=np.float64)
    V_fb = np.zeros((m*p, rank_schoolbook), dtype=np.float64)
    W_fb = np.zeros((p*n, rank_schoolbook), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1.0
                V_fb[j*p+k, col] = 1.0
                W_fb[k*n+i, col] = 1.0
                col += 1
    
    def validate(U, V, W, T):
        """Check if U, V, W exactly decompose T."""
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        if not np.allclose(T_rec, T, atol=1e-9):
            return False
        for mat in [U, V, W]:
            if not np.all(np.abs(mat * 2 - np.round(mat * 2)) < 1e-9):
                return False
        return True
    
    def to_int_or_half(arr):
        """Round to nearest integer or half-integer."""
        arr = np.asarray(arr, dtype=np.float64)
        return np.round(arr * 2) / 2.0
    
    def compute_err(U, V, W, T):
        T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        err = T_rec - T
        return err, np.sum(err**2)
    
    def optimize(U_init, V_init, W_init, T, max_iters=5000, lr_init=0.01):
        """Adam optimizer for tensor decomposition."""
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        # Adam state
        mU, vU = np.zeros_like(U), np.zeros_like(U)
        mV, vV = np.zeros_like(V), np.zeros_like(V)
        mW, vW = np.zeros_like(W), np.zeros_like(W)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            err, err_sq = compute_err(U, V, W, T)
            
            if err_sq < best_err:
                best_err = err_sq
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err_sq < 1e-12:
                break
            
            # Gradients
            gU = 2 * np.einsum('abc,br,cr->ar', err, V, W, optimize=True)
            gV = 2 * np.einsum('abc,ar,cr->br', err, U, W, optimize=True)
            gW = 2 * np.einsum('abc,ar,br->cr', err, U, V, optimize=True)
            
            # Adam update
            t += 1
            lr = lr_init * np.sqrt(1 - beta2**t) / (1 - beta1**t)
            
            mU = beta1 * mU + (1 - beta1) * gU
            vU = beta2 * vU + (1 - beta2) * gU**2
            mU_hat = mU / (1 - beta1**t)
            vU_hat = vU / (1 - beta2**t)
            U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
            
            mV = beta1 * mV + (1 - beta1) * gV
            vV = beta2 * vV + (1 - beta2) * gV**2
            mV_hat = mV / (1 - beta1**t)
            vV_hat = vV / (1 - beta2**t)
            V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
            
            mW = beta1 * mW + (1 - beta1) * gW
            vW = beta2 * vW + (1 - beta2) * gW**2
            mW_hat = mW / (1 - beta1**t)
            vW_hat = vW / (1 - beta2**t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        
        return best_U, best_V, best_W, best_err
    
    # Try ranks from 40 to 54 (schoolbook)
    # The two-block approach with rank-23 for 3x3x3 gives 46
    # Try to beat that
    best_result = None
    best_rank = rank_schoolbook + 1
    
    for target_rank in range(40, min(55, rank_schoolbook + 1)):
        if time.time() > deadline - 30:
            break
        
        if target_rank >= best_rank:
            break
        
        found = False
        n_restarts = max(1, int((deadline - 30 - time.time()) / 3.0))
        n_restarts = min(n_restarts, 8)
        
        for seed in range(n_restarts):
            if time.time() > deadline - 15:
                break
            np.random.seed(seed * 7919 + target_rank * 131)
            
            # Structured initialization: start from schoolbook and add random perturbation
            # or use random init
            if seed == 0:
                # Structured: use schoolbook-like init with rank reduction
                U_init = np.random.randn(n*m, target_rank) * 0.5
                V_init = np.random.randn(m*p, target_rank) * 0.5
                W_init = np.random.randn(p*n, target_rank) * 0.5
            else:
                # Pure random
                U_init = np.random.randn(n*m, target_rank) * 0.3
                V_init = np.random.randn(m*p, target_rank) * 0.3
                W_init = np.random.randn(p*n, target_rank) * 0.3
            
            U_opt, V_opt, W_opt, err_sq = optimize(
                U_init, V_init, W_init, T,
                max_iters=3000, lr_init=0.005
            )
            
            if err_sq < 1e-10:
                # Try rounding to integer/half-integer
                U_r = to_int_or_half(U_opt)
                V_r = to_int_or_half(V_opt)
                W_r = to_int_or_half(W_opt)
                
                T_rec = np.einsum('ar,br,cr->abc', U_r, V_r, W_r, optimize=True)
                if np.allclose(T_rec, T, atol=1e-9):
                    if validate(U_r, V_r, W_r, T):
                        best_result = (U_r, V_r, W_r, target_rank)
                        best_rank = target_rank
                        found = True
                        break
        
        if found:
            break
    
    if best_result is not None:
        U, V, W, r = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
