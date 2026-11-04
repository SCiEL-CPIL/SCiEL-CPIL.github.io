"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time
import itertools

DIMENSIONS = [3, 3, 8]
DEADLINE = 150.0  # wall-clock seconds

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix 
    by a 3x8 matrix.
    
    The matrix multiplication tensor T has entries T[(i,j),(j,k),(i,k)] = 1
    where i in {0,1,2}, j in {0,1,2}, k in {0,...,7}.
    
    We seek rank-r decomposition:
    T = sum_{c=0}^{r-1} u_c ⊗ v_c ⊗ w_c
    
    where u_c has entries indexed by (i,j) [9 entries],
          v_c has entries indexed by (j,k) [24 entries],  
          w_c has entries indexed by (i,k) [24 entries].
    
    The output factor follows W[k*n+i, r] convention.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    # Build the matrix multiplication tensor
    # T[a, b, c] where a = i*m+j (9 values), b = j*p+k (24 values), c = k*n+i (24 values)
    # T[a, b, c] = 1 if a corresponds to (i,j), b to (j,k), c to (i,k) with consistent i,j,k
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that U ⊗ V ⊗ W reconstructs T exactly."""
        r = U.shape[1]
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        max_err = np.max(np.abs(T_recon - T))
        return max_err < tol, max_err
    
    def schoolbook():
        """Return the schoolbook decomposition (rank 72)."""
        r = n * m * p
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W
    
    def optimize_decomposition(r, seed=42, max_iter=300, lr=0.02, tol=1e-8):
        """Gradient-based optimization for rank-r decomposition with adaptive step size."""
        rng = np.random.RandomState(seed)
        U = rng.randn(n*m, r) * 0.1
        V = rng.randn(m*p, r) * 0.1
        W = rng.randn(p*n, r) * 0.1
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        lr_current = lr
        
        for it in range(max_iter):
            if time_left() < 10.0:
                break
            
            # Compute T_recon and error
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = T_recon - T
            
            # Compute gradients
            dU = np.einsum('abc,br,cr->ar', err, V, W)
            dV = np.einsum('abc,ar,cr->br', err, U, W)
            dW = np.einsum('abc,ar,br->cr', err, U, V)
            
            # Adaptive learning rate
            err_norm = np.linalg.norm(err)
            if err_norm > 1e-4:
                lr_current = max(0.001, min(0.1, lr * 0.95))
            
            # Update
            U -= lr_current * dU
            V -= lr_current * dV
            W -= lr_current * dW
            
            cur_err = np.max(np.abs(T_recon - T))
            if cur_err < best_err:
                best_err = cur_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < tol:
                break
        
        return best_U, best_V, best_W, best_err
    
    def polish(U, V, W, max_iter=100, tol=1e-10):
        """Final high-precision polish pass with very small steps."""
        for it in range(max_iter):
            if time_left() < 5.0:
                break
            
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = T_recon - T
            max_err = np.max(np.abs(err))
            
            if max_err < tol:
                break
            
            # Very small gradient steps
            lr = 0.001
            dU = np.einsum('abc,br,cr->ar', err, V, W)
            dV = np.einsum('abc,ar,cr->br', err, U, W)
            dW = np.einsum('abc,ar,br->cr', err, U, V)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        max_err = np.max(np.abs(T_recon - T))
        return U, V, W, max_err
    
    # Try different ranks from low to high
    best_result = None
    
    # Try ranks: start from plausible lower bounds
    # Theoretical lower bound for 3x3x3 is at least 19 (from literature)
    # For 3x3 x 3x8, we can't do better than roughly (3x3x3 rank) * 8 / 3 ≈ 51
    # Let's try a range of ranks
    ranks_to_try = [24, 27, 30, 32, 36, 40, 45, 48, 54, 60, 64, 72]
    
    for r in ranks_to_try:
        if time_left() < 20.0:
            break
        
        found = False
        # Try multiple seeds for this rank
        for seed in range(30):
            if time_left() < 15.0:
                break
            
            U, V, W, err = optimize_decomposition(r, seed=seed, max_iter=200, lr=0.03, tol=1e-7)
            
            if err < 1e-6:
                # Polish with tighter tolerance
                U, V, W, err = polish(U, V, W, max_iter=100, tol=1e-10)
                
                if err < 1e-8:
                    best_result = (r, U, V, W, err)
                    found = True
                    break
        
        if found:
            break
    
    if best_result is None:
        # Fall back to schoolbook
        U, V, W = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        r, U, V, W, err = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
