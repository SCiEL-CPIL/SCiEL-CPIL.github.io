"""Rewritten program for 3x3 x 3x8 matrix multiplication tensor decomposition."""
import numpy as np
from fractions import Fraction
import time
import itertools

DIMENSIONS = [3, 3, 8]

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
    
    # Build the matrix multiplication tensor
    # T[a, b, c] where a = i*m+j (9 values), b = j*p+k (24 values), c = k*n+i (24 values)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def schoolbook():
        """Return the standard schoolbook decomposition with rank n*m*p = 72."""
        r = n * m * p
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that U ⊗ V ⊗ W reconstructs T exactly."""
        r = U.shape[1]
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        max_err = np.max(np.abs(T_recon - T))
        return max_err < tol, max_err
    
    def try_block_decomposition():
        """
        Try a structured decomposition that exploits the block structure.
        
        Key insight: For 3x3 * 3x8, we can view this as 8 independent 3x3 * 3x1 
        multiplications, but more cleverly, we can group columns of B.
        
        Actually, a better approach: The tensor T for (n,m,p) has a specific structure.
        For fixed j, the sub-tensor T[:, j*p+k, :] for all k is related to the 
        3x3x3 tensor.
        
        Let's try: Use Strassen-like ideas for the 3x3 part, then extend to 8 columns.
        
        For 3x3 * 3x1, the rank is known to be 16 (not 9). But for 3x3 * 3x8,
        we can potentially do better than 72 by sharing structure across columns.
        
        Approach: Group the 8 columns into 2 groups of 4. For each group, we compute
        a 3x3 * 3x4 product. Then combine.
        
        Actually, let's think differently. The tensor T[i,j,k] = A[i,j] * B[j,k].
        For fixed j, we have a rank-1 tensor in (i,k) space: outer(A[:,j], B[j,:]).
        
        So T = sum_j outer(A[:,j], B[j,:]) ⊗ e_j (where e_j is the j-th basis vector in m-dim).
        This is the schoolbook with rank 9*8 = 72.
        
        To do better, we need to find a low-rank decomposition of this specific tensor.
        
        Let's try a numerical approach with better initialization.
        """
        pass
    
    def optimize_decomposition(r, seed=42, max_iter=1000, lr=0.01, tol=1e-8):
        """
        Gradient-based optimization for rank-r decomposition.
        Uses Adam-like adaptive learning rate for better convergence.
        """
        rng = np.random.RandomState(seed)
        
        # Initialize with small random values
        U = rng.randn(n*m, r) * 0.1
        V = rng.randn(m*p, r) * 0.1
        W = rng.randn(p*n, r) * 0.1
        
        # Adam optimizer state
        mU = np.zeros_like(U)
        vU = np.zeros_like(U)
        mV = np.zeros_like(V)
        vV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vW = np.zeros_like(W)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            # Compute T_recon
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = T_recon - T
            
            # Gradients
            dU = np.einsum('abc,br,cr->ar', err, V, W)
            dV = np.einsum('abc,ar,cr->br', err, U, W)
            dW = np.einsum('abc,ar,br->cr', err, U, V)
            
            t += 1
            
            # Adam update for U
            mU = beta1 * mU + (1 - beta1) * dU
            vU = beta2 * vU + (1 - beta2) * dU**2
            mU_hat = mU / (1 - beta1**t)
            vU_hat = vU / (1 - beta2**t)
            U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
            
            # Adam update for V
            mV = beta1 * mV + (1 - beta1) * dV
            vV = beta2 * vV + (1 - beta2) * dV**2
            mV_hat = mV / (1 - beta1**t)
            vV_hat = vV / (1 - beta2**t)
            V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
            
            # Adam update for W
            mW = beta1 * mW + (1 - beta1) * dW
            vW = beta2 * vW + (1 - beta2) * dW**2
            mW_hat = mW / (1 - beta1**t)
            vW_hat = vW / (1 - beta2**t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            
            cur_err = np.max(np.abs(T_recon - T))
            if cur_err < best_err:
                best_err = cur_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < tol:
                break
        
        return best_U, best_V, best_W, best_err
    
    def try_random_rank(r, seed=0, iterations=50, max_iter_per_iter=500, lr=0.05):
        """Try to find a rank-r decomposition using random initialization + optimization."""
        for it in range(iterations):
            U, V, W, err = optimize_decomposition(r, seed=seed + it, max_iter=max_iter_per_iter, lr=lr, tol=1e-7)
            if err < 1e-6:
                return U, V, W, err
        return None
    
    # Try different ranks from low to high
    best_result = None
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Try ranks from 24 up (since 72 is schoolbook, try to do better)
    # Start with more promising ranks
    ranks_to_try = [24, 27, 30, 32, 36, 40, 48, 54, 60, 64, 72]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        
        # For smaller ranks, try more iterations
        max_iter = 300 if r <= 36 else 200
        iterations = 30 if r <= 36 else 20
        
        result = try_random_rank(r, seed=0, iterations=iterations, max_iter_per_iter=max_iter, lr=0.05)
        
        if result is not None:
            U, V, W, err = result
            # Verify
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            max_err = np.max(np.abs(T_recon - T))
            if max_err < 1e-6:
                best_result = (r, U, V, W, max_err)
                break
    
    if best_result is None:
        # Fall back to schoolbook
        return schoolbook()
    else:
        r, U, V, W, err = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
