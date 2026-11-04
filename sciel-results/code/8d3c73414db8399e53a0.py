"""Find a rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication."""

import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import random
import sys

def _build_tensors(n, m, p):
    """Build U, V, W tensors for the matrix multiplication tensor."""
    U = np.zeros((n*m, n*m*p))
    V = np.zeros((m*p, n*m*p))
    W = np.zeros((p*n, n*m*p))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1
                V[j*p+k, idx] = 1
                W[k*n+i, idx] = 1
                idx += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is exact."""
    # Build the matrix multiplication tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Compute from decomposition
    # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    # Efficiently: for each c, outer product
    T_recon = np.zeros((n*m, m*p, p*n))
    for c in range(U.shape[1]):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        T_recon += np.einsum('i,j,k->ijk', uc, vc, wc)
    
    return np.allclose(T, T_recon, atol=tol)

def _try_rank_r(n, m, p, r, max_iter=100, seed=42):
    """Try to find a rank-r decomposition via iterative optimization."""
    rng = np.random.RandomState(seed)
    
    # Initialize
    U = rng.randn(n*m, r)
    V = rng.randn(m*p, r)
    W = rng.randn(p*n, r)
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Compute gradient
        # T_recon = sum_c U[:,c] outer V[:,c] outer W[:,c]
        # Loss = 0.5 * ||T - T_recon||_F^2
        
        # Efficient gradient computation
        # dT/du_ic = V_jc * W_kc for each j,k
        # grad_U[i,c] = sum_{j,k} (T_recon[i,j,k] - T[i,j,k]) * V[j,c] * W[k,c]
        
        T_recon = np.zeros((n*m, m*p, p*n))
        for c in range(r):
            T_recon += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
        
        diff = T_recon - T
        loss = 0.5 * np.sum(diff**2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-15:
            return best_U, best_V, best_W, True
        
        # Compute gradients
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for c in range(r):
            # grad_U[i,c] = sum_{j,k} diff[i,j,k] * V[j,c] * W[k,c]
            # grad_V[j,c] = sum_{i,k} diff[i,j,k] * U[i,c] * W[k,c]
            # grad_W[k,c] = sum_{i,j} diff[i,j,k] * U[i,c] * V[j,c]
            
            # Use einsum for efficiency
            grad_U[:, c] = np.einsum('ijk,jc,kc->ic', diff, V[:, c], W[:, c])
            grad_V[:, c] = np.einsum('ijk,ic,kc->jc', diff, U[:, c], W[:, c])
            grad_W[:, c] = np.einsum('ijk,ic,jc->kc', diff, U[:, c], V[:, c])
        
        # Simple gradient descent with adaptive step
        lr = 0.01
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
    
    # Final check
    if best_loss < 1e-15:
        return best_U, best_V, best_W, True
    return None, None, None, False

def _try_integer_polish(U, V, W, n, m, p):
    """Try to find integer/half-integer solutions near the found solution."""
    r = U.shape[1]
    
    # Scale to see if we can get integers
    # For now, just return as is since the task allows half-integers
    # We need to round to nearest 0.5 and verify
    
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    if _verify(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-10):
        return U_rounded, V_rounded, W_rounded, True
    
    return None, None, None, False

def construct():
    deadline = time.time() + 120  # 120 second deadline
    n, m, p = DIMENSIONS
    full_rank = n * m * p  # 50
    
    # First, return schoolbook as fallback
    U_school = np.zeros((n*m, full_rank))
    V_school = np.zeros((m*p, full_rank))
    W_school = np.zeros((p*n, full_rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, idx] = 1
                V_school[j*p+k, idx] = 1
                W_school[k*n+i, idx] = 1
                idx += 1
    
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = full_rank
    
    # Try to find lower rank decompositions
    # Known bounds: R(2,5,5) is between some values
    # Let's try from rank 40 down to 30
    
    for r in range(45, 29, -1):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            
            U, V, W, success = _try_rank_r(n, m, p, r, max_iter=50, seed=seed + r*100)
            
            if success:
                # Try to polish to integer/half-integer
                U2, V2, W2, success2 = _try_integer_polish(U, V, W, n, m, p)
                if success2:
                    best_rank = r
                    best_result = {
                        "U": U2.tolist(),
                        "V": V2.tolist(),
                        "W": W2.tolist()
                    }
                    break
                else:
                    # Use the floating point solution if it's exact enough
                    # But the task requires integer/half-integer entries
                    # So we can only use it if we can convert
                    pass
    
    return best_result
# EVOLVE-BLOCK-END
