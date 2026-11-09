"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction

import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _tensor_from_decomp(U, V, W):
    """Compute tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] using einsum."""
    return np.einsum('ar,br,cr->abc', U, V, W)


def _loss_and_grad(U, V, W, T):
    """
    Compute smooth loss L = ||T - sum_r u_r v_r w_r||^2 and its gradient.
    Uses efficient einsum-based computation.
    """
    # T_approx[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    T_approx = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    diff = T_approx - T
    loss = np.sum(diff * diff)
    
    # Gradient: dL/dU[a,r] = 2 * sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
    #            = 2 * (diff * V * W) contracted over b,c
    # More precisely: dL/dU[a,r] = 2 * sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
    # We can compute this as: for each r, grad_U[:,r] = 2 * einsum('abc,br,cr->a', diff, V, W)
    
    # Efficient: grad_U[a,r] = 2 * sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
    # = 2 * (diff @ (V.T * W.T)) but that's not quite right due to dimensions.
    # Let's do it step by step:
    # temp1[a,b] = sum_c diff[a,b,c] * W[c,r] ... no, r is fixed.
    
    # Actually: grad_U[a,r] = 2 * sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
    # For fixed r: sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
    # = sum_b (sum_c diff[a,b,c] * W[c,r]) * V[b,r]
    # = sum_b (diff[:, b, :] @ W[:, r]) * V[b, r]
    
    # Let's compute using einsum:
    # grad_U[a,r] = 2 * einsum('abc,br,cr->ar', diff, V, W)
    # grad_V[b,r] = 2 * einsum('abc,ar,cr->br', diff, U, W)
    # grad_W[c,r] = 2 * einsum('abc,ar,br->cr', diff, U, V)
    
    grad_U = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
    grad_V = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
    grad_W = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
    
    return loss, grad_U, grad_V, grad_W


def _optimize_decomposition(n, m, p, r, T, max_iters, lr_init, deadline, seed=42):
    """
    Optimize a rank-r decomposition using gradient descent with adaptive learning rate.
    Returns (U, V, W, final_loss) or None if deadline exceeded.
    """
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, r) * 0.1
    V = rng.randn(m * p, r) * 0.1
    W = rng.randn(p * n, r) * 0.1
    
    lr = lr_init
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    patience = 50
    no_improve = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            return None
        
        loss, gU, gV, gW = _loss_and_grad(U, V, W, T)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        # Adaptive learning rate: reduce if no improvement
        if no_improve >= patience:
            lr *= 0.5
            no_improve = 0
            patience = max(20, patience // 2)
            if lr < 1e-8:
                break
        
        # Gradient step
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
        
        # Early stopping if loss is very small
        if loss < 1e-10:
            break
    
    return best_U, best_V, best_W, best_loss


def _snap_to_half_integers(U, V, W, tol=1e-6):
    """
    Try to snap values to nearest half-integer if close.
    Returns True if all values are within tol of a half-integer.
    """
    def snap(M):
        snapped = np.round(M * 2) / 2
        if np.allclose(M, snapped, atol=tol, rtol=0):
            return snapped
        return None
    
    U_s = snap(U)
    V_s = snap(V)
    W_s = snap(W)
    
    if U_s is not None and V_s is not None and W_s is not None:
        return True, U_s, V_s, W_s
    return False, U, V, W


def _verify_decomposition(U, V, W, T, tol=1e-6):
    """Check if the decomposition exactly reproduces T."""
    T_approx = _tensor_from_decomp(U, V, W)
    return np.allclose(T_approx, T, atol=tol, rtol=0)


def _polish_decomposition(n, m, p, r, U, V, W, T, deadline, max_iters=200):
    """
    High-precision polish: use smaller learning rate and more iterations.
    """
    if time.time() > deadline:
        return U, V, W
    
    lr = 1e-5
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        loss, gU, gV, gW = _loss_and_grad(U, V, W, T)
        
        if loss < 1e-12:
            break
        
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
    
    return U, V, W


def _try_find_smaller_decomposition(n, m, p, deadline):
    """
    Try to find a decomposition with rank smaller than n*m*p.
    Uses gradient-based optimization with multiple restarts.
    """
    start = time.time()
    best_rank = n * m * p
    best_decomp = _schoolbook(n, m, p)
    
    T = _build_target_tensor(n, m, p)
    
    # Known lower bounds: for matrix multiplication, rank >= max(n*m, m*p, p*n)
    # Actually the lower bound is more subtle. Let's use a reasonable range.
    # For 3x4*4x8, schoolbook is 96. Try ranks from 95 down to ~80.
    
    # First, try to verify if rank 95 works with a few random initializations
    # and gradient descent.
    
    ranks_to_try = list(range(best_rank - 1, best_rank - 16, -1))
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple random restarts
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            U, V, W, final_loss = _optimize_decomposition(
                n, m, p, r, T, 
                max_iters=300, 
                lr_init=0.01, 
                deadline=deadline,
                seed=42 + attempt
            )
            
            if U is None:
                continue
            
            # Polish
            U, V, W = _polish_decomposition(n, m, p, r, U, V, W, T, deadline)
            
            # Check if exact
            if _verify_decomposition(U, V, W, T, tol=1e-6):
                # Try to snap to half-integers
                snapped, U_s, V_s, W_s = _snap_to_half_integers(U, V, W)
                if snapped:
                    if _verify_decomposition(U_s, V_s, W_s, T, tol=1e-6):
                        best_rank = r
                        best_decomp = {
                            "U": U_s.tolist(),
                            "V": V_s.tolist(),
                            "W": W_s.tolist()
                        }
                        found = True
                        break
                else:
                    # Use the float values converted to list
                    best_rank = r
                    best_decomp = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    found = True
                    break
        
        if found:
            # Continue trying for even smaller rank
            continue
    
    return best_rank, best_decomp


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Leave 40s buffer for the 150s limit
    
    # Try to find a smaller decomposition
    rank, decomp = _try_find_smaller_decomposition(n, m, p, deadline)
    
    return decomp
# EVOLVE-BLOCK-END
