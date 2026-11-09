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
    """Build the target matrix multiplication tensor."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _tensor_from_decomp(U, V, W):
    """Compute tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]."""
    # Use einsum for efficiency
    return np.einsum('ar,br,cr->abc', U, V, W)


def _loss_and_grad(U, V, W, T):
    """Compute loss ||T - tensor(U,V,W)||^2 and gradients."""
    T_approx = _tensor_from_decomp(U, V, W)
    diff = T_approx - T
    loss = np.sum(diff * diff)
    
    # Gradients: d/dU[a,r] loss = 2 * sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
    # = 2 * einsum('abc,br,cr->ar', diff, V, W)
    gU = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
    gV = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
    gW = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
    
    return loss, gU, gV, gW


def _optimize_decomposition(U, V, W, T, max_iters=200, lr_init=0.01, deadline=None):
    """Optimize U,V,W to minimize ||T - tensor(U,V,W)||^2 using Adam-like optimizer."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    # Adam optimizer state
    mU = np.zeros_like(U)
    vU = np.zeros_like(U)
    mV = np.zeros_like(V)
    vV = np.zeros_like(V)
    mW = np.zeros_like(W)
    vW = np.zeros_like(W)
    
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    lr = lr_init
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        loss, gU, gV, gW = _loss_and_grad(U, V, W, T)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-10:
            break
        
        # Adam update
        mU = beta1 * mU + (1 - beta1) * gU
        vU = beta2 * vU + (1 - beta2) * gU**2
        mU_hat = mU / (1 - beta1**(it + 1))
        vU_hat = vU / (1 - beta2**(it + 1))
        U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
        
        mV = beta1 * mV + (1 - beta1) * gV
        vV = beta2 * vV + (1 - beta2) * gV**2
        mV_hat = mV / (1 - beta1**(it + 1))
        vV_hat = vV / (1 - beta2**(it + 1))
        V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
        
        mW = beta1 * mW + (1 - beta1) * gW
        vW = beta2 * vW + (1 - beta2) * gW**2
        mW_hat = mW / (1 - beta1**(it + 1))
        vW_hat = vW / (1 - beta2**(it + 1))
        W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        
        # Decay learning rate
        if it > 50:
            lr *= 0.95
    
    return best_U, best_V, best_W, best_loss


def _snap_to_half_integers(U, V, W, tol=1e-6):
    """Snap values to nearest half-integer if close enough."""
    U_snapped = np.round(U * 2) / 2.0
    V_snapped = np.round(V * 2) / 2.0
    W_snapped = np.round(W * 2) / 2.0
    
    # Check if snapping is valid (values are close to half-integers)
    max_diff_U = np.max(np.abs(U - U_snapped))
    max_diff_V = np.max(np.abs(V - V_snapped))
    max_diff_W = np.max(np.abs(W - W_snapped))
    
    if max_diff_U < tol and max_diff_V < tol and max_diff_W < tol:
        return U_snapped, V_snapped, W_snapped
    else:
        return U, V, W


def _verify_decomposition(U, V, W, T, tol=1e-6):
    """Check if decomposition exactly matches target tensor."""
    T_approx = _tensor_from_decomp(U, V, W)
    return np.max(np.abs(T_approx - T)) < tol


def _polish_decomposition(U, V, W, T, deadline):
    """High-precision polish: use smaller learning rate and more iterations."""
    U, V, W, loss = _optimize_decomposition(U, V, W, T, max_iters=100, lr_init=0.001, deadline=deadline)
    U, V, W = _snap_to_half_integers(U, V, W, tol=1e-8)
    return U, V, W


def _try_find_smaller_decomposition(n, m, p, deadline):
    """
    Try to find a decomposition with rank smaller than n*m*p.
    Uses gradient descent with random restarts.
    """
    start = time.time()
    best_rank = n * m * p
    best_decomp = _schoolbook(n, m, p)
    
    # Build the target tensor
    T = _build_target_tensor(n, m, p)
    
    # Try ranks from best_rank-1 down to a lower bound
    # Lower bound: at least max(n*m, m*p, p*n) = 32
    lower_bound = max(n * m, m * p, p * n)
    
    np.random.seed(42)
    
    for r in range(best_rank - 1, max(lower_bound - 1, best_rank - 30), -1):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Random initialization
            U_rand = np.random.randn(n * m, r)
            V_rand = np.random.randn(m * p, r)
            W_rand = np.random.randn(p * n, r)
            
            # Optimize
            U_opt, V_opt, W_opt, loss = _optimize_decomposition(
                U_rand, V_rand, W_opt if False else W_rand, T,
                max_iters=150, lr_init=0.01, deadline=deadline
            )
            
            if loss < 1e-8:
                # Verify
                U_v, V_v, W_v = _snap_to_half_integers(U_opt, V_opt, W_opt)
                if _verify_decomposition(U_v, V_v, W_v, T):
                    found = True
                    best_rank = r
                    best_decomp = {
                        "U": U_v.tolist(),
                        "V": V_v.tolist(),
                        "W": W_v.tolist()
                    }
                    break
            
            # Also try polishing even if not perfectly converged
            if loss < 1e-4 and time.time() < deadline:
                U_p, V_p, W_p = _polish_decomposition(U_opt, V_opt, W_opt, deadline)
                if _verify_decomposition(U_p, V_p, W_p, T):
                    found = True
                    best_rank = r
                    best_decomp = {
                        "U": U_p.tolist(),
                        "V": V_p.tolist(),
                        "W": W_p.tolist()
                    }
                    break
        
        if found:
            # Try to go even lower
            continue
    
    return best_rank, best_decomp


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Leave 20s buffer for the 150s limit
    
    # Try to find a smaller decomposition
    rank, decomp = _try_find_smaller_decomposition(n, m, p, deadline)
    
    return decomp
# EVOLVE-BLOCK-END
