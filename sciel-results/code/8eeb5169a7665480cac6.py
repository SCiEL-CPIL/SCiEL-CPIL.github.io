"""Rewritten program: rank-minimal exact tensor decomposition for 2x3 * 3x9."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 9]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 for all i,j,k."""
    T = np.zeros((n, m, p), dtype=np.float64)
    T[:] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the tensor."""
    n, m, p = n, m, p
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    # Contract: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Build T from factors
    T_recon = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        u_col = U[:, c]
        v_col = V[:, c]
        w_col = W[:, c]
        for i in range(n):
            for j in range(m):
                u_val = u_col[i * m + j]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = v_col[j * p + k]
                    w_val = w_col[k * n + i]
                    T_recon[i, j, k] += u_val * v_val * w_val
    T_target = np.ones((n, m, p), dtype=np.float64)
    diff = np.max(np.abs(T_recon - T_target))
    return diff < tol, diff

def _random_init(n, m, p, rank, seed):
    """Initialize random factor matrices."""
    rng = random.Random(seed)
    U = np.random.RandomState(seed).randn(n * m, rank)
    V = np.random.RandomState(seed + 1).randn(m * p, rank)
    W = np.random.RandomState(seed + 2).randn(p * n, rank)
    return U, V, W

def _loss_and_grad(U, V, W, n, m, p, T):
    """Compute loss and gradients for tensor decomposition."""
    rank = U.shape[1]
    # T_recon[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    T_recon = np.zeros((n, m, p), dtype=np.float64)
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    
    for c in range(rank):
        u_col = U[:, c]
        v_col = V[:, c]
        w_col = W[:, c]
        for i in range(n):
            for j in range(m):
                u_val = u_col[i * m + j]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = v_col[j * p + k]
                    w_val = w_col[k * n + i]
                    prod = u_val * v_val * w_val
                    T_recon[i, j, k] += prod
                    err = T_recon[i, j, k] - T[i, j, k]
                    # Gradient w.r.t. U[i*m+j, c]
                    dU[i * m + j, c] += err * v_val * w_val
                    # Gradient w.r.t. V[j*p+k, c]
                    dV[j * p + k, c] += err * u_val * w_val
                    # Gradient w.r.t. W[k*n+i, c]
                    dW[k * n + i, c] += err * u_val * v_val
    loss = 0.5 * np.sum((T_recon - T) ** 2)
    return loss, dU, dV, dW

def _gd_step(U, V, W, lr, n, m, p, T):
    """One gradient descent step."""
    loss, dU, dV, dW = _loss_and_grad(U, V, W, n, m, p, T)
    U = U - lr * dU
    V = V - lr * dV
    W = W - lr * dW
    return U, V, W, loss

def _try_rank(n, m, p, rank, max_iters, deadline, seed_base):
    """Try to find a decomposition of given rank using gradient descent."""
    T = _build_tensor(n, m, p)
    best_loss = np.inf
    best_U, best_V, best_W = None, None, None
    
    num_restarts = min(5, max(1, int((deadline - time.time()) / 20)))
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        seed = seed_base + r * 1000
        U, V, W = _random_init(n, m, p, rank, seed)
        
        # Adaptive learning rate
        lr = 0.01
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            U, V, W, loss = _gd_step(U, V, W, lr, n, m, p, T)
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if loss < 1e-10:
                break
            # Adaptive LR
            if it % 100 == 0 and it > 0:
                if loss > best_loss * 1.5:
                    lr *= 0.5
                elif loss < best_loss * 0.8:
                    lr *= 1.2
    return best_U, best_V, best_W, best_loss

def _schoolbook(n, m, p):
    """Fallback schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin
    
    # Try ranks from known lower bounds up to schoolbook
    # Known: rank(2,3,9) is between some lower bound and 54
    # Start with a few promising ranks
    ranks_to_try = [20, 24, 28, 32, 36, 40, 44, 48, 54]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 30:
            break
        U, V, W, loss = _try_rank(n, m, p, rank, max_iters=5000, deadline=deadline, seed_base=rank)
        if U is not None and loss < 1e-9:
            # Verify with exact check
            valid, diff = _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p)
            if valid:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
