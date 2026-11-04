"""Matrix multiplication 2x5x6: rank-minimal exact tensor decomposition."""
import numpy as np
import itertools
import time
from collections import deque

DIMENSIONS = [2, 5, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    r = U.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    result = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(result, T, atol=tol)

def _adam_step(U, V, W, T, mU, sU, mV, sV, mW, sW, t, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
    """One step of Adam optimizer for tensor decomposition."""
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    R = T - recon
    grad_U = -np.einsum('abc,br,cr->ar', R, V, W)
    grad_V = -np.einsum('abc,ar,cr->br', R, U, W)
    grad_W = -np.einsum('abc,ar,br->cr', R, U, V)
    
    mU = b1 * mU + (1 - b1) * grad_U
    sU = b2 * sU + (1 - b2) * (grad_U ** 2)
    mU_hat = mU / (1 - b1 ** t)
    sU_hat = sU / (1 - b2 ** t)
    U -= lr * mU_hat / (np.sqrt(sU_hat) + eps)
    
    mV = b1 * mV + (1 - b1) * grad_V
    sV = b2 * sV + (1 - b2) * (grad_V ** 2)
    mV_hat = mV / (1 - b1 ** t)
    sV_hat = sV / (1 - b2 ** t)
    V -= lr * mV_hat / (np.sqrt(sV_hat) + eps)
    
    mW = b1 * mW + (1 - b1) * grad_W
    sW = b2 * sW + (1 - b2) * (grad_W ** 2)
    mW_hat = mW / (1 - b1 ** t)
    sW_hat = sW / (1 - b2 ** t)
    W -= lr * mW_hat / (np.sqrt(sW_hat) + eps)
    
    return U, V, W, mU, sU, mV, sV, mW, sW

def _gradient_descent_adam(U, V, W, T, n, m, p, max_steps=200, lr=0.01, deadline=None):
    """Gradient descent with Adam optimizer."""
    nm, mp, pn = n*m, m*p, p*n
    r = U.shape[1]
    
    best_loss = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    mU = np.zeros_like(U)
    sU = np.zeros_like(U)
    mV = np.zeros_like(V)
    sV = np.zeros_like(V)
    mW = np.zeros_like(W)
    sW = np.zeros_like(W)
    
    for step in range(max_steps):
        if deadline and time.time() > deadline:
            break
        
        U, V, W, mU, sU, mV, sV, mW, sW = _adam_step(U, V, W, T, mU, sU, mV, sV, mW, sW, step + 1, lr=lr)
        
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        R = T - recon
        loss = np.sum(R**2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if loss < 1e-10:
            break
            
    return best_U, best_V, best_W, best_loss

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    best_rank = n * m * p
    best_result = None
    
    # Schoolbook fallback
    U_sb = np.zeros((nm, best_rank))
    V_sb = np.zeros((mp, best_rank))
    W_sb = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb)
    
    np.random.seed(42)
    
    # Try to find lower rank
    lower_bound = max(nm, mp, pn)
    
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline - 15:
            break
            
        found = False
        max_restarts = 10
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            scale = 1.0 / np.sqrt(rank)
            U = np.random.randn(nm, rank) * scale
            V = np.random.randn(mp, rank) * scale
            W = np.random.randn(pn, rank) * scale
            
            # Adam GD
            U, V, W, loss = _gradient_descent_adam(U, V, W, T, n, m, p, max_steps=100, lr=0.01, deadline=deadline - 5)
            
            if loss < 1e-8:
                if _verify(U, V, W, n, m, p, tol=1e-8):
                    found = True
                    best_rank = rank
                    best_result = (U, V, W)
                    break
            elif loss < 0.1:
                # Try to polish with more steps if close
                U, V, W, loss2 = _gradient_descent_adam(U, V, W, T, n, m, p, max_steps=50, lr=0.001, deadline=deadline - 5)
                if loss2 < 1e-8:
                    if _verify(U, V, W, n, m, p, tol=1e-8):
                        found = True
                        best_rank = rank
                        best_result = (U, V, W)
                        break
        
        if found:
            break
    
    U_final, V_final, W_final = best_result
    U_list = [[float(x) for x in row] for row in U_final]
    V_list = [[float(x) for x in row] for row in V_final]
    W_list = [[float(x) for x in row] for row in W_final]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
