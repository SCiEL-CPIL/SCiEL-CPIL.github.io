import numpy as np
import time
import random
import sys

DIMENSIONS = [3, 3, 6]

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook decomposition."""
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, r] = 1.0
                V[b, r] = 1.0
                W[c, r] = 1.0
                r += 1
    return U, V, W

def _optimize_from_schoolbook(n, m, p, target_rank, max_iter=30000, seed=42, time_budget=20.0):
    """
    Initialize from schoolbook by randomly combining terms, then optimize.
    """
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    # Start with schoolbook
    U_s, V_s, W_s = _build_schoolbook(n, m, p)
    R_school = n * m * p
    
    # Create target_rank combinations of schoolbook terms
    # Each new term is a random linear combination of schoolbook terms
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    for r in range(target_rank):
        # Pick a random subset of schoolbook terms to combine
        weights = rng.randn(R_school) * 0.5
        U[:, r] = U_s @ weights
        V[:, r] = V_s @ weights
        W[:, r] = W_s @ weights
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # The 54 non-zero entries
    nonzero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero_indices.append((a, b, c))
    
    lr = 0.005
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        loss = 0.0
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for (a, b, c) in nonzero_indices:
            val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
            err = val - 1.0
            loss += err * err
            grad_vec = np.multiply(V[b, :], W[c, :])
            dU[a, :] += 2.0 * err * grad_vec
            dV[b, :] += 2.0 * err * np.multiply(U[a, :], W[c, :])
            dW[c, :] += 2.0 * err * np.multiply(U[a, :], V[b, :])
        
        # Off-diagonal penalty (sample)
        for _ in range(30):
            a = rng.randint(0, n * m - 1)
            b = rng.randint(0, m * p - 1)
            c = rng.randint(0, p * n - 1)
            i = a // m
            j = a % m
            jj = b // p
            k = b % p
            kk = c // n
            ii = c % n
            if i == ii and j == jj and j == k:
                continue
            val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
            loss += val * val
            grad_vec = np.multiply(V[b, :], W[c, :])
            dU[a, :] += 2.0 * val * grad_vec
            dV[b, :] += 2.0 * val * np.multiply(U[a, :], W[c, :])
            dW[c, :] += 2.0 * val * np.multiply(U[a, :], V[b, :])
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-20:
            break
        
        if loss < best_loss * 0.98:
            lr *= 1.01
        else:
            lr *= 0.99
        lr = min(lr, 0.02)
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    if best_loss < 1e-15:
        return best_U, best_V, best_W
    return None

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = 0.0
                for r in range(R):
                    val += U[a, r] * V[b, r] * W[c, r]
                if abs(val - 1.0) > tol:
                    return False
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i = a // m
                j = a % m
                jj = b // p
                k = b % p
                kk = c // n
                ii = c % n
                if i == ii and j == jj and j == k:
                    continue
                val = 0.0
                for r in range(R):
                    val += U[a, r] * V[b, r] * W[c, r]
                if abs(val) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    target_ranks = [46, 44, 42, 40, 38, 36, 34, 32, 30, 28]
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        found = False
        for seed in range(3):
            if time.time() > deadline:
                break
            
            result = _optimize_from_schoolbook(n, m, p, target_rank, max_iter=30000, seed=seed,
                                                time_budget=20.0)
            if result is not None:
                U_opt, V_opt, W_opt = result
                if _verify_decomposition(U_opt, V_opt, W_opt, n, m, p):
                    best_rank = target_rank
                    best_result = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
                    found = True
                    break
        
        if found:
            continue
    
    return best_result
