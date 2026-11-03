import numpy as np
import scipy.optimize
import time
import random

DIMENSIONS = [3, 3, 4]

def _schoolbook():
    n, m, p = DIMENSIONS
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
    return U, V, W

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _precompute_indices():
    n, m, p = DIMENSIONS
    idx_i = np.array([i for i in range(n) for j in range(m) for k in range(p)])
    idx_j = np.array([j for i in range(n) for j in range(m) for k in range(p)])
    idx_k = np.array([k for i in range(n) for j in range(m) for k in range(p)])
    U_idx = idx_i * m + idx_j
    V_idx = idx_j * p + idx_k
    W_idx = idx_k * n + idx_i
    return U_idx, V_idx, W_idx

def _optimize_with_lbfgs(rank, seed=0, max_iter=300):
    np.random.seed(seed)
    n, m, p = DIMENSIONS
    U_idx, V_idx, W_idx = _precompute_indices()
    
    U0 = np.random.randn(n*m, rank) * 0.3
    V0 = np.random.randn(m*p, rank) * 0.3
    W0 = np.random.randn(p*n, rank) * 0.3
    
    x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    def objective(x):
        U = x[:n*m*rank].reshape(n*m, rank)
        V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        s = np.zeros(n*m*p)
        for c in range(rank):
            s += U[U_idx, c] * V[V_idx, c] * W[W_idx, c]
        return np.sum((s - 1.0) ** 2)
    
    result = scipy.optimize.minimize(objective, x0, method='L-BFGS-B', maxiter=max_iter)
    
    x = result.x
    U = x[:n*m*rank].reshape(n*m, rank)
    V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
    W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
    
    U_round = np.round(U * 2) / 2
    V_round = np.round(V * 2) / 2
    W_round = np.round(W * 2) / 2
    
    return U_round, V_round, W_round

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    best_rank = n * m * p
    best_U, best_V, best_W = _schoolbook()
    
    for rank in range(21, 30):
        if time.time() > deadline - 10:
            break
        
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            
            U, V, W = _optimize_with_lbfgs(rank, seed=seed, max_iter=300)
            
            if _verify(U, V, W):
                if rank < best_rank:
                    best_rank = rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
