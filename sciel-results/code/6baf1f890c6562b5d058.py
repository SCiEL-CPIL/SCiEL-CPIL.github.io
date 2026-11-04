"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0
    
    target_abc = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_abc.append((a, b, c))
    target_set = set(target_abc)
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for (a, b, c) in target_abc:
            U[a, col] = 1
            V[b, col] = 1
            W[c, col] = 1
            col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    all_abc = [(a, b, c) for a in range(n*m) for b in range(m*p) for c in range(p*n)]
    T_vals = np.array([1.0 if (a, b, c) in target_set else 0.0 for (a, b, c) in all_abc])
    
    def als_step(U, V, W, r):
        for a in range(n*m):
            X = np.array([V[b, :] * W[c, :] for b in range(m*p) for c in range(p*n)])
            Tv = np.array([1.0 if (a, b, c) in target_set else 0.0 for b in range(m*p) for c in range(p*n)])
            U[a, :], _, _, _ = np.linalg.lstsq(X, Tv, rcond=None)
        
        for b in range(m*p):
            X = np.array([U[a, :] * W[c, :] for a in range(n*m) for c in range(p*n)])
            Tv = np.array([1.0 if (a, b, c) in target_set else 0.0 for a in range(n*m) for c in range(p*n)])
            V[b, :], _, _, _ = np.linalg.lstsq(X, Tv, rcond=None)
        
        for c in range(p*n):
            X = np.array([U[a, :] * V[b, :] for a in range(n*m) for b in range(m*p)])
            Tv = np.array([1.0 if (a, b, c) in target_set else 0.0 for a in range(n*m) for b in range(m*p)])
            W[c, :], _, _, _ = np.linalg.lstsq(X, Tv, rcond=None)
        
        return U, V, W
    
    def compute_error(U, V, W):
        vals = np.array([np.dot(U[a, :], V[b, :] * W[c, :]) for (a, b, c) in all_abc])
        return np.sum((vals - T_vals) ** 2)
    
    def try_rank_als(r, max_iter=200, seed=0):
        if time.time() > deadline - 10:
            return None
        np.random.seed(seed)
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            U, V, W = als_step(U, V, W, r)
            if it % 30 == 0:
                err = compute_error(U, V, W)
                if err < 1e-10:
                    return (U, V, W, r)
        err = compute_error(U, V, W)
        if err < 1e-8:
            return (U, V, W, r)
        return None
    
    for r in range(35, best_rank + 1):
        if time.time() > deadline - 20:
            break
        for seed in range(3):
            if time.time() > deadline - 15:
                break
            result = try_rank_als(r, max_iter=150, seed=seed)
            if result is not None:
                best_U, best_V, best_W, best_rank = result
                break
        if best_rank == r:
            break
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
