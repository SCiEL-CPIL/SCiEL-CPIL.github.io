"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # 5s safety margin
    
    def build_target_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1
        return T
    
    def schoolbook_decomposition():
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
        return U, V, W, rank
    
    def check_decomposition(U, V, W, T=None):
        """Check if U, V, W exactly decompose the multiplication tensor."""
        if T is None:
            T = build_target_tensor()
        rank = U.shape[1]
        T_decomp = np.zeros((n * m, m * p, p * n))
        for r in range(rank):
            T_decomp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(T, T_decomp, atol=1e-10)
    
    def try_rank(r, max_iters=500, seed=42, T=None):
        """Try to find a rank-r decomposition using random initialization and local search."""
        if T is None:
            T = build_target_tensor()
        np.random.seed(seed)
        
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            # Compute current error
            T_decomp = np.zeros((n * m, m * p, p * n))
            for r_idx in range(r):
                T_decomp += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
            err = np.linalg.norm(T - T_decomp)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-10:
                break
            
            # Gradient-free step: perturb one factor
            factor = np.random.choice(3)
            r_idx = np.random.randint(r)
            if factor == 0:
                U[:, r_idx] += np.random.randn(n * m) * 0.1
            elif factor == 1:
                V[:, r_idx] += np.random.randn(m * p) * 0.1
            else:
                W[:, r_idx] += np.random.randn(p * n) * 0.1
        
        if best_err < 1e-10:
            return best_U, best_V, best_W, True
        return best_U, best_V, best_W, False
    
    # First, get the schoolbook decomposition
    U, V, W, rank = schoolbook_decomposition()
    T = build_target_tensor()
    
    if not check_decomposition(U, V, W, T):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find lower rank decompositions
    # Start from rank 53 down to 26
    for r in range(53, 25, -1):
        if time.time() > deadline:
            break
        for seed in [42 + r, 100 + r, 200 + r]:
            if time.time() > deadline:
                break
            U_r, V_r, W_r, success = try_rank(r, max_iters=300, seed=seed, T=T)
            if success:
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # If no lower rank found, return schoolbook
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
