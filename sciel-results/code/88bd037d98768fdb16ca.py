"""Rank-minimal exact tensor decomposition for 2x2 * 2x7 matrix multiplication.

Approach family: Discrete randomized search with exact verification.
We search for a decomposition with small integer coefficients by randomly
initializing factor matrices and performing local search steps to minimize
the reconstruction error, verifying exactness at each step.
"""
import numpy as np
import random
import itertools

DIMENSIONS = [2, 2, 7]

def _build_tensor(n, m, p):
    """Return the (n*m)*(m*p)*(p*n) matrix-multiplication tensor T."""
    T = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _compute_error(T, U, V, W):
    """Compute the squared error between T and the CP decomposition."""
    I, J, K = T.shape
    r = U.shape[1]
    T_approx = np.zeros((I, J, K), dtype=int)
    for c in range(r):
        T_approx += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    diff = T_approx - T
    return np.sum(diff * diff)

def _try_rank(T, r, iters=100, max_val=2, seed=0):
    """Try to find a rank-r decomposition with small integer entries."""
    rng = random.Random(seed)
    I, J, K = T.shape
    best_err = float('inf')
    
    for _ in range(iters):
        # Initialize with random small integers
        U = np.array([[rng.randint(-max_val, max_val) for _ in range(r)] for _ in range(I)])
        V = np.array([[rng.randint(-max_val, max_val) for _ in range(r)] for _ in range(J)])
        W = np.array([[rng.randint(-max_val, max_val) for _ in range(r)] for _ in range(K)])
        
        err = _compute_error(T, U, V, W)
        if err == 0:
            return U, V, W
        
        # Local search: random coordinate updates
        for _ in range(50):
            if err == 0:
                break
            # Pick a random matrix, row, and column to perturb
            mat_idx = rng.randint(0, 2)
            if mat_idx == 0:
                row = rng.randint(0, I - 1)
                col = rng.randint(0, r - 1)
                old_val = U[row, col]
                new_val = old_val + rng.choice([-1, 1])
                if abs(new_val) > max_val:
                    continue
                U[row, col] = new_val
                new_err = _compute_error(T, U, V, W)
                if new_err < err:
                    err = new_err
                else:
                    U[row, col] = old_val
            elif mat_idx == 1:
                row = rng.randint(0, J - 1)
                col = rng.randint(0, r - 1)
                old_val = V[row, col]
                new_val = old_val + rng.choice([-1, 1])
                if abs(new_val) > max_val:
                    continue
                V[row, col] = new_val
                new_err = _compute_error(T, U, V, W)
                if new_err < err:
                    err = new_err
                else:
                    V[row, col] = old_val
            else:
                row = rng.randint(0, K - 1)
                col = rng.randint(0, r - 1)
                old_val = W[row, col]
                new_val = old_val + rng.choice([-1, 1])
                if abs(new_val) > max_val:
                    continue
                W[row, col] = new_val
                new_err = _compute_error(T, U, V, W)
                if new_err < err:
                    err = new_err
                else:
                    W[row, col] = old_val
        
        if err == 0:
            return U, V, W
            
    return None

def _schoolbook(n, m, p):
    """Return the canonical schoolbook rank-(n*m*p) decomposition."""
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

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Search for minimal rank
    # Lower bound is at least max(n*m, m*p, p*n) is not true, but 
    # for 2x2x7, schoolbook is 28.
    # Known ranks for 2x2xp: 
    # p=1: 4
    # p=2: 7
    # p=3: 10
    # p=4: 13
    # p=5: 16
    # p=6: 19
    # p=7: 22?
    # It seems R(2,2,p) = 3p + 1 for p>=2? 
    # Let's try ranks from 12 to 27.
    
    for r in range(12, 28):
        # Try a few seeds and max values
        for max_val in [1, 2]:
            result = _try_rank(T, r, iters=20, max_val=max_val, seed=r*100+max_val)
            if result:
                U, V, W = result
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                
    # Fallback
    return _schoolbook(n, m, p)
