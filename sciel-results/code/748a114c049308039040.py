"""Find rank-minimal exact tensor decomposition for 2x3 x 3x7 matrix multiplication."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > tol:
                    return False
    
    T_check = T.copy()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_check[a, b, c] = 0
    if np.max(np.abs(T_check)) > tol:
        return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
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

def _build_target_tensor(n, m, p):
    """Build the target tensor for matrix multiplication."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _structured_init_v2(n, m, p, target_rank, rng, mode=0):
    """Improved structured initialization."""
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    if mode == 0:
        # Block structure: for each j, create independent blocks
        cols_per_j = target_rank // m
        for j in range(m):
            start = j * cols_per_j
            end = start + cols_per_j
            for i in range(n):
                U[i * m + j, start:end] = rng.randn(cols_per_j) * 0.3
            for k in range(p):
                V[j * p + k, start:end] = rng.randn(cols_per_j) * 0.3
            for k in range(p):
                for i in range(n):
                    W[k * n + i, start:end] = rng.randn(cols_per_j) * 0.3
        remaining = target_rank - m * cols_per_j
        if remaining > 0:
            U[:, m * cols_per_j:] = rng.randn(n * m, remaining) * 0.2
            V[:, m * cols_per_j:] = rng.randn(m * p, remaining) * 0.2
            W[:, p * n, m * cols_per_j:] = rng.randn(p * n, remaining) * 0.2
    elif mode == 1:
        # Exploit n=2 symmetry: try to share structure between i=0 and i=1
        U = rng.randn(n * m, target_rank) * 0.3
        V = rng.randn(m * p, target_rank) * 0.3
        W = rng.randn(p * n, target_rank) * 0.3
        # Make U symmetric in some sense
        for j in range(m):
            U[0 * m + j, :] += U[1 * m + j, :] * 0.5
    elif mode == 2:
        # Sparse initialization
        U = rng.randn(n * m, target_rank) * 0.4
        V = rng.randn(m * p, target_rank) * 0.4
        W = rng.randn(p * n, target_rank) * 0.4
        for r in range(target_rank):
            if rng.rand() < 0.3:
                U[rng.choice(n * m, size=n * m // 4, replace=False), r] = 0
            if rng.rand() < 0.3:
                V[rng.choice(m * p, size=m * p // 4, replace=False), r] = 0
            if rng.rand() < 0.3:
                W[rng.choice(p * n, size=p * n // 4, replace=False), r] = 0
    elif mode == 3:
        # Perturbed schoolbook
        U0, V0, W0 = _schoolbook(n, m, p)
        if target_rank <= n * m * p:
            idx = rng.choice(n * m * p, size=target_rank, replace=False)
            U = U0[:, idx].copy() + rng.randn(n * m, target_rank) * 0.15
            V = V0[:, idx].copy() + rng.randn(m * p, target_rank) * 0.15
            W = W0[:, idx].copy() + rng.randn(p * n, target_rank) * 0.15
        else:
            U = U0.copy()
            V = V0.copy()
            W = W0.copy()
            U = np.hstack([U, rng.randn(n * m, target_rank - n * m * p) * 0.3])
            V = np.hstack([V, rng.randn(m * p, target_rank - n * m * p) * 0.3])
            W = np.hstack([W, rng.randn(p * n, target_rank - n * m * p) * 0.3])
    elif mode == 4:
        # Kronecker-inspired: exploit that 7 = 3+4 or other decompositions
        U = rng.randn(n * m, target_rank) * 0.35
        V = rng.randn(m * p, target_rank) * 0.35
        W = rng.randn(p * n, target_rank) * 0.35
        # Add structured components
        for j in range(m):
            block_size = target_rank // (m * 2)
            start = j * (target_rank // m)
            mid = start + block_size
            end = start + 2 * block_size
            U[j * n:(j + 1) * n, start:mid] += rng.randn(n, block_size) * 0.2
            V[j * p:(j + 1) * p, start:mid] += rng.randn(p, block_size) * 0.2
            W[:, start:mid] += rng.randn(p * n, block_size) * 0.15
    else:
        scale = 1.0 / (target_rank ** 0.3)
        U = rng.randn(n * m, target_rank) * scale
        V = rng.randn(m * p, target_rank) * scale
        W = rng.randn(p * n, target_rank) * scale
    
    return U, V, W

def _gradient_descent_improved(U, V, W, T_target, n, m, p, max_iter=3000, tol=1e-10, seed=42, momentum=0.9):
    """Improved gradient descent with adaptive learning rate and better convergence."""
    rng = np.random.RandomState(seed)
    U = U.copy().astype(float)
    V = V.copy().astype(float)
    W = W.copy().astype(float)
    
    vU = np.zeros_like(U)
    vV = np.zeros_like(V)
    vW = np.zeros_like(W)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.015
    patience = 0
    best_patience_err = float('inf')
    lr_decay = 0.999
    
    for it in range(max_iter):
        T_curr = np.einsum('ar,br,cr->abc', U, V, W)
        residual = T_curr - T_target
        err = np.sum(residual ** 2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < best_patience_err - 1e-9:
                best_patience_err = err
                patience = 0
            else:
                patience += 1
        
        if best_err < tol:
            break
        
        if patience > 60:
            U = best_U + rng.randn(*best_U.shape) * 0.04
            V = best_V + rng.randn(*best_V.shape) * 0.04
            W = best_W + rng.randn(*best_W.shape) * 0.04
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            patience = 0
            lr = max(0.003, lr * 0.6)
            continue
        
        grad_U = np.einsum('abc,br,cr->ar', residual, V, W)
        grad_V = np.einsum('abc,ar,cr->br', residual, U, W)
        grad_W = np.einsum('abc,ar,br->cr', residual, U, V)
        
        U_norm = np.linalg.norm(U, axis=0, keepdims=True) + 1e-12
        V_norm = np.linalg.norm(V, axis=0, keepdims=True) + 1e-12
        W_norm = np.linalg.norm(W, axis=0, keepdims=True) + 1e-12
        
        grad_U = grad_U / (V_norm * W_norm)
        grad_V = grad_V / (U_norm * W_norm)
        grad_W = grad_W / (U_norm * V_norm)
        
        vU = momentum * vU + grad_U
        vV = momentum * vV + grad_V
        vW = momentum * vW + grad_W
        
        U -= lr * vU
        V -= lr * vV
        W -= lr * vW
        
        lr *= lr_decay
    
    return best_U, best_V, best_W, best_err

def _try_rank(n, m, p, target_rank, T_target, max_seeds=8, max_iter=2000, tol=1e-10, deadline=None):
    """Try to find a decomposition of given rank."""
    base_seed = 99999
    modes = [0, 1, 2, 3, 4]
    
    for seed_idx in range(max_seeds):
        if deadline is not None and time.time() > deadline - 3:
            break
        
        seed = base_seed + seed_idx * 777
        rng = np.random.RandomState(seed)
        mode = modes[seed_idx % len(modes)]
        
        U, V, W = _structured_init_v2(n, m, p, target_rank, rng, mode=mode)
        
        U, V, W, err = _gradient_descent_improved(U, V, W, T_target, n, m, p, 
                                                   max_iter=max_iter, tol=tol, seed=seed)
        
        if err < tol:
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=tol):
                return U, V, W, True
    
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = n * m * p
    
    T_target = _build_target_tensor(n, m, p)
    
    # Try ranks from low to high
    for target_rank in range(26, 42):
        if time.time() > deadline - 10:
            break
        
        max_seeds = 10 if target_rank < 30 else 6
        max_iter = 2500 if target_rank < 30 else 1500
        
        U, V, W, success = _try_rank(n, m, p, target_rank, T_target, 
                                       max_seeds=max_seeds, max_iter=max_iter, 
                                       tol=1e-10, deadline=deadline)
        
        if success:
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = target_rank
    
    # Try to polish if we found something better
    if best_rank < n * m * p and time.time() < deadline - 8:
        U_curr = np.array(best_result["U"])
        V_curr = np.array(best_result["V"])
        W_curr = np.array(best_result["W"])
        curr_rank = U_curr.shape[1]
        
        for target_rank in range(max(22, curr_rank - 3), curr_rank):
            if time.time() > deadline - 4:
                break
            
            U, V, W, success = _try_rank(n, m, p, target_rank, T_target,
                                           max_seeds=5, max_iter=1000,
                                           tol=1e-10, deadline=deadline)
            if success:
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = target_rank
    
    return best_result
# EVOLVE-BLOCK-END
