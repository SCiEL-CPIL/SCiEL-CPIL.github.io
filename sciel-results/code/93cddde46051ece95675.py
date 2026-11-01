"""Tensor decomposition search for 3x3 matrix multiplication."""
DIMENSIONS = [3, 3, 3]
import time
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
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
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    T = np.zeros((dim_a, dim_b, dim_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-8):
    r = U.shape[1]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    T_exp = _build_target_tensor(n, m, p)
    return np.allclose(T, T_exp, atol=tol, rtol=0)

def _rank_one_projection(U, V, W, n, m, p):
    """Compute the rank-one tensor contribution for each term."""
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    r = U.shape[1]
    # For each term r, the tensor is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # Sum all terms
    T = np.zeros((dim_a, dim_b, dim_c))
    for rr in range(r):
        T += np.einsum('a,b,c->abc', U[:,rr], V[:,rr], W[:,rr])
    return T

def _als_step(T, U, V, W, n, m, p):
    """One ALS iteration. Returns updated U, V, W and the current error."""
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    r = U.shape[1]
    
    # Update U
    M = np.einsum('br,cr->bcr', V, W).reshape(dim_b * dim_c, r)
    for a in range(dim_a):
        T_flat = T[a, :, :].reshape(-1)
        U[a, :], _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
    
    # Update V
    M = np.einsum('ar,cr->acr', U, W).reshape(dim_a * dim_c, r)
    for b in range(dim_b):
        T_flat = T[:, b, :].reshape(-1)
        V[b, :], _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
    
    # Update W
    M = np.einsum('ar,br->abr', U, V).reshape(dim_a * dim_b, r)
    for c in range(dim_c):
        T_flat = T[:, :, c].reshape(-1)
        W[c, :], _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
    
    # Compute error
    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.linalg.norm(T_approx - T)
    return U, V, W, error

def _als_decompose(T, rank, n, m, p, init_U=None, init_V=None, init_W=None, 
                   max_iters=300, tol=1e-10, deadline=None, verbose=False):
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    
    if init_U is not None:
        U, V, W = init_U.copy(), init_V.copy(), init_W.copy()
    else:
        U = np.random.randn(dim_a, rank)
        V = np.random.randn(dim_b, rank)
        W = np.random.randn(dim_c, rank)
    
    # Normalize columns
    for rr in range(rank):
        nu = np.linalg.norm(U[:, rr])
        nv = np.linalg.norm(V[:, rr])
        nw = np.linalg.norm(W[:, rr])
        if nu > 1e-10 and nv > 1e-10 and nw > 1e-10:
            U[:, rr] /= nu
            V[:, rr] /= nv
            W[:, rr] /= nw
    
    prev_error = float('inf')
    stall = 0
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        U, V, W, error = _als_step(T, U, V, W, n, m, p)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if verbose:
            print(f"  ALS iter {it}: error={error:.2e}, best={best_error:.2e}")
        
        if abs(prev_error - error) < tol:
            stall += 1
            if stall >= 5:
                break
        else:
            stall = 0
        prev_error = error
    
    return best_U, best_V, best_W, best_error

def _polish(U, V, W, T, n, m, p, deadline, max_iters=500, tol=1e-14):
    """High-precision polish with very tight tolerance."""
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        U, V, W, error = _als_step(T, U, V, W, n, m, p)
        if error < tol:
            break
    return U, V, W, error

def _try_rank(target_rank, n, m, p, deadline, max_attempts=200, seed=42):
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    T = _build_target_tensor(n, m, p)
    U0, V0, W0 = _schoolbook(n, m, p)
    total_terms = n*m*p
    n_drop = total_terms - target_rank
    
    if n_drop <= 0:
        return U0, V0, W0
    
    rng = random.Random(seed)
    all_indices = list(range(total_terms))
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Generate diverse initializations
        strategy = attempt % 4
        
        if strategy == 0:
            # Drop random terms from schoolbook + small perturbation
            drop = set(rng.sample(all_indices, n_drop))
            keep = [i for i in all_indices if i not in drop]
            U = U0[:, keep].copy()
            V = V0[:, keep].copy()
            W = W0[:, keep].copy()
            scale = rng.uniform(0.01, 0.1)
            for rr in range(target_rank):
                for mask, mat in [(U[:, rr] != 0, U), (V[:, rr] != 0, V), (W[:, rr] != 0, W)]:
                    if mask.any():
                        mat[mask, rr] += rng.uniform(-scale, scale)
        elif strategy == 1:
            # Drop random terms + larger perturbation
            drop = set(rng.sample(all_indices, n_drop))
            keep = [i for i in all_indices if i not in drop]
            U = U0[:, keep].copy()
            V = V0[:, keep].copy()
            W = W0[:, keep].copy()
            scale = rng.uniform(0.1, 0.5)
            for rr in range(target_rank):
                for mask, mat in [(U[:, rr] != 0, U), (V[:, rr] != 0, V), (W[:, rr] != 0, W)]:
                    if mask.any():
                        mat[mask, rr] += rng.uniform(-scale, scale)
        elif strategy == 2:
            # Sparse random initialization
            density = rng.uniform(0.2, 0.4)
            U = np.zeros((dim_a, target_rank))
            V = np.zeros((dim_b, target_rank))
            W = np.zeros((dim_c, target_rank))
            for rr in range(target_rank):
                for a in range(dim_a):
                    if rng.random() < density:
                        U[a, rr] = rng.uniform(-1, 1)
                for b in range(dim_b):
                    if rng.random() < density:
                        V[b, rr] = rng.uniform(-1, 1)
                for c in range(dim_c):
                    if rng.random() < density:
                        W[c, rr] = rng.uniform(-1, 1)
        else:
            # Fully random
            U = rng.uniform(-1, 1, size=(dim_a, target_rank))
            V = rng.uniform(-1, 1, size=(dim_b, target_rank))
            W = rng.uniform(-1, 1, size=(dim_c, target_rank))
        
        # Run ALS
        U, V, W, err = _als_decompose(T, target_rank, n, m, p,
                                       init_U=U, init_V=V, init_W=W,
                                       max_iters=150, tol=1e-12,
                                       deadline=deadline)
        
        if err < 1e-7:
            # Polish
            U, V, W, err2 = _polish(U, V, W, T, n, m, p, deadline, max_iters=300, tol=1e-14)
            if err2 < 1e-8:
                if _verify(U, V, W, n, m, p, tol=1e-8):
                    return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill
    
    # Try ranks from 23 upward
    for target_rank in [23, 24, 25, 26]:
        if time.time() > deadline:
            break
        result = _try_rank(target_rank, n, m, p, deadline, max_attempts=300, seed=42)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
