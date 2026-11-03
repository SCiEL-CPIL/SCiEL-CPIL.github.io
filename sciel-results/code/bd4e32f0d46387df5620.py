"""Find rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 3, 4]
DEADLINE = 140.0  # Safety margin before 150s kill

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 for all i,j,k."""
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _compute_residual(U, V, W, n, m, p):
    """Compute ||T - sum_r U_r V_r W_r||_F^2 using vectorized ops."""
    rank = U.shape[1]
    U_mat = U.reshape(n, m, rank)  # [i, j, r]
    V_mat = V.reshape(m, p, rank)  # [j, k, r]
    W_mat = W.reshape(p, n, rank)  # [k, i, r]
    
    T_approx = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        # T_approx[i,j,k] += U[i,j,r] * V[j,k,r] * W[k,i,r]
        T_approx += np.einsum('ijr,jkr,kir->ijk', U_mat, V_mat, W_mat)
        # Actually let's do it more carefully per term
        break  # redo
    
    # Better: loop over r and accumulate
    T_approx = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        u_r = U_mat[:, :, r]  # n x m
        v_r = V_mat[:, :, r]  # m x p
        w_r = W_mat[:, :, r]  # p x n
        # Contribution: T_r[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[k,i]
        # This is outer product structure
        # For fixed r: T_r[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[k,i]
        # = (u_r[i,j] * v_r[j,k]) * w_r[k,i]
        # Let A[i,k] = sum_j u_r[i,j] * v_r[j,k]  (matrix mult u_r @ v_r)
        # Then T_r[i,j,k] = ... hmm, this isn't right because v_r depends on j
        # Let's just compute directly
        T_approx += np.einsum('ij,jk,ki->ijk', u_r, v_r, w_r)
    
    T_target = np.ones((n, m, p), dtype=float)
    residual = np.sum((T_approx - T_target) ** 2)
    return residual

def _verify_exact(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact (within tolerance)."""
    residual = _compute_residual(U, V, W, n, m, p)
    return residual < tol

def _snap_to_half_int(x):
    """Snap a float to nearest half-integer."""
    return round(2 * x) / 2.0

def _is_half_integer(val, tol=1e-9):
    """Check if value is a half-integer."""
    doubled = 2 * val
    return abs(doubled - round(doubled)) < tol

def _check_all_half_int(U, V, W, tol=1e-9):
    """Check all entries are half-integers."""
    for arr in [U, V, W]:
        doubled = 2 * arr
        if not np.all(np.abs(doubled - np.round(doubled)) < tol):
            return False
    return True

def _snap_all_to_half_int(U, V, W):
    """Snap all entries to nearest half-integer."""
    U_snapped = np.round(2 * U) / 2.0
    V_snapped = np.round(2 * V) / 2.0
    W_snapped = np.round(2 * W) / 2.0
    return U_snapped, V_snapped, W_snapped

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _random_init_half_int(n, m, p, rank, scale=1.0, seed=None):
    """Initialize U, V, W with random half-integers."""
    rng = np.random.RandomState(seed)
    U = rng.randint(-int(2*scale), int(2*scale)+1, size=(n*m, rank)) / 2.0
    V = rng.randint(-int(2*scale), int(2*scale)+1, size=(m*p, rank)) / 2.0
    W = rng.randint(-int(2*scale), int(2*scale)+1, size=(p*n, rank)) / 2.0
    return U, V, W

def _structured_init(n, m, p, rank, seed=None):
    """Structured initialization based on known low-rank patterns."""
    rng = np.random.RandomState(seed)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Base: schoolbook-like structure with some mixing
    # Start with a subset of schoolbook terms
    used = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if used >= rank:
                    break
                U[i * m + j, used] = 1.0
                V[j * p + k, used] = 1.0
                W[k * n + i, used] = 1.0
                used += 1
            if used >= rank:
                break
        if used >= rank:
            break
    
    # Add random perturbations to remaining columns
    for r in range(used, rank):
        U[:, r] = rng.randn(n * m) * 0.5
        V[:, r] = rng.randn(m * p) * 0.5
        W[:, r] = rng.randn(p * n) * 0.5
        # Snap to half-integers
        U[:, r] = np.round(2 * U[:, r]) / 2.0
        V[:, r] = np.round(2 * V[:, r]) / 2.0
        W[:, r] = np.round(2 * W[:, r]) / 2.0
    
    return U, V, W

def _coordinate_descent(U, V, W, n, m, p, max_iters=2000, tol=1e-10, time_limit=None):
    """Local search via coordinate descent with half-integer constraint."""
    start_time = time.time()
    U, V, W = U.copy(), V.copy(), W.copy()
    rank = U.shape[1]
    
    best_residual = _compute_residual(U, V, W, n, m, p)
    
    for iteration in range(max_iters):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        improved = False
        
        # Try updating each entry of U, V, W by +-0.5
        for arr_name, arr in [('U', U), ('V', V), ('W', W)]:
            for idx in range(arr.size):
                old_val = arr.flat[idx]
                # Try old_val + 0.5
                arr.flat[idx] = old_val + 0.5
                res = _compute_residual(U, V, W, n, m, p)
                if res < best_residual - tol:
                    best_residual = res
                    improved = True
                else:
                    arr.flat[idx] = old_val
                # Try old_val - 0.5
                arr.flat[idx] = old_val - 0.5
                res = _compute_residual(U, V, W, n, m, p)
                if res < best_residual - tol:
                    best_residual = res
                    improved = True
                else:
                    arr.flat[idx] = old_val
        
        if not improved:
            break
    
    return U, V, W, best_residual

def _gradient_descent_refine(U, V, W, n, m, p, lr=0.1, max_iters=500, tol=1e-10):
    """Gradient-based refinement (continuous), then snap to half-integers."""
    U, V, W = U.copy().astype(float), V.copy().astype(float), W.copy().astype(float)
    rank = U.shape[1]
    
    for _ in range(max_iters):
        # Compute gradient numerically (central difference)
        eps = 1e-7
        gU = np.zeros_like(U)
        gV = np.zeros_like(V)
        gW = np.zeros_like(W)
        
        for idx in range(U.size):
            U_plus = U.copy()
            U_plus.flat[idx] += eps
            res_plus = _compute_residual(U_plus, V, W, n, m, p)
            U_minus = U.copy()
            U_minus.flat[idx] -= eps
            res_minus = _compute_residual(U_minus, V, W, n, m, p)
            gU.flat[idx] = (res_plus - res_minus) / (2 * eps)
        
        for idx in range(V.size):
            V_plus = V.copy()
            V_plus.flat[idx] += eps
            res_plus = _compute_residual(U, V_plus, W, n, m, p)
            V_minus = V.copy()
            V_minus.flat[idx] -= eps
            res_minus = _compute_residual(U, V_minus, W, n, m, p)
            gV.flat[idx] = (res_plus - res_minus) / (2 * eps)
        
        for idx in range(W.size):
            W_plus = W.copy()
            W_plus.flat[idx] += eps
            res_plus = _compute_residual(U, V, W_plus, n, m, p)
            W_minus = W.copy()
            W_minus.flat[idx] -= eps
            res_minus = _compute_residual(U, V, W_minus, n, m, p)
            gW.flat[idx] = (res_plus - res_minus) / (2 * eps)
        
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
        
        if np.max(np.abs(gU)) + np.max(np.abs(gV)) + np.max(np.abs(gW)) < tol:
            break
    
    # Snap to half-integers
    U_snapped, V_snapped, W_snapped = _snap_all_to_half_int(U, V, W)
    residual = _compute_residual(U_snapped, V_snapped, W_snapped, n, m, p)
    return U_snapped, V_snapped, W_snapped, residual

def _try_rank(n, m, p, rank, time_budget=20.0, num_restarts=10):
    """Try to find a valid decomposition of given rank."""
    start_time = time.time()
    best_U, best_V, best_W = None, None, None
    best_res = float('inf')
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Choose initialization strategy
        if restart % 3 == 0:
            U, V, W = _random_init_half_int(n, m, p, rank, scale=2.0, seed=restart)
        elif restart % 3 == 1:
            U, V, W = _structured_init(n, m, p, rank, seed=restart)
        else:
            U, V, W = _random_init_half_int(n, m, p, rank, scale=1.0, seed=restart + 1000)
        
        # Check if already exact
        res = _compute_residual(U, V, W, n, m, p)
        if res < 1e-9:
            return U, V, W, res
        
        # Coordinate descent refinement
        U, V, W, res = _coordinate_descent(U, V, W, n, m, p, max_iters=500, time_limit=time_budget/4)
        
        if res < best_res:
            best_res = res
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if res < 1e-9:
            return best_U, best_V, best_W, res
        
        # Gradient refinement + snap
        U2, V2, W2, res2 = _gradient_descent_refine(U, V, W, n, m, p, lr=0.05, max_iters=200)
        if res2 < best_res:
            best_res = res2
            best_U, best_V, best_W = U2.copy(), V2.copy(), W2.copy()
        
        if res2 < 1e-9:
            return best_U, best_V, best_W, res2
    
    if best_res < 1e-9:
        return best_U, best_V, best_W, best_res
    return None, None, None, best_res

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Fallback: schoolbook
    U_s, V_s, W_s = _schoolbook(n, m, p)
    
    # Try ranks from 35 down to 20
    for rank in range(35, 19, -1):
        elapsed = time.time() - start_time
        time_budget = min(15.0, DEADLINE - elapsed - 10.0)  # Leave 10s safety
        if time_budget < 3.0:
            break
        
        print(f"Trying rank {rank}, time budget {time_budget:.1f}s")
        U, V, W, res = _try_rank(n, m, p, rank, time_budget=time_budget, num_restarts=5)
        
        if U is not None and res < 1e-9:
            print(f"Found rank {rank} decomposition!")
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return schoolbook as fallback
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
