"""Tensor decomposition for 3x3x7 matrix multiplication."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START

def _get_target_indices(n, m, p):
    """Get all target (a,b,c) indices that should be 1."""
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                indices.append((a, b, c))
    return indices

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-8):
    """Fast verification using vectorized operations."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Check all 1-entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check a sample of 0-entries for speed
    rng = np.random.RandomState(42)
    for _ in range(500):
        a = rng.randint(0, n * m)
        b = rng.randint(0, m * p)
        c = rng.randint(0, p * n)
        i_a = a // m
        j_a = a % m
        j_b = b // p
        k_b = b % p
        k_c = c // n
        i_c = c % n
        if j_a == j_b and i_a == i_c and k_b == k_c:
            continue
        val = np.dot(U[a, :], V[b, :] * W[c, :])
        if abs(val) > tol:
            return False
    return True

def _objective(params, U_shape, V_shape, W_shape, target_indices, n, m, p):
    """Objective function for optimization: sum of squared errors."""
    r = U_shape[1]
    U = params[:U_shape[0]*r].reshape(U_shape)
    V = params[U_shape[0]*r:U_shape[0]*r + V_shape[0]*r].reshape(V_shape)
    W = params[U_shape[0]*r + V_shape[0]*r:].reshape(W_shape)
    
    loss = 0.0
    for a, b, c in target_indices:
        val = np.dot(U[a, :], V[b, :] * W[c, :])
        loss += (val - 1.0) ** 2
    
    # Add penalty for non-zero entries (sampled)
    rng = np.random.RandomState(123)
    for _ in range(100):
        a = rng.randint(0, U_shape[0])
        b = rng.randint(0, V_shape[0])
        c = rng.randint(0, W_shape[0])
        i_a = a // m
        j_a = a % m
        j_b = b // p
        k_b = b % p
        k_c = c // n
        i_c = c % n
        if not (j_a == j_b and i_a == i_c and k_b == k_c):
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            loss += val ** 2
    
    return loss

def _polish(U, V, W, target_indices, n, m, p, max_iter=100, tol=1e-10):
    """High-precision polish using alternating least squares."""
    U = U.copy().astype(np.float64)
    V = V.copy().astype(np.float64)
    W = W.copy().astype(np.float64)
    r = U.shape[1]
    
    for _ in range(max_iter):
        # Update U
        for a in range(U.shape[0]):
            # Collect all (b,c) pairs involving this a
            # For each target index where this a appears
            A_mat = np.zeros((len(target_indices), r))
            b_vals = np.zeros(len(target_indices))
            c_vals = np.zeros(len(target_indices))
            for idx, (ta, tb, tc) in enumerate(target_indices):
                if ta == a:
                    b_vals[idx] = tb
                    c_vals[idx] = tc
            
            # Build system: U[a,:] * (V[b,:] * W[c,:]) = 1 for all (b,c) with this a
            # This is underdetermined, so use least squares
            valid = b_vals != 0
            if np.sum(valid) > 0:
                b_valid = b_vals[valid].astype(int)
                c_valid = c_vals[valid].astype(int)
                # Build matrix where each row is V[b,:] * W[c,:]
                M = np.zeros((np.sum(valid), r))
                for i_idx in range(np.sum(valid)):
                    M[i_idx, :] = V[b_valid[i_idx], :] * W[c_valid[i_idx], :]
                # Solve M @ u = ones
                try:
                    u_new, _, _, _ = np.linalg.lstsq(M, np.ones(np.sum(valid)), rcond=None)
                    U[a, :] = u_new
                except:
                    pass
        
        # Update V similarly
        for b in range(V.shape[0]):
            b_vals = np.zeros(len(target_indices))
            c_vals = np.zeros(len(target_indices))
            a_vals = np.zeros(len(target_indices))
            for idx, (ta, tb, tc) in enumerate(target_indices):
                if tb == b:
                    a_vals[idx] = ta
                    c_vals[idx] = tc
            
            valid = a_vals != 0
            if np.sum(valid) > 0:
                a_valid = a_vals[valid].astype(int)
                c_valid = c_vals[valid].astype(int)
                M = np.zeros((np.sum(valid), r))
                for i_idx in range(np.sum(valid)):
                    M[i_idx, :] = U[a_valid[i_idx], :] * W[c_valid[i_idx], :]
                try:
                    v_new, _, _, _ = np.linalg.lstsq(M, np.ones(np.sum(valid)), rcond=None)
                    V[b, :] = v_new
                except:
                    pass
        
        # Update W similarly
        for c in range(W.shape[0]):
            a_vals = np.zeros(len(target_indices))
            b_vals = np.zeros(len(target_indices))
            for idx, (ta, tb, tc) in enumerate(target_indices):
                if tc == c:
                    a_vals[idx] = ta
                    b_vals[idx] = tb
            
            valid = a_vals != 0
            if np.sum(valid) > 0:
                a_valid = a_vals[valid].astype(int)
                b_valid = b_vals[valid].astype(int)
                M = np.zeros((np.sum(valid), r))
                for i_idx in range(np.sum(valid)):
                    M[i_idx, :] = U[a_valid[i_idx], :] * V[b_valid[i_idx], :]
                try:
                    w_new, _, _, _ = np.linalg.lstsq(M, np.ones(np.sum(valid)), rcond=None)
                    W[c, :] = w_new
                except:
                    pass
    
    return U, V, W

def _try_rank_structured(r, n, m, p, timeout=20):
    """Try rank-r with structured initialization."""
    start_time = time.time()
    target_indices = _get_target_indices(n, m, p)
    
    # Try several random seeds
    for seed in range(10):
        if time.time() - start_time > timeout:
            break
        
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, r) * 0.1
        V = rng.randn(m * p, r) * 0.1
        W = rng.randn(p * n, r) * 0.1
        
        # Alternating least squares polish
        U, V, W = _polish(U, V, W, target_indices, n, m, p, max_iter=50, tol=1e-8)
        
        if _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-7):
            # Final high-precision polish
            U, V, W = _polish(U, V, W, target_indices, n, m, p, max_iter=200, tol=1e-12)
            if _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
                return U, V, W
    
    return None

def _trivial_decomposition(n, m, p):
    """Return the trivial rank-n*m*p decomposition."""
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

def construct():
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Try decreasing ranks, starting from lower bounds
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # Known upper bounds from literature suggest rank around 21-25 for this case
    
    for r in [21, 22, 23, 24, 25, 26, 27]:
        if time.time() > deadline - 10:
            break
        
        timeout = min(20, deadline - time.time() - 5)
        if timeout < 2:
            break
        
        result = _try_rank_structured(r, n, m, p, timeout=timeout)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to trivial decomposition
    U, V, W = _trivial_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
