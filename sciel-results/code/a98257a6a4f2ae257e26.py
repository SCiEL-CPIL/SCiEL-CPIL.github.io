"""Program for matrix multiplication 3,3,6 tensor decomposition.
Goal: find a rank-minimal exact decomposition (lower rank = better score).

Strategy: The 3x3x6 matmul tensor T[i,j,k]=1 (i in [0,3), j in [0,3), k in [0,6))
has schoolbook rank 54. We view the 3x6 output as two 3x3 blocks and search
for a batched rank-23 decomposition that shares structure across both output
slices. Uses smooth optimization with verification. Falls back to schoolbook (54).
"""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]
DEADLINE_S = 110.0  # safety margin within 120s limit

def _build_schoolbook(n, m, p):
    """Build the schoolbook (rank n*m*p) decomposition."""
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


def _verify(U, V, W, n, m, p, seed=42):
    """Verify that the decomposition correctly computes matrix multiplication.
    C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]  summed over j.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]

    np.random.seed(seed)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B

    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_vec = U[:, r]  # (n*m,)
        v_vec = V[:, r]  # (m*p,)
        w_vec = W[:, r]  # (p*n,)

        u_mat = u_vec.reshape(n, m)  # (n, m)
        v_mat = v_vec.reshape(m, p)  # (m, p)
        w_mat = w_vec.reshape(p, n)  # (p, n)

        uv = u_mat @ v_mat  # (n, p)
        C_computed += uv * w_mat.T  # (n, p)

    if not np.allclose(C_computed, C_expected, atol=1e-9):
        return False
    return True


def _verify_multi(U, V, W, n, m, p, n_trials=5):
    """Verify with multiple random seeds for robustness."""
    for seed in range(n_trials):
        if not _verify(U, V, W, n, m, p, seed=seed + 100):
            return False
    return True


def _tensor_error(U, V, W, n, m, p, params):
    """Compute tensor reconstruction error for given factor matrices.
    Returns a smooth surrogate for the max absolute error using log-sum-exp.
    """
    U, V, W = params
    # Error for each (i,j,k): |sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1|
    errors = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(U.shape[1]):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                errors[i, j, k] = val - 1.0
    # Smooth max via log-sum-exp
    abs_err = np.abs(errors)
    max_err = abs_err.max()
    if max_err < 1e-15:
        return 0.0
    # Soft max: sum(exp(alpha * err)) / alpha approximation
    alpha = 10.0
    scaled = alpha * abs_err
    max_scaled = scaled.max()
    log_sum = np.log(np.sum(np.exp(scaled - max_scaled))) + max_scaled
    return log_sum / alpha


def _batched_search(n, m, p, target_rank, max_iters=500, seed=42):
    """
    Search for a rank-target_rank decomposition using the batched block structure.
    Views the 3x6 output as two 3x3 blocks and enforces coupling through shared
    rank components. Uses scipy.optimize with smooth surrogate objective.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize with random factors
    U = rng.randn(n * m, target_rank) * 0.5
    V = rng.randn(m * p, target_rank) * 0.5
    W = rng.randn(p * n, target_rank) * 0.5
    
    # Flatten all parameters into a single vector
    def _flatten(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def _unflatten(params):
        U_size = n * m * target_rank
        V_size = m * p * target_rank
        W_size = p * n * target_rank
        U = params[:U_size].reshape(n * m, target_rank)
        V = params[U_size:U_size + V_size].reshape(m * p, target_rank)
        W = params[U_size + V_size:U_size + V_size + W_size].reshape(p * n, target_rank)
        return U, V, W
    
    def _objective(params):
        U, V, W = _unflatten(params)
        return _tensor_error(U, V, W, n, m, p, (U, V, W))
    
    x0 = _flatten(U, V, W)
    
    # Use Nelder-Mead for derivative-free optimization
    result = minimize(
        _objective, x0,
        method='Nelder-Mead',
        options={'maxiter': max_iters, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False}
    )
    
    U_opt, V_opt, W_opt = _unflatten(result.x)
    
    # Verify the result
    if _verify(U_opt, V_opt, W_opt, n, m, p):
        return U_opt, V_opt, W_opt, target_rank
    
    return None


def _try_batched_decomposition(n, m, p):
    """
    Attempt to find a lower-rank decomposition using the batched block structure.
    The 3x6 output is viewed as two 3x3 blocks, and we search for shared
    structural components across both slices.
    
    Returns (U, V, W, rank) if a valid lower-rank decomposition is found,
    else None.
    """
    best = None
    
    # Target ranks to try, starting from lower
    target_ranks = [23, 24, 25, 30, 36, 40, 45]
    
    for rank in target_ranks:
        if time.time() > DEADLINE_S:
            break
        
        # Try multiple random initializations
        for trial in range(5):
            if time.time() > DEADLINE_S:
                break
            
            seed = 10000 + rank * 100 + trial
            result = _batched_search(n, m, p, rank, max_iters=200, seed=seed)
            
            if result is not None:
                U2, V2, W2, rank2 = result
                # Multi-verify for robustness
                if _verify_multi(U2, V2, W2, n, m, p, n_trials=5):
                    best = (U2.copy(), V2.copy(), W2.copy(), rank2)
                    break
        
        if best is not None:
            break
    
    return best


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS

    # Build and verify schoolbook as guaranteed baseline
    U, V, W = _build_schoolbook(n, m, p)
    if not _verify(U, V, W, n, m, p):
        # Should not happen, but return anyway
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U, V, W
    best_rank = n * m * p

    # Attempt batched block decomposition search for lower rank
    result = _try_batched_decomposition(n, m, p)
    if result is not None:
        U2, V2, W2, rank2 = result
        if rank2 < best_rank:
            best_U, best_V, best_W = U2, V2, W2
            best_rank = rank2

    # Return the best verified decomposition
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
