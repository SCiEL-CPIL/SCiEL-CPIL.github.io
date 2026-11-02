"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares

def _build_tensor():
    """Build the matrix multiplication tensor for n x m times m x p."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k] = 1.0
    return T

def _reconstruct(U, V, W, n, m, p, rank):
    """Reconstruct the tensor from factor matrices."""
    recon = np.zeros((n, m, m, p))
    for c in range(rank):
        u4 = U[:, c].reshape(n, m)
        v4 = V[:, c].reshape(m, p)
        w4 = W[:, c].reshape(p, n)
        recon += u4[:, :, np.newaxis, np.newaxis] * \
                 v4[np.newaxis, np.newaxis, :, :] * \
                 w4.T[np.newaxis, np.newaxis, :, :]
    return recon

def _try_rank_with_pattern(rank, n, m, p, T, pattern_seed, max_iter=3000):
    """Try rank-r decomposition with a specific support pattern."""
    np.random.seed(pattern_seed)
    
    # For each factor matrix, decide which entries are non-zero
    # Use a sparse pattern: each row has at most 1-2 non-zeros
    U_size = n * m * rank
    V_size = m * p * rank
    W_size = p * n * rank
    total_params = U_size + V_size + W_size
    
    # Create a mask of which parameters to optimize
    # For small ranks, try dense first, then sparse patterns
    mask = np.ones(total_params, dtype=bool)
    
    # For rank <= 18, try dense optimization
    U0 = np.zeros((n * m, rank))
    V0 = np.zeros((m * p, rank))
    W0 = np.zeros((p * n, rank))
    
    # Initialize with random values
    U0 = np.random.randn(n * m, rank) * 0.5
    V0 = np.random.randn(m * p, rank) * 0.5
    W0 = np.random.randn(p * n, rank) * 0.5
    
    params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    def objective(params):
        U = params[:U_size].reshape(n * m, rank)
        V = params[U_size:U_size + V_size].reshape(m * p, rank)
        W = params[U_size + V_size:].reshape(p * n, rank)
        recon = _reconstruct(U, V, W, n, m, p, rank)
        return (recon - T).flatten()
    
    best_cost = np.inf
    best_params = None
    
    for trial in range(5):
        trial_params = params0 + np.random.randn(total_params) * 0.1
        try:
            result = least_squares(objective, trial_params, method='lm', max_nfev=1500)
            if result.cost < best_cost:
                best_cost = result.cost
                best_params = result.x
        except Exception:
            continue
    
    if best_params is None or best_cost > 1e-8:
        return None
    
    U = best_params[:U_size].reshape(n * m, rank)
    V = best_params[U_size:U_size + V_size].reshape(m * p, rank)
    W = best_params[U_size + V_size:].reshape(p * n, rank)
    
    # Snap to nearest half-integer
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    recon = _reconstruct(U_snap, V_snap, W_snap, n, m, p, rank)
    if np.allclose(recon, T, atol=1e-10):
        return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor()
    
    deadline = time.time() + 140
    
    # Try ranks from low to high with multiple random seeds
    for rank in range(1, 19):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for seed in range(10):
            if time.time() > deadline:
                break
            result = _try_rank_with_pattern(rank, n, m, p, T, seed)
            if result is not None:
                U, V, W = result
                return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
