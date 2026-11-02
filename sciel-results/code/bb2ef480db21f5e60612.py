"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

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

def _tensor_residual(U, V, W, n, m, p):
    """Compute the residual tensor norm for a candidate decomposition."""
    nm, mp, pn = n * m, m * p, p * n
    r = U.shape[1]
    # Build the decomposed tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Target: T[i,j,k] = 1 if this is a valid multiplication term, else 0
    # Actually for matrix multiplication, the tensor is T[i,j,k] = 1 for all i,j,k
    # since C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is the all-ones tensor of shape (n, m, p)
    residual = np.zeros((n, m, p))
    for r in range(r):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        # Outer product contribution
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    residual[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    target = np.ones((n, m, p))
    return np.linalg.norm(residual - target)

def _try_rank(r, n, m, p, deadline, rng):
    """Try to find a rank-r decomposition. Returns (U, V, W) or None."""
    nm, mp, pn = n * m, m * p, p * n
    # Initialize with random vectors
    U = rng.standard_normal((nm, r)) * 0.1
    V = rng.standard_normal((mp, r)) * 0.1
    W = rng.standard_normal((pn, r)) * 0.1
    
    # Use alternating least squares / gradient descent
    def loss(params):
        if time.time() > deadline:
            return 1e10
        U_flat = params[:nm*r].reshape(nm, r)
        V_flat = params[nm*r:nm*r+mp*r].reshape(mp, r)
        W_flat = params[nm*r+mp*r:].reshape(pn, r)
        return _tensor_residual(U_flat, V_flat, W_flat, n, m, p)
    
    # Try a few random restarts
    best_loss = 1e10
    best_params = None
    n_restarts = 3
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        U = rng.standard_normal((nm, r)) * 0.5
        V = rng.standard_normal((mp, r)) * 0.5
        W = rng.standard_normal((pn, r)) * 0.5
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use Nelder-Mead for small dimensions
        try:
            result = minimize(loss, params, method='Nelder-Mead', 
                            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10,
                                   'adaptive': True})
            if result.fun < best_loss:
                best_loss = result.fun
                best_params = result.x
        except Exception:
            pass
    
    if best_params is None or best_loss > 1e-6:
        return None
    
    U = best_params[:nm*r].reshape(nm, r)
    V = best_params[nm*r:nm*r+mp*r].reshape(mp, r)
    W = best_params[nm*r+mp*r:].reshape(pn, r)
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook as fallback
    U_fb, V_fb, W_fb = _schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    rng = np.random.default_rng(42)
    
    # Try ranks below schoolbook
    # For 2x5x5, known bounds suggest rank might be reducible
    # Try a few candidate ranks
    candidate_ranks = [45, 44, 43, 42, 40, 38, 36, 35, 34, 32, 30]
    
    for r in candidate_ranks:
        if time.time() > deadline - 20:
            break
        result = _try_rank(r, n, m, p, deadline, rng)
        if result is not None:
            U, V, W = result
            # Verify the solution
            residual = _tensor_residual(U, V, W, n, m, p)
            if residual < 1e-6:
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break  # Found a valid lower rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
