"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """
    Build the matrix multiplication tensor T of size (n*m) x (m*p) x (p*n).
    T[i*m+j, j*p+k, k*n+i] = 1 for all valid i, j, k.
    All other entries are 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _get_nonzero_indices(T):
    """Get indices of non-zero entries in the tensor."""
    nz = np.argwhere(T != 0)
    return nz


def _verify_exact(T, U, V, W, tol=1e-8):
    """
    Verify that the decomposition is exact (within tolerance).
    T_{abc} = sum_r U_{ar} V_{br} W_{cr}
    """
    r = U.shape[1]
    recon = np.zeros(T.shape, dtype=np.float64)
    for rr in range(r):
        recon += np.einsum('a,b,c->abc', U[:, rr], V[:, rr], W[:, rr])
    residual = T - recon
    return np.max(np.abs(residual)) < tol


def _objective_and_grad(x, T, a_size, b_size, c_size, rank, nz_indices):
    """
    Compute objective (sum of squared residuals) and gradient.
    We only need to check non-zero entries and a sample of zero entries
    to keep the problem tractable, but for exactness we check all.
    
    Actually, for small tensors we can just compute the full residual.
    """
    U = x[:a_size*rank].reshape(a_size, rank)
    V = x[a_size*rank:a_size*rank+b_size*rank].reshape(b_size, rank)
    W = x[a_size*rank+b_size*rank:].reshape(c_size, rank)
    
    # Compute reconstruction only at non-zero entries for speed
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # We need this to equal T[a,b,c] for all (a,b,c)
    
    # For the objective, compute sum over all entries of (recon - T)^2
    # This is expensive for full tensor, so use non-zero + random zeros
    
    # Actually for 6x24x16 = 2304 entries, it's manageable
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    residual = recon - T
    obj = np.sum(residual**2)
    
    # Gradient computation
    # d/dU[a,r] = 2 * sum_{b,c} residual[a,b,c] * V[b,r] * W[c,r]
    # d/dV[b,r] = 2 * sum_{a,c} residual[a,b,c] * U[a,r] * W[c,r]
    # d/dW[c,r] = 2 * sum_{a,b} residual[a,b,c] * U[a,r] * V[b,r]
    
    grad_U = 2.0 * np.einsum('abc,br,cr->ar', residual, V, W)
    grad_V = 2.0 * np.einsum('abc,ar,cr->br', residual, U, W)
    grad_W = 2.0 * np.einsum('abc,ar,br->cr', residual, U, V)
    
    grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    
    return obj, grad


def _try_rank(T, n, m, p, rank, time_limit, seed=0):
    """
    Try to find a rank-`rank` decomposition of T.
    Uses multiple strategies: random restarts, structured initialization,
    and gradient-based optimization with polishing.
    """
    np.random.seed(seed)
    
    a_size = n * m
    b_size = m * p
    c_size = p * n
    nz = _get_nonzero_indices(T)
    
    start_time = time.time()
    
    def _time_remaining():
        return time_limit - (time.time() - start_time)
    
    def _flatten(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def _unflatten(x):
        U = x[:a_size*rank].reshape(a_size, rank)
        V = x[a_size*rank:a_size*rank+b_size*rank].reshape(b_size, rank)
        W = x[a_size*rank+b_size*rank:].reshape(c_size, rank)
        return U, V, W
    
    # Strategy 1: Structured initialization based on schoolbook
    # Each non-zero entry (i,j,k) maps to position (i*m+j, j*p+k, k*n+i)
    # For rank r, try to group entries
    
    # First, try a "block" initialization: assign each non-zero entry 
    # to a random column, then optimize
    max_restarts = 100
    
    for attempt in range(max_restarts):
        if _time_remaining() < 0.5:
            break
        
        # Different initialization strategies
        if attempt < 20:
            # Strategy A: Random sparse initialization
            # Each column gets a few non-zero entries
            U = np.zeros((a_size, rank))
            V = np.zeros((b_size, rank))
            W = np.zeros((c_size, rank))
            
            # Assign each non-zero tensor entry to a random column
            for idx in range(len(nz)):
                col = idx % rank
                a, b, c = nz[idx]
                # Set values to help the decomposition
                U[a, col] += 1.0
                V[b, col] += 1.0
                W[c, col] += 1.0
            
            # Add small random perturbation
            U += np.random.randn(a_size, rank) * 0.01
            V += np.random.randn(b_size, rank) * 0.01
            W += np.random.randn(c_size, rank) * 0.01
            
        elif attempt < 40:
            # Strategy B: Random dense initialization with small scale
            scale = 0.05 + 0.05 * (attempt - 20) / 20.0
            U = np.random.randn(a_size, rank) * scale
            V = np.random.randn(b_size, rank) * scale
            W = np.random.randn(c_size, rank) * scale
            
        elif attempt < 60:
            # Strategy C: Block diagonal structure
            # Divide non-zero entries into rank groups
            U = np.zeros((a_size, rank))
            V = np.zeros((b_size, rank))
            W = np.zeros((c_size, rank))
            
            entries_per_col = len(nz) // rank + 1
            for col in range(rank):
                start_idx = col * entries_per_col
                end_idx = min(start_idx + entries_per_col, len(nz))
                for idx in range(start_idx, end_idx):
                    a, b, c = nz[idx]
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
            
            U += np.random.randn(a_size, rank) * 0.02
            V += np.random.randn(b_size, rank) * 0.02
            W += np.random.randn(c_size, rank) * 0.02
            
        else:
            # Strategy D: Pure random with varying scales
            scale = 0.01 + 0.1 * (attempt - 60) / 40.0
            U = np.random.randn(a_size, rank) * scale
            V = np.random.randn(b_size, rank) * scale
            W = np.random.randn(c_size, rank) * scale
        
        x0 = _flatten(U, V, W)
        
        # Optimize with L-BFGS-B
        remaining = _time_remaining()
        maxiter = min(500, int(remaining * 50))
        
        if maxiter < 10:
            continue
        
        try:
            # Use custom gradient for better convergence
            def _obj_func(x):
                obj, _ = _objective_and_grad(x, T, a_size, b_size, c_size, rank, nz)
                return obj
            
            def _jac_func(x):
                _, grad = _objective_and_grad(x, T, a_size, b_size, c_size, rank, nz)
                return grad
            
            result = minimize(_obj_func, x0, jac=_jac_func, method='L-BFGS-B',
                              options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 40})
            
            if result.fun < 1e-10:
                U, V, W = _unflatten(result.x)
                if _verify_exact(T, U, V, W, tol=1e-8):
                    # Polishing pass with tighter tolerance
                    x_polish = result.x
                    try:
                        result2 = minimize(_obj_func, x_polish, jac=_jac_func, method='L-BFGS-B',
                                           options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 20})
                        if result2.fun < 1e-12:
                            U, V, W = _unflatten(result2.x)
                            if _verify_exact(T, U, V, W, tol=1e-10):
                                return U, V, W
                    except Exception:
                        pass
                    return U, V, W
                    
        except Exception:
            pass
        
        # If L-BFGS-B didn't work, try a few more restarts with different seeds
        if _time_remaining() < 2.0:
            break
    
    return None


def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    schoolbook_rank = n * m * p  # 48
    
    time_start = time.time()
    time_limit_total = 110  # Leave 10s margin for 120s limit (but evaluator kills at 150s)
    
    best_rank = schoolbook_rank
    best_decomp = None
    
    # Build schoolbook decomposition as fallback
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    # Lower bound for matrix multiplication rank
    # rank >= max(n*p, m*p, n*m) = max(16, 24, 6) = 24
    # But actually for this specific tensor, the lower bound might be higher
    lower_bound = max(n * p, m * p, n * m)
    
    # Try ranks from lower bound up to schoolbook_rank - 1
    # Use more time for lower ranks (more likely to be the answer)
    for rank in range(lower_bound, best_rank):
        elapsed = time.time() - time_start
        if elapsed > time_limit_total:
            break
        
        remaining_time = time_limit_total - elapsed
        # Allocate more time to lower ranks
        time_fraction = max(0.05, remaining_time / (best_rank - rank))
        per_try_time = min(time_fraction, 20.0)
        
        # Use different seeds for each rank
        seed = 42 + rank * 100
        
        result = _try_rank(T, n, m, p, rank, per_try_time, seed=seed)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_decomp = (U, V, W)
            # Continue searching for even lower rank? No, we go from low to high
            # So if we found rank r, we can stop (since we're going up)
            break
    
    def _to_list(mat):
        """Convert numpy array to list with integer/half-integer rounding."""
        result = []
        for row in mat:
            row_list = []
            for val in row:
                # Round to nearest integer or half-integer if close
                nearest_int = round(val)
                nearest_half = round(2 * val) / 2.0
                if abs(val - nearest_int) < 1e-6:
                    row_list.append(int(nearest_int))
                elif abs(val - nearest_half) < 1e-6:
                    row_list.append(nearest_half)
                else:
                    # Keep as float but round to reasonable precision
                    row_list.append(round(float(val), 10))
            result.append(row_list)
        return result
    
    if best_decomp is not None:
        U, V, W = best_decomp
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    else:
        # Schoolbook fallback
        return {"U": _to_list(U_sb), "V": _to_list(V_sb), "W": _to_list(W_sb)}

# EVOLVE-BLOCK-END
