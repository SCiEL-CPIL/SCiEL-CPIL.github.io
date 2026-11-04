"""
Signed autocorrelation maximum absolute-value variant.
Strategy: Sparse polynomial parameterization with differential evolution.
Model x_i as sum of K sparse terms with optimized positions, signs, and magnitudes.
This reduces dimensionality from N=400 to ~3K, enabling global optimization.
"""
N = 400
DEADLINE = 115  # seconds, safety margin before 120s limit

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.signal import fftconvolve

def _compute_metric(x):
    """Compute max |autoconv| / (sum x)^2, excluding the DC term (lag 0)."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return np.inf
    # Linear autoconvolution
    c = fftconvolve(x, x, mode='full')
    # Lag 0 is at index N-1
    # Exclude lag 0
    c_side = np.concatenate([c[:N-1], c[N:]])
    max_side = np.max(np.abs(c_side))
    return max_side / (s * s)

def _sparse_to_signal(params, N):
    """
    Convert sparse parameters to signal.
    params: array of length 3*K
    For each term k: [position, sign, magnitude]
    position in [0, N-1], sign in [-1, 1], magnitude in [0, 1]
    """
    K = len(params) // 3
    x = np.zeros(N, dtype=np.float64)
    for k in range(K):
        pos = int(params[3*k]) % N
        sign = params[3*k + 1]
        mag = params[3*k + 2]
        x[pos] += sign * mag
    return x

def _objective(params, N):
    """Objective function for differential evolution."""
    x = _sparse_to_signal(params, N)
    return _compute_metric(x)

def _construct_sparse(N, deadline):
    """Construct signal using sparse polynomial with differential evolution."""
    t0 = time.time()
    
    # Try different sparsities K
    best_x = np.ones(N, dtype=np.float64)
    best_val = _compute_metric(best_x)
    
    for K in [10, 15, 20, 30]:
        if time.time() - t0 > deadline - 10:
            break
        
        # Bounds for each parameter
        bounds = []
        for k in range(K):
            bounds.append((0, N - 1))      # position
            bounds.append((-1, 1))          # sign
            bounds.append((0, 1))           # magnitude
        
        # Run differential evolution
        try:
            result = differential_evolution(
                _objective,
                bounds,
                args=(N,),
                maxiter=50,
                popsize=15,
                tol=1e-4,
                seed=42,
                polish=False,
                updating='immediate'
            )
            
            if result.success or result.fun < best_val:
                x_candidate = _sparse_to_signal(result.x, N)
                val_candidate = _compute_metric(x_candidate)
                if val_candidate < best_val:
                    best_val = val_candidate
                    best_x = x_candidate.copy()
        except Exception:
            pass
        
        if time.time() - t0 > deadline - 5:
            break
    
    return best_x, best_val

def _local_refine(x, deadline):
    """Local refinement: flip signs to improve metric."""
    t0 = time.time()
    current_val = _compute_metric(x)
    n = len(x)
    
    # Compute convolution once
    c = fftconvolve(x, x, mode='full')
    
    def fast_update_conv(x, idx, old_val, new_val):
        """Fast update of convolution when one element changes."""
        delta = new_val - old_val
        # c[k] = sum_i x[i]*x[i+k]
        # Changing x[idx] by delta affects:
        # c[idx - j] += delta * x[j] for j where idx-j in [0, n-1]
        # c[idx + j] += delta * x[j] for j where idx+j in [0, n-1]
        # Actually, let's just recompute if it's fast enough
        return fftconvolve(x, x, mode='full')
    
    for i in range(2000):
        if time.time() - t0 > deadline - 2:
            break
        
        # Randomly pick an index to flip
        idx = np.random.randint(n)
        old_val = x[idx]
        new_val = -x[idx]
        x[idx] = new_val
        
        c_new = fftconvolve(x, x, mode='full')
        s_new = np.sum(x)
        
        if abs(s_new) < 1e-10:
            x[idx] = old_val
            continue
        
        c_side = np.concatenate([c_new[:N-1], c_new[N:]])
        max_side = np.max(np.abs(c_side))
        val_new = max_side / (s_new * s_new)
        
        if val_new < current_val:
            current_val = val_new
            c = c_new
        else:
            x[idx] = old_val
    
    return x, current_val

def construct():
    """
    Returns signed step heights.
    Uses sparse polynomial parameterization with differential evolution,
    followed by local refinement.
    """
    t0 = time.time()
    
    # Phase 1: Sparse global optimization
    x_sparse, val_sparse = _construct_sparse(N, DEADLINE)
    
    # Phase 2: Local refinement if time permits
    if time.time() - t0 < DEADLINE - 15:
        x_refined, val_refined = _local_refine(x_sparse, DEADLINE)
        if val_refined < val_sparse:
            return x_refined
        else:
            return x_sparse
    
    return x_sparse

# EVOLVE-BLOCK-END
