"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _autoconv(x):
    """Full autoconvolution via FFT (zero-padded to avoid circular effects)."""
    n = len(x)
    size = 2 * n - 1
    # Pad to next power of 2 for FFT speed
    fft_size = 1
    while fft_size < size:
        fft_size <<= 1
    X = np.fft.rfft(x, n=fft_size)
    ac = np.fft.irfft(X * X.conj(), n=fft_size)[:size]
    return ac

def _objective(x):
    """Objective: max |autoconv| / (sum x)^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    ac = _autoconv(x)
    max_abs = np.max(np.abs(ac))
    return max_abs / (s * s)

def _objective_grad(x):
    """Objective and gradient via finite differences on the max-|ac| term."""
    n = len(x)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10, np.zeros(n)
    
    ac = _autoconv(x)
    max_abs = np.max(np.abs(ac))
    obj = max_abs / (s * s)
    
    # Gradient of max|ac| w.r.t. x: the sign of the ac element achieving max
    # ac[k] = sum_i x[i] * x[i+k] for k>=0 (with appropriate indexing)
    # d(ac[k])/dx[j] = x[j+k] + x[j-k] (for valid indices)
    # Use subgradient: pick the lag(s) where |ac| is maximal
    
    # Find indices where |ac| is near max
    tol = 1e-6
    mask = np.abs(ac) >= max_abs - tol
    max_indices = np.where(mask)[0]
    
    # Subgradient of max|ac|
    grad_max = np.zeros(n)
    for k in max_indices:
        sign = np.sign(ac[k]) if ac[k] != 0 else 1.0
        # d(ac[k])/dx[j] = x[j+k] (if j+k < n) + x[j-k] (if j-k >= 0)
        # ac[k] = sum_{j=0}^{n-1-k} x[j] * x[j+k]
        # d/dx[j] ac[k] = x[j+k] (for j in [0, n-1-k]) + x[j-k] (for j in [k, n-1])
        j_range1 = np.arange(0, n - k)
        j_range2 = np.arange(k, n)
        if len(j_range1) > 0:
            grad_max[j_range1] += sign * x[j_range1 + k]
        if len(j_range2) > 0:
            grad_max[j_range2] += sign * x[j_range2 - k]
    
    # Normalize by number of max indices for a smoother subgradient
    if len(max_indices) > 0:
        grad_max /= len(max_indices)
    
    # Gradient of obj = max_abs / s^2
    # d(obj)/dx[j] = (grad_max[j] * s^2 - max_abs * 2 * s) / s^4
    #             = grad_max[j] / s^2 - 2 * max_abs / s^3
    grad = grad_max / (s * s) - 2.0 * max_abs / (s ** 3)
    
    return obj, grad

def _local_search(x0, max_iter=2000, timeout=5.0):
    """L-BFGS-B local optimization with time guard."""
    start = time.time()
    best_x = x0.copy()
    best_obj = _objective(x0)
    
    def obj_and_grad(x):
        if time.time() - start > timeout:
            return _objective(x), np.zeros_like(x)
        return _objective_grad(x)
    
    try:
        res = minimize(obj_and_grad, x0, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
        if time.time() - start < timeout:
            obj_res = _objective(res.x)
            if obj_res < best_obj:
                best_x = res.x.copy()
                best_obj = obj_res
    except Exception:
        pass
    
    return best_x, best_obj

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    best_x = np.ones(N)
    best_obj = _objective(best_x)
    
    # Generate diverse initializations
    inits = []
    
    # 1. All ones (baseline)
    inits.append(np.ones(N))
    
    # 2. Random ±1
    rng = np.random.default_rng(42)
    for _ in range(3):
        inits.append(rng.choice([-1.0, 1.0], size=N))
    
    # 3. Block patterns
    for block_size in [10, 20, 50]:
        n_blocks = N // block_size
        signs = rng.choice([-1.0, 1.0], size=n_blocks + 1)
        x = np.repeat(signs[:n_blocks], block_size)[:N]
        inits.append(x)
    
    # 4. Smooth random
    for _ in range(2):
        x = rng.normal(0, 1, N)
        inits.append(x)
    
    # 5. Spectral initialization: try to make autoconv small
    # Use random phases in frequency domain
    for _ in range(2):
        freq = rng.uniform(0, 2 * np.pi, N)
        x = np.cos(freq) + 0.1 * rng.normal(0, 1, N)
        inits.append(x)
    
    # Evaluate and optimize each initialization
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 20:
            break
        
        # Quick evaluation
        obj0 = _objective(x0)
        if obj0 < best_obj:
            best_obj = obj0
            best_x = x0.copy()
        
        # Local optimization
        x_opt, obj_opt = _local_search(x0, max_iter=1500, timeout=12.0)
        if obj_opt < best_obj:
            best_obj = obj_opt
            best_x = x_opt.copy()
        
        # Also try sign-flipped version
        x0_flip = -x0
        obj0f = _objective(x0_flip)
        if obj0f < best_obj:
            best_obj = obj0f
            best_x = x0_flip.copy()
    
    # Final polish with more iterations on the best
    if time.time() < deadline - 5:
        x_final, obj_final = _local_search(best_x, max_iter=3000, timeout=30.0)
        if obj_final < best_obj:
            best_obj = obj_final
            best_x = x_final.copy()
    
    # Round to reasonable precision
    best_x = np.round(best_x, 10)
    
    return best_x
# EVOLVE-BLOCK-END
