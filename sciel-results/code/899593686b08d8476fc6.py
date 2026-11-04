"""Optimized program for the second autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
_DEADLINE = 115.0  # seconds, safety margin before 120s kill

def _autoconvolution_peak(h):
    """Compute the supremum norm of the autoconvolution of h."""
    # Autoconvolution: (h*h)(t) = integral h(s)h(t-s) ds
    # For discrete step heights on equal intervals, this is a discrete convolution
    # h has N values, so the convolution has 2N-1 values
    conv = np.convolve(h, h, mode='full')
    return np.max(conv)

def _l2_norm_sq(h):
    """Squared L2 norm (sum of squares for equal intervals, interval width cancels in ratio)."""
    return np.sum(h**2)

def _l1_norm(h):
    """L1 norm (sum of values for equal intervals)."""
    return np.sum(h)

def _ratio(h):
    """Compute the ratio: squared L2 norm / (L1 norm * sup norm of autoconvolution)."""
    l2sq = _l2_norm_sq(h)
    l1 = _l1_norm(h)
    peak = _autoconvolution_peak(h)
    if l1 == 0 or peak == 0:
        return 0.0
    return l2sq / (l1 * peak)

def _smooth_max(x, sigma=1.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    if x_max == 0:
        return 0.0
    # Shift for numerical stability
    x_shifted = x - x_max
    log_sum_exp = x_max + np.log(np.sum(np.exp(x_shifted / sigma))) * sigma
    return log_sum_exp

def _objective(params):
    """Objective to minimize: negative of the ratio (we want to maximize ratio)."""
    h = np.maximum(params, 0.0)  # Ensure nonnegativity
    if np.sum(h) == 0:
        return 0.0
    l2sq = np.sum(h**2)
    l1 = np.sum(h)
    conv = np.convolve(h, h, mode='full')
    peak = _smooth_max(conv, sigma=0.01)
    if peak == 0:
        return 0.0
    return -l2sq / (l1 * peak)

def _gradient(params):
    """Numerical gradient for the objective."""
    eps = 1e-7
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        params_minus = params.copy()
        params_minus[i] -= eps
        grad[i] = (_objective(params_plus) - _objective(params_minus)) / (2 * eps)
    return grad

def _optimize_from_start(h0, time_limit):
    """Run L-BFGS-B optimization from a starting point."""
    start_time = time.time()
    
    # Use scipy's L-BFGS-B with bounds
    bounds = [(0, None)] * N
    result = minimize(
        _objective,
        h0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000}
    )
    
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return h0
    
    h_opt = np.maximum(result.x, 0.0)
    return h_opt

def construct():
    """Construct 50 nonnegative heights to maximize the ratio."""
    start_time = time.time()
    
    # Multiple diverse initializations
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    initializations = []
    
    # 1. Uniform
    initializations.append(np.ones(N))
    
    # 2. Gaussian-like
    x = np.linspace(-1, 1, N)
    initializations.append(np.exp(-x**2 / 0.1))
    initializations.append(np.exp(-x**2 / 0.5))
    initializations.append(np.exp(-x**2 / 2.0))
    
    # 3. Bimodal
    initializations.append(np.exp(-((x - 0.4)**2) / 0.1) + np.exp(-((x + 0.4)**2) / 0.1))
    initializations.append(np.exp(-((x - 0.6)**2) / 0.05) + np.exp(-((x + 0.6)**2) / 0.05))
    
    # 4. Triangle / tent
    initializations.append(1 - np.abs(x))
    
    # 5. Cosine window
    initializations.append(0.5 + 0.5 * np.cos(np.pi * x))
    
    # 6. Random smooth
    rng = np.random.RandomState(42)
    for seed in [1, 2, 3, 4, 5]:
        rng_local = np.random.RandomState(seed)
        h_rand = np.abs(rng_local.randn(N))
        # Smooth it
        kernel = np.ones(5) / 5
        h_smooth = np.convolve(h_rand, kernel, mode='same')
        initializations.append(h_smooth)
    
    # 7. Peak in center
    h_center = np.zeros(N)
    h_center[N//2 - 2:N//2 + 3] = 1.0
    initializations.append(h_center)
    
    # 8. Peak at edges
    h_edge = np.zeros(N)
    h_edge[0:3] = 1.0
    h_edge[-3:] = 1.0
    initializations.append(h_edge)
    
    # 9. Alternating
    h_alt = np.ones(N)
    h_alt[::2] = 0.5
    initializations.append(h_alt)
    
    # 10. Linear ramp
    initializations.append(np.linspace(0.1, 1.0, N))
    initializations.append(np.linspace(1.0, 0.1, N))
    
    # 11. DFT-based: squared magnitude of DFT of a random complex vector
    for seed in [10, 20, 30]:
        rng_local = np.random.RandomState(seed)
        z = rng_local.randn(N) + 1j * rng_local.randn(N)
        H = np.fft.fft(z)
        h_dft = np.abs(H)**2
        h_dft = h_dft / np.max(h_dft)
        initializations.append(h_dft)
    
    # 12. More structured patterns
    initializations.append(np.sin(np.pi * (x + 1) / 2)**2)
    initializations.append(np.cos(np.pi * x / 2)**2)
    
    # Optimize from each initialization
    time_per_opt = 8.0  # seconds per optimization attempt
    
    for i, h0 in enumerate(initializations):
        elapsed = time.time() - start_time
        if elapsed + time_per_opt > _DEADLINE:
            break
        
        # Normalize for better conditioning
        h0_norm = h0.copy()
        if np.sum(h0_norm) > 0:
            h0_norm = h0_norm / np.sum(h0_norm) * N  # Keep sum roughly constant
        
        h_opt = _optimize_from_start(h0_norm, time_per_opt)
        
        score = _ratio(h_opt)
        if score > best_score:
            best_score = score
            best_h = h_opt
    
    # Final polish: run a few more iterations from the best solution
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 10:
        # Try small perturbations around the best
        for pert_seed in [100, 200, 300]:
            elapsed = time.time() - start_time
            if elapsed + 5 > _DEADLINE:
                break
            rng_local = np.random.RandomState(pert_seed)
            h_pert = best_h + rng_local.randn(N) * 0.05
            h_pert = np.maximum(h_pert, 0)
            h_opt = _optimize_from_start(h_pert, 5.0)
            score = _ratio(h_opt)
            if score > best_score:
                best_score = score
                best_h = h_opt
    
    return list(best_h)
