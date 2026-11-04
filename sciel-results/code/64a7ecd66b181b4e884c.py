"""Optimized program for second autocorrelation inequality."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_ratio(heights):
    """Compute the ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    
    where h*h is the discrete autoconvolution.
    """
    h = np.asarray(heights, dtype=float)
    if np.any(h < 0):
        h = np.clip(h, 0, None)
    
    l2_sq = np.sum(h ** 2)
    l1 = np.sum(h)
    
    if l1 < 1e-15:
        return 0.0
    
    # Discrete autoconvolution: (h*h)[k] = sum_i h[i]*h[i+k]
    # Using FFT for efficiency
    n_fft = 2 * N
    H = np.fft.fft(h, n=n_fft)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    autocorr = autocorr[:N]  # Take the first N values (non-negative lags)
    
    sup_norm = np.max(autocorr)
    
    if sup_norm < 1e-15:
        return 0.0
    
    return l2_sq / (l1 * sup_norm)


def _optimize_from_initial(h0, deadline, best_state):
    """Run gradient-based optimization from initial point h0."""
    n = len(h0)
    
    def objective(x):
        """Minimize negative of the ratio."""
        # Enforce non-negativity via softplus-like transformation
        # x -> exp(x) gives positive values, but we want to allow zero
        # Use: h = np.maximum(x, 0) for simplicity, or a smooth version
        h = np.maximum(x, 0.0)
        if np.sum(h) < 1e-15:
            return 0.0
        return -_compute_ratio(h)
    
    def gradient(x):
        """Numerical gradient of the objective."""
        eps = 1e-7
        grad = np.zeros_like(x)
        base = objective(x)
        for i in range(n):
            xp = x.copy()
            xp[i] += eps
            xm = x.copy()
            xm[i] -= eps
            grad[i] = (objective(xp) - objective(xm)) / (2 * eps)
        return grad
    
    # Scale initial point to reasonable magnitude
    h0_scaled = h0.copy()
    if np.max(h0_scaled) > 0:
        h0_scaled = h0_scaled / np.max(h0_scaled) * 1.0
    
    result = minimize(
        objective,
        h0_scaled,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
    )
    
    h_opt = np.maximum(result.x, 0.0)
    ratio = _compute_ratio(h_opt)
    
    if ratio > best_state[0]:
        best_state[0] = ratio
        best_state[1] = h_opt.copy()
    
    return best_state


def construct():
    """Construct 50 nonnegative heights to maximize the ratio."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_ratio = 0.0
    best_heights = np.ones(N)
    
    # Initial guess: uniform
    uniform = np.ones(N)
    best_ratio = _compute_ratio(uniform)
    best_heights = uniform.copy()
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Uniform (already done)
    initializations.append(uniform.copy())
    
    # 2. Gaussian centered
    x = np.linspace(-1, 1, N)
    gauss = np.exp(-x**2 * 2)
    initializations.append(gauss)
    
    # 3. Lorentzian
    lorentz = 1.0 / (1.0 + x**2 * 4)
    initializations.append(lorentz)
    
    # 4. Power law (edge-enhanced)
    power = np.abs(x) ** (-0.5)
    power = power / np.max(power)
    initializations.append(power)
    
    # 5. Two peaks
    two_peak = np.exp(-(x - 0.3)**2 * 8) + np.exp(-(x + 0.3)**2 * 8)
    initializations.append(two_peak)
    
    # 6. Flat top with raised edges
    flat_top = np.ones(N)
    flat_top[:5] = 1.5
    flat_top[-5:] = 1.5
    initializations.append(flat_top)
    
    # 7. Cosine bump
    cosine_bump = np.cos(x * np.pi / 2) ** 2
    initializations.append(cosine_bump)
    
    # 8. Random smooth (via random Fourier coefficients)
    rng = np.random.default_rng(42)
    for seed in [42, 123, 777]:
        rng_s = np.random.default_rng(seed)
        coeffs = rng_s.normal(size=N//2 + 1) + 1j * rng_s.normal(size=N//2 + 1)
        # Build a smooth nonnegative function via Fourier series
        smooth = np.zeros(N)
        for k in range(len(coeffs)):
            smooth += coeffs[k] * np.exp(2j * np.pi * k * x / N)
        smooth = np.abs(smooth)
        if np.max(smooth) > 0:
            smooth = smooth / np.max(smooth)
        initializations.append(smooth)
    
    # Optimize from each initialization
    best_state = [best_ratio, best_heights]
    
    for i, h0 in enumerate(initializations):
        if time.time() > deadline:
            break
        _optimize_from_initial(h0, deadline, best_state)
    
    # Additional local refinement: perturb the best solution and re-optimize
    if time.time() < deadline - 5:
        rng = np.random.default_rng(2024)
        for _ in range(5):
            if time.time() > deadline:
                break
            perturbed = best_state[1] + rng.normal(0, 0.1, size=N)
            perturbed = np.maximum(perturbed, 0)
            _optimize_from_initial(perturbed, deadline, best_state)
    
    # Final: return the best heights found
    return list(best_state[1])
