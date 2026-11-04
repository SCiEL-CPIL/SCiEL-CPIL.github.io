"""
Spectral Design via Chirp-Phase Parameterization.

Strategy:
1. Instead of optimizing 400 independent phase angles (high dimensionality, many local minima),
   we parameterize the phase spectrum using a low-dimensional family of "chirp" phases.
   A chirp phase phi_k = a * k^2 + b * k + c (mod 2pi) corresponds to a quadratic phase
   in the frequency domain, which produces a signal with specific autoconvolution properties.
   More generally, we use a small number of basis functions (e.g., k^2, k^3, sin(k), cos(k))
   to span a low-dimensional manifold of phase functions.
2. We optimize the coefficients of these basis functions using differential evolution
   (a global optimization method) or simple random search + local refinement.
3. The key insight from the knowledge cell: "Multi-sine and spectral-domain parameterizations
   reduce the effective dimensionality from 400 to ~10-16, enabling global optimization to escape
   local minima that trap the 400-dim L-BFGS approach."
4. We also include a "flat magnitude" design where we only optimize the phase, but restrict it
   to this low-dimensional manifold.

This is a STRUCTURAL change: different representation (low-dim manifold vs full 400-dim),
different optimizer (global/random search vs SGD), and different design family (chirp-like
phases vs arbitrary phases).
"""

import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 400

def _autoconv_lin(x):
    """Compute linear autoconvolution using FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]

def _metric_offdiag(x):
    """Metric: max |c_k| for k != 0 / S^2"""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    if len(c) > 1:
        return float(np.max(np.abs(c[1:]))) / (s * s)
    return 0.0

def _phase_to_signal(phi, mag):
    """Convert phase vector and fixed magnitude to time-domain signal."""
    X = mag * np.exp(1j * phi)
    x = np.fft.ifft(X).real
    return x

def _chirp_phase_basis(k, coeffs):
    """
    Generate a phase vector from a low-dimensional set of coefficients.
    coeffs[0] = constant phase
    coeffs[1] = linear phase (frequency shift)
    coeffs[2] = quadratic phase (chirp)
    coeffs[3] = cubic phase
    coeffs[4] = sin(k) component
    coeffs[5] = cos(k) component
    coeffs[6] = sin(2k) component
    coeffs[7] = cos(2k) component
    """
    phi = np.zeros(len(k))
    if len(coeffs) > 0:
        phi += coeffs[0]
    if len(coeffs) > 1:
        phi += coeffs[1] * k
    if len(coeffs) > 2:
        phi += coeffs[2] * k**2
    if len(coeffs) > 3:
        phi += coeffs[3] * k**3
    if len(coeffs) > 4:
        phi += coeffs[4] * np.sin(k)
    if len(coeffs) > 5:
        phi += coeffs[5] * np.cos(k)
    if len(coeffs) > 6:
        phi += coeffs[6] * np.sin(2 * k)
    if len(coeffs) > 7:
        phi += coeffs[7] * np.cos(2 * k)
    return phi

def _objective_from_coeffs(coeffs, mag):
    """Compute the metric from coefficient vector."""
    k = np.arange(N)
    phi = _chirp_phase_basis(k, coeffs)
    phi[0] = 0.0  # Fix DC phase
    x = _phase_to_signal(phi, mag)
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    return _metric_offdiag(x)

def _optimize_chirp(mag, time_budget=100.0, n_dims=8):
    """
    Optimize the chirp phase coefficients using differential evolution
    followed by local refinement.
    """
    start = time.time()
    
    # Define bounds for coefficients
    # Quadratic phase: k^2 * coeffs[2]. For k up to N, k^2 is up to N^2.
    # We want phi to stay within reasonable range, say [-10*pi, 10*pi].
    # So coeffs[2] should be in [-10*pi/N^2, 10*pi/N^2] roughly.
    # But let's be more generous and let the optimizer figure it out.
    # Actually, phase is modulo 2pi, so we can wrap it. But for smoothness,
    # let's bound the coefficients.
    
    # For k in [0, N-1]:
    # k^2 max is (N-1)^2 ~ N^2
    # We want coeffs[2] * N^2 to be in [-pi, pi] roughly for one "chirp" cycle.
    # So coeffs[2] in [-pi/N^2, pi/N^2] gives one cycle.
    # Let's allow several cycles: [-5*pi/N^2, 5*pi/N^2]
    
    bounds = [
        (-np.pi, np.pi),           # constant
        (-np.pi, np.pi),           # linear
        (-10 * np.pi / N, 10 * np.pi / N),  # quadratic (scaled)
        (-10 * np.pi / N**2, 10 * np.pi / N**2),  # cubic (scaled)
        (-np.pi, np.pi),           # sin(k)
        (-np.pi, np.pi),           # cos(k)
        (-np.pi, np.pi),           # sin(2k)
        (-np.pi, np.pi),           # cos(2k)
    ]
    
    # Truncate to n_dims
    bounds = bounds[:n_dims]
    
    # Use differential evolution for global search
    try:
        result = differential_evolution(
            _objective_from_coeffs,
            bounds=bounds,
            args=(mag,),
            maxiter=50,
            popsize=15,
            tol=1e-4,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            polish=False,  # We'll do our own polish
            time_budget=time_budget * 0.6,
            init='latinhypercube'
        )
        best_coeffs = result.x
        best_score = result.fun
    except Exception:
        best_coeffs = np.zeros(n_dims)
        best_score = _objective_from_coeffs(best_coeffs, mag)
    
    # Local refinement using Nelder-Mead
    if time.time() - start < time_budget * 0.9:
        try:
            result2 = minimize(
                _objective_from_coeffs,
                best_coeffs,
                args=(mag,),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if result2.fun < best_score:
                best_coeffs = result2.x
                best_score = result2.fun
        except Exception:
            pass
    
    return best_coeffs, best_score

def construct():
    """
    Construct a signed step function minimizing max|autoconv| / (sum(x))^2.
    Uses low-dimensional chirp-phase parameterization with global optimization.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_x = None
    best_score = float('inf')
    
    # Fixed flat magnitude spectrum
    mag = np.ones(N)
    mag[0] = N  # Ensure non-zero integral
    
    # Try multiple random initializations and dimensions
    configs = [
        (8, 42),    # 8 dims, seed 42
        (8, 123),   # 8 dims, seed 123
        (6, 999),   # 6 dims, seed 999
        (5, 314),   # 5 dims, seed 314
        (4, 271),   # 4 dims, seed 271
    ]
    
    for n_dims, seed in configs:
        if time.time() > deadline - 15.0:
            break
        
        # Set random seed for reproducibility in this config
        np.random.seed(seed)
        
        try:
            coeffs, score = _optimize_chirp(mag, time_budget=25.0, n_dims=n_dims)
            
            k = np.arange(N)
            phi = _chirp_phase_basis(k, coeffs)
            phi[0] = 0.0
            x = _phase_to_signal(phi, mag)
            
            if abs(np.sum(x)) > 1e-10:
                sc = _metric_offdiag(x)
                if sc < best_score:
                    best_score = sc
                    best_x = x.copy()
        except Exception:
            pass
    
    # Also try a pure random phase search as fallback
    if time.time() < deadline - 10.0:
        for trial in range(20):
            if time.time() > deadline - 5.0:
                break
            rng = np.random.default_rng(trial * 100 + 7)
            phi = rng.uniform(-np.pi, np.pi, N)
            phi[0] = 0.0
            x = _phase_to_signal(phi, mag)
            if abs(np.sum(x)) > 1e-10:
                sc = _metric_offdiag(x)
                if sc < best_score:
                    best_score = sc
                    best_x = x.copy()
    
    # Final polish: Try thresholding to ±1
    if best_x is not None:
        x_thresh = np.sign(best_x)
        if np.all(x_thresh == 1):
            x_thresh[::2] = -1
        elif np.all(x_thresh == -1):
            x_thresh[::2] = 1
        
        s_thresh = np.sum(x_thresh)
        if abs(s_thresh) > 1e-10:
            score_thresh = _metric_offdiag(x_thresh)
            if score_thresh < best_score:
                best_score = score_thresh
                best_x = x_thresh.copy()
    
    if best_x is None:
        return np.ones(N)
    
    return best_x

if __name__ == "__main__":
    x = construct()
    print("Score:", _metric_offdiag(x))
