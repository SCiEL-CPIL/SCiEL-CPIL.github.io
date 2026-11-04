"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

def _autoconv_max_ratio(x):
    """Compute max|autoconvolution| / (sum x)^2 for a real signal x."""
    # Ensure real
    x = np.real(x)
    I = np.sum(x)
    if abs(I) < 1e-12:
        return 1e10
    # Full autoconvolution via FFT
    n = len(x)
    size = 2 * n
    f = np.fft.fft(x, size)
    ac = np.fft.ifft(f * f, size)
    ac = np.real(ac[:n])  # linear autoconvolution (lags 0..n-1)
    # Also need negative lags for full autocorrelation
    # Full autoconv: lags -(n-1)..(n-1)
    ac_full = np.fft.ifft(f * f, size)
    ac_full = np.real(ac_full)
    # ac_full[0..n-1] are lags 0..n-1, ac_full[n..2n-1] are lags -(n-1)..-1
    max_abs = np.max(np.abs(ac_full))
    return max_abs / (I * I)

def _from_fourier_coeffs(coeffs):
    """Convert Fourier coefficients to time-domain signal of length N."""
    # coeffs: real array of length M, representing symmetric spectrum
    # We'll construct a real signal whose FFT has these coefficients
    M = len(coeffs)
    # Build full FFT spectrum
    spec = np.zeros(N, dtype=complex)
    # Map M coefficients to first M frequency bins (and mirror for real signal)
    for i in range(M):
        spec[i] = coeffs[i]
        if i < N - i and i > 0:
            spec[N - i] = coeffs[i]  # conjugate symmetry for real signal
    # For DC and Nyquist, no mirror needed
    x = np.fft.ifft(spec)
    return x

def _score_from_coeffs(coeffs):
    """Objective: minimize max|autoconv| / I^2 from Fourier coefficients."""
    x = _from_fourier_coeffs(coeffs)
    return _autoconv_max_ratio(x)

def construct():
    global _START_TIME
    elapsed = time.time() - _START_TIME
    remaining = _DEADLINE - elapsed
    
    # Use spectral parameterization with ~31 coefficients
    # This reduces dimensionality from 400 to ~31
    M = 31  # number of Fourier coefficients to optimize
    
    # Bounds for each coefficient
    bounds = [(-50.0, 50.0)] * M
    
    # Objective function
    def objective(coeffs):
        return _score_from_coeffs(coeffs)
    
    # Initial best: all ones in time domain
    x_ones = np.ones(N)
    best_score = _autoconv_max_ratio(x_ones)
    best_x = x_ones.copy()
    
    # Try differential evolution with time budget
    if remaining > 10.0:
        try:
            result = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=200,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=False,
                init='latinhypercube'
            )
            if result.fun < best_score:
                best_score = result.fun
                best_x = _from_fourier_coeffs(result.x)
        except Exception:
            pass
    
    # Check time
    if time.time() - _START_TIME > _DEADLINE - 5.0:
        return np.real(best_x)
    
    # Local refinement with Nelder-Mead on the Fourier coefficients
    if remaining > 5.0:
        try:
            from scipy.optimize import minimize
            # Start from best Fourier coefficients found
            # Reconstruct: we need the coefficients that gave best_x
            # For now, try a few random starts in Fourier space
            for seed_try in range(5):
                if time.time() - _START_TIME > _DEADLINE - 3.0:
                    break
                rng = np.random.default_rng(seed_try + 100)
                x0 = rng.uniform(-20, 20, size=M)
                res = minimize(
                    objective,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.fun < best_score:
                    best_score = res.fun
                    best_x = _from_fourier_coeffs(res.x)
        except Exception:
            pass
    
    return np.real(best_x)
# EVOLVE-BLOCK-END
