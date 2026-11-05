"""
Strategy: Direct DFT Coefficient Optimization with Smooth Max Surrogate.

Instead of parameterizing h in the time domain via low-frequency Fourier modes,
we parameterize h directly by its DFT coefficients. Since the DFT is a linear
transform, this is equivalent, but we can exploit the fact that the autocorrelation
is related to the power spectrum via the Wiener-Khinchin theorem. Specifically,
the autocorrelation r[k] = sum h[j] h[j+k] has DFT |H[m]|^2.

However, the objective involves max over lag, which is a non-smooth operation.
We use a smooth max (soft-max) surrogate to enable gradient-based optimization.
We optimize the real and imaginary parts of the DFT coefficients directly.

Key insight: By working in the frequency domain, we can directly control the
power spectrum, which determines the autocorrelation. We aim to flatten the
power spectrum to minimize peak autocorrelation.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400
_DEADLINE = 145.0  # Safety margin before 150s limit

def _value(h):
    """Compute the task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _smooth_max(x, beta=100.0):
    """Smooth approximation of max(x)."""
    return (1.0 / beta) * np.log(np.sum(np.exp(beta * x)))

def _smooth_value_from_dft(dft_coeffs):
    """
    Compute smooth surrogate of the value from DFT coefficients.
    dft_coeffs is a complex array of length N.
    """
    # Reconstruct h from DFT
    h = np.real(np.fft.ifft(dft_coeffs))
    s = np.sum(h)
    if abs(s) < 1e-8 or not np.isfinite(s):
        return 1e10
    
    # Compute autocorrelation via FFT
    # Full autocorrelation: convolve h with reversed h
    # But we can use FFT: autocorr = ifft(fft(h) * conj(fft(h)))
    H = np.fft.fft(h)
    power = np.abs(H)**2
    autocorr = np.real(np.fft.ifft(power))
    
    # The autocorrelation from ifft is circular. We need linear.
    # For large N, circular and linear are similar for small lags.
    # But to be exact, let's compute linear autocorrelation.
    # Actually, let's just use the circular one for speed in the surrogate,
    # and verify with linear at the end.
    
    # Normalize
    autocorr_norm = autocorr / (s * s)
    
    # The value is 2*N*max(autocorr)/(s^2)
    # So we want to minimize max(autocorr)
    smax = _smooth_max(autocorr_norm, beta=50.0)
    return 2.0 * N * smax

def _dft_obj(dft_real, dft_imag):
    """Objective function for optimizer, taking real and imag parts separately."""
    dft_coeffs = dft_real + 1j * dft_imag
    return _smooth_value_from_dft(dft_coeffs)

def construct():
    """
    Construct step heights using DFT coefficient optimization.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    best_val = _value(np.ones(N))
    best_h = np.ones(N)
    
    rng = np.random.default_rng(123)
    
    # We'll optimize the DFT coefficients.
    # Fix DC component (index 0) to be real and positive to fix scale.
    # Actually, let's just optimize all and let the optimizer find the scale.
    # But to avoid the zero solution, we'll add a penalty or fix DC.
    
    # Let's fix the DC component to N (so sum(h) = N initially)
    # and optimize the other components.
    
    n_restarts = 15
    
    for i in range(n_restarts):
        if time.time() > deadline - 15:
            break
        
        # Initialize DFT coefficients
        dft_coeffs = np.zeros(N, dtype=complex)
        dft_coeffs[0] = N  # DC component
        
        # Add random small perturbations to other frequencies
        # Use a mix of low and mid frequencies
        for m in range(1, N):
            # Skip the Nyquist frequency for real signals
            if m == N // 2:
                continue
            # Random amplitude and phase
            amp = rng.uniform(0, 1.0)
            if i % 2 == 0:
                # Focused on low frequencies
                amp *= np.exp(-m / 50.0)
            else:
                # Broader spectrum
                amp *= np.exp(-m / 200.0)
            phase = rng.uniform(0, 2 * np.pi)
            dft_coeffs[m] = amp * np.exp(1j * phase)
            # Ensure conjugate symmetry for real h
            dft_coeffs[N - m] = np.conj(dft_coeffs[m])
        
        # Extract real and imaginary parts for optimization
        dft_real = dft_coeffs.real.copy()
        dft_imag = dft_coeffs.imag.copy()
        
        # Fix DC component
        dft_real[0] = N
        dft_imag[0] = 0.0
        
        # Define bounds: allow coefficients to vary, but keep DC fixed
        # We'll use L-BFGS-B with bounds
        bounds = []
        for m in range(N):
            if m == 0:
                bounds.append((N, N))  # Fix DC real
                bounds.append((0, 0))  # Fix DC imag
            else:
                bounds.append((-5.0, 5.0))
                bounds.append((-5.0, 5.0))
        
        x0 = np.concatenate([dft_real, dft_imag])
        
        try:
            res = minimize(
                _dft_obj,
                x0,
                args=(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            if res.fun < best_val:
                dft_opt_real = res.x[:N]
                dft_opt_imag = res.x[N:]
                dft_opt = dft_opt_real + 1j * dft_opt_imag
                h_opt = np.real(np.fft.ifft(dft_opt))
                
                if np.all(np.isfinite(h_opt)) and np.sum(h_opt) != 0:
                    v_opt = _value(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
                        
        except Exception:
            pass
    
    # Local refinement in time domain
    if time.time() < deadline - 5:
        h_curr = best_h.copy()
        v_curr = best_val
        step_size = 0.05
        for _ in range(3000):
            if time.time() > deadline - 2:
                break
            idx = rng.integers(0, N)
            h_new = h_curr.copy()
            h_new[idx] += rng.normal(0, step_size)
            v_new = _value(h_new)
            if v_new < v_curr:
                v_curr = v_new
                h_curr = h_new
                step_size *= 0.95
            else:
                step_size = min(step_size * 1.1, 0.2)
        
        if v_curr < best_val:
            best_val = v_curr
            best_h = h_curr
    
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END
