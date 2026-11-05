"""
Structural change: Parametric frequency-domain design.
Models step heights as a band-limited signal with a tunable DC null.
Optimizes low-dimensional spectral parameters (cutoff, roll-off) instead of 400 heights.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    """
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    # Pre-compute FFT size for convolution
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2

    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return float('inf')
        H = np.fft.rfft(h, n=fft_size)
        conv = np.fft.irfft(H * H, n=fft_size)
        max_c = np.max(conv)
        val = abs(2.0 * N * max_c / (s * s))
        return val

    # Parametric model:
    # h is the real part of an IFFT of a spectrum H(k).
    # H(k) is designed to be 0 near k=0 (DC) to suppress the main lobe of autocorrelation/convolution,
    # and 1 elsewhere (flat spectrum) to minimize side lobes.
    # Parameters:
    #   p[0]: cutoff_freq (number of frequency bins to null out, symmetric)
    #   p[1]: roll_off (sharpness of the transition, >0 means smoother, <1 means sharper)
    #   p[2]: phase_shift (optional, to break symmetry if needed, though real h is symmetric spectrum)
    
    # We optimize over a small vector of parameters.
    # Let's define the spectrum mask directly in the frequency domain.
    # k = 0, ..., N-1.
    # We want H(k) = 0 for small |k| and H(k) = 1 for large |k|.
    
    def h_from_params(params):
        cutoff = params[0]
        roll_off = params[1]
        
        # Frequency indices
        k = np.arange(N)
        
        # Create mask
        # We treat k and N-k as symmetric.
        # Distance from DC (0) and Nyquist (N/2)
        # For real signals, H(k) = H(N-k).
        # We want to null out the center.
        
        # Simple approach: Gaussian or Hyperbolic Secant roll-off
        # mask[k] = 1 / (1 + exp(-roll_off * (dist(k) - cutoff)))
        # dist(k) is distance from 0 (or N).
        
        # Let's use a simpler polynomial or linear roll-off for stability
        # mask = 0 if dist < cutoff_inner
        # mask = 1 if dist > cutoff_outer
        # linear interpolation in between
        
        inner = max(0, cutoff - 2.0)
        outer = max(inner + 0.1, cutoff + 2.0)
        
        # Calculate distance from 0 in circular frequency domain
        # k is 0..N-1. Distance is min(k, N-k)
        dist = np.minimum(k, N - k)
        
        mask = np.zeros(N)
        # Where dist >= outer, mask = 1
        mask[dist >= outer] = 1.0
        # Where dist < inner, mask = 0
        # In between, linear ramp
        ramp_mask = (dist >= inner) & (dist < outer)
        if np.any(ramp_mask):
            mask[ramp_mask] = (dist[ramp_mask] - inner) / (outer - inner)
        
        # Apply mask to a base spectrum.
        # Base spectrum: We can use a flat spectrum (all ones) which gives a Dirac delta in time (not good for finite N) 
        # or a specific structure.
        # Actually, if we just take the IFFT of the mask, we get a sinc-like function (Bartlett/Hamming window in time).
        # Let's try IFFT of the mask.
        
        H_spec = mask.astype(complex)
        # Ensure conjugate symmetry for real output? 
        # The mask is symmetric: mask[k] = mask[N-k]. 
        # So IFFT will be real.
        h = np.real(np.fft.ifft(H_spec))
        
        # Normalize to avoid numerical issues, though the objective is scale-invariant 
        # except for the sum(h)^2 term.
        # If sum(h) is close to 0, value is inf. 
        # The IFFT of a mask that is 0 at DC has sum(h) = H(0) = 0. 
        # Wait, if H(0)=0, then sum(h) = 0. 
        # The objective has sum(h)^2 in denominator. 
        # We MUST have non-zero sum.
        # The task says "np.sum(h) must not be zero".
        # If we null DC completely, sum(h)=0. 
        # So we cannot null DC completely. We must leave a small DC component.
        
        # Let's modify the mask to have a small non-zero DC value.
        # Or, add a small constant to h.
        # Better: Design the mask such that H(0) is small but non-zero.
        
        # Let's redefine the mask to have H(0) = epsilon.
        # We can just add epsilon to the mask at k=0.
        H_spec[0] += 1e-3  # Small DC component
        
        h = np.real(np.fft.ifft(H_spec))
        
        return h

    def objective(params):
        h = h_from_params(params)
        if not np.all(np.isfinite(h)):
            return 1e10
        val = evaluate(h)
        if not np.isfinite(val):
            return 1e10
        return val

    # Initial guess
    # We want to suppress low frequencies. 
    # N=400. 
    # Let's try cutoff around 10-20 bins.
    initial_params = np.array([10.0, 1.0])
    
    best_value = float('inf')
    best_h = np.ones(N)
    
    # Evaluate seed
    seed_val = evaluate(best_h)
    best_value = seed_val

    # Optimization
    # Use Nelder-Mead as it doesn't require gradients and is robust for low-dim
    try:
        result = minimize(
            objective,
            initial_params,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5, 'disp': False}
        )
        
        if elapsed() > DEADLINE:
            return best_h
            
        h_opt = h_from_params(result.x)
        val_opt = evaluate(h_opt)
        
        if val_opt < best_value:
            best_value = val_opt
            best_h = h_opt
            
    except Exception as e:
        pass

    # Try a few more random initializations if time permits
    rng = np.random.default_rng(42)
    for _ in range(5):
        if elapsed() > DEADLINE:
            break
            
        # Random cutoff between 5 and 30
        p0 = rng.uniform(5, 30)
        p1 = rng.uniform(0.5, 2.0)
        init_p = np.array([p0, p1])
        
        try:
            result = minimize(
                objective,
                init_p,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False}
            )
            
            if elapsed() > DEADLINE:
                break
                
            h_opt = h_from_params(result.x)
            val_opt = evaluate(h_opt)
            
            if val_opt < best_value:
                best_value = val_opt
                best_h = h_opt
                
        except Exception:
            continue

    return best_h
