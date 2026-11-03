"""Rewritten program: spectral projected gradient descent for nonnegative step functions."""
import numpy as np
import time

N = 600

def _autoconv_peak(f):
    # f is array of length N; return max of circular autoconvolution
    F = np.fft.fft(f)
    conv = np.fft.ifft(F * F).real
    return np.max(conv)

def _score(f):
    # normalized maximum of autoconvolution
    return _autoconv_peak(f) / (np.sum(f) ** 2)

def construct():
    deadline = time.time() + 140.0
    # Start with a box
    f = np.ones(N)
    best = f.copy()
    best_score = _score(best)
    
    # Spectral representation: optimize real FFT coefficients
    # We want to minimize max(f*f) subject to f >= 0.
    # Use projected gradient descent in frequency domain.
    # Actually, the constraint f >= 0 is easier in time domain.
    # Let's do alternating: update in frequency, project in time.
    
    # Gradient of max(f*f) wrt f is 2*(f*f) at the peak index, 0 elsewhere.
    # But we want to minimize the peak, which is a min-max problem.
    # We can use the subgradient: the gradient of the peak constraint.
    
    # Let's use a simpler approach: minimize the L-infinity norm of f*f
    # via projected gradient descent on f directly, with a smoothing.
    
    # Actually, the knowledge says: "Spectral parameterization with coordinate descent...
    # Measured: 1 -> 1.155"
    # Let's try to optimize the FFT coefficients directly.
    
    # Let's use the fact that for a nonnegative function, the DC component is fixed
    # if we normalize the integral. But we can scale f arbitrarily (score is scale-invariant).
    
    # Let's try: start with a box, then apply small random perturbations in frequency,
    # project to nonnegativity, keep if score improves.
    
    rng = np.random.default_rng(42)
    
    for iter in range(2000):
        if time.time() > deadline:
            break
        
        # Random direction in frequency space
        # We'll perturb the FFT coefficients
        F = np.fft.fft(f)
        # Random real perturbation to F, keeping conjugate symmetry
        dF = rng.normal(0, 0.1, N)
        dF = (dF + np.conj(dF[::-1])) / 2  # ensure real result
        dF[0] = 0  # keep DC fixed
        
        # Try step
        F_new = F + dF
        f_new = np.fft.ifft(F_new).real
        f_new = np.maximum(f_new, 0)  # project to nonnegativity
        
        if np.sum(f_new) == 0:
            continue
        
        s = _score(f_new)
        if s < best_score:
            best_score = s
            best = f_new
            f = f_new
    
    # Final projection
    best = np.maximum(best, 0)
    if np.sum(best) == 0:
        return [1.0] * N
    
    # Normalize to have sum 1 (score is scale-invariant, but let's be safe)
    best = best / np.sum(best)
    
    return list(best)
