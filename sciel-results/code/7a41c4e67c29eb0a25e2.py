"""
Program for the suite task: Signed autocorrelation maximum absolute-value variant.

Strategy: Fractional Fourier Transform (FRFT) based design.
The sequence is modeled as the impulse response of a Fractional Fourier Transform
applied to a flat-spectrum random phase vector. The fractional order and initial
phases are optimized to exploit time-frequency rotation properties for peak suppression.

This differs from standard FFT-based spectral design by allowing the sequence to
exist in an intermediate time-frequency domain, potentially finding configurations
that reduce autoconvolution peaks more effectively than direct time-domain optimization.
"""

import numpy as np
import time

N = 400
DEADLINE = 120.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def _autoconv(f):
    """Compute linear autoconvolution of f."""
    return np.convolve(f, f, mode='full')

def _metric(f):
    """
    Compute the score: max_k |C_k| / (sum(f))^2.
    Lower is better.
    """
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    C = _autoconv(f)
    return np.max(np.abs(C)) / (s * s)

def _frft(f, alpha):
    """
    Fractional Fourier Transform of order alpha.
    alpha=0: identity, alpha=1: FFT, alpha=2: flip, alpha=3: IFFT.
    Uses the matrix form for fractional orders.
    """
    n = len(f)
    k = np.arange(n)
    # Build the FRFT matrix
    # For efficiency, we use the decomposition: FRFT = exp(-i*pi*alpha*n) * exp(-i*pi*alpha*k^2/2) * FFT * exp(-i*pi*alpha*k^2/2)
    # This is the "rotated" form
    phase1 = np.exp(-1j * np.pi * alpha * k * k / 2.0)
    phase2 = np.exp(-1j * np.pi * alpha * n / 2.0)
    
    # FFT of (f * phase1)
    F = np.fft.fft(f * phase1)
    result = phase2 * np.fft.ifft(F * phase1)
    return result

def _frft_inv(f, alpha):
    """Inverse Fractional Fourier Transform."""
    return _frft(f, -alpha)

def _random_phases(n, seed=None):
    """Generate random phases in [0, 2*pi)."""
    rng = np.random.RandomState(seed)
    return rng.uniform(0, 2*np.pi, size=n)

def _optimize_phases(phases, alpha, time_budget):
    """
    Optimize the phases using L-BFGS-B to minimize the metric.
    The sequence is f = real(FRFT(exp(i*phases), alpha)).
    """
    from scipy.optimize import minimize
    
    def objective(phases_flat):
        phases = phases_flat
        # Build complex vector with unit magnitude and given phases
        z = np.exp(1j * phases)
        # Apply FRFT
        f_complex = _frft(z, alpha)
        # Take real part to get signed step heights
        f = np.real(f_complex)
        # Normalize to have non-zero sum
        s = np.sum(f)
        if abs(s) < 1e-10:
            return 1e10
        # Scale to unit sum for fair comparison
        f_norm = f / s
        return _metric(f_norm)
    
    # Use L-BFGS-B
    result = minimize(objective, phases, method='L-BFGS-B', 
                      options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6})
    return result.x

def construct():
    """
    Returns the signed step function minimizing the maximum absolute autoconvolution
    divided by squared integral, using FRFT-based design.
    """
    start_time = time.time()
    
    # Start with constant vector (known good baseline)
    best_f = np.ones(N, dtype=float)
    best_score = _metric(best_f)
    
    # Try different fractional orders
    alpha_candidates = [0.25, 0.33, 0.5, 0.66, 0.75, 1.0, 1.25, 1.5, 2.0]
    
    # Multi-start optimization over phases
    num_starts = 5
    rng = np.random.RandomState(42)
    
    for alpha in alpha_candidates:
        if time.time() - start_time > DEADLINE - 30:
            break
        
        for start_idx in range(num_starts):
            if time.time() - start_time > DEADLINE - 30:
                break
            
            # Initialize with random phases
            phases = _random_phases(N, seed=start_idx * 100 + int(alpha * 100))
            
            # Optimize phases
            time_budget = min(10.0, DEADLINE - 30 - (time.time() - start_time))
            if time_budget < 1:
                break
            
            try:
                optimized_phases = _optimize_phases(phases, alpha, time_budget)
                
                # Evaluate the result
                z = np.exp(1j * optimized_phases)
                f_complex = _frft(z, alpha)
                f = np.real(f_complex)
                
                # Check if sum is non-zero
                s = np.sum(f)
                if abs(s) < 1e-10:
                    continue
                
                # Normalize
                f_norm = f / s
                score = _metric(f_norm)
                
                if score < best_score:
                    best_score = score
                    best_f = f_norm.copy()
                    
            except Exception:
                continue
    
    # Final polish: try small perturbations around the best solution
    if time.time() - start_time < DEADLINE - 10:
        # Try gradient-free refinement with random restarts
        for _ in range(10):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Random perturbation
            noise = rng.normal(0, 0.01, size=N)
            f_perturbed = best_f + noise
            s = np.sum(f_perturbed)
            if abs(s) > 1e-10:
                f_perturbed = f_perturbed / s
                score = _metric(f_perturbed)
                if score < best_score:
                    best_score = score
                    best_f = f_perturbed.copy()
    
    return best_f
# EVOLVE-BLOCK-END
