"""
Program to minimize the autocorrelation inequality value using Fourier parameterization.

Strategy:
The objective function involves the maximum of the autoconvolution.
The autoconvolution is the inverse FFT of the squared magnitude of the FFT of h.
If we restrict h to be a linear combination of a few low-frequency sinusoids (and DC),
we can search over the coefficients of these sinusoids.
This reduces the dimensionality from 400 to a small number (e.g., 10-20),
making it feasible to find a global minimum using stochastic global optimizers
like Differential Evolution, which is robust to the non-convexity of the max operation.

The seed np.ones(400) corresponds to only the DC term being non-zero.
By introducing low-frequency oscillations, we can "spread out" the energy in the
time domain, potentially reducing the peak-to-average ratio of the autoconvolution,
thereby lowering the value.
"""

import numpy as np
import time
from scipy.optimize import differential_evolution

N = 400

def _evaluate(h):
    """Compute the objective value."""
    # h must be 1D array of length N
    s = np.sum(h)
    if s == 0:
        return 1e10
    # Full autoconvolution
    # Note: np.convolve(h, h, mode='full') has length 2N-1
    # The center is at index N-1
    conv = np.convolve(h, h, mode='full')
    max_val = np.max(conv)
    return abs(2 * N * max_val / (s * s))

def _h_from_coeffs(coeffs):
    """
    Reconstruct h from Fourier coefficients.
    coeffs: array of shape (K,)
    We use real-valued basis functions:
    k=0: DC term (1)
    k=1: cos(2*pi*1*n/N), sin(2*pi*1*n/N) -> 2 coeffs
    k=2: cos(2*pi*2*n/N), sin(2*pi*2*n/N) -> 2 coeffs
    ...
    So total coeffs = 1 + 2*(K-1) if we index coeffs[0] as DC, coeffs[1:3] as freq 1, etc.
    Let's define coeffs such that:
    coeffs[0] = A0 (DC)
    coeffs[1] = A1 (cos freq 1)
    coeffs[2] = B1 (sin freq 1)
    coeffs[3] = A2 (cos freq 2)
    coeffs[4] = B2 (sin freq 2)
    ...
    """
    n = np.arange(N)
    h = np.zeros(N)
    
    # DC term
    if len(coeffs) > 0:
        h += coeffs[0]
        
    # Frequency terms
    # Number of frequency pairs available
    num_pairs = (len(coeffs) - 1) // 2
    for i in range(num_pairs):
        freq = i + 1
        a = coeffs[1 + 2*i]
        b = coeffs[2 + 2*i]
        h += a * np.cos(2 * np.pi * freq * n / N)
        h += b * np.sin(2 * np.pi * freq * n / N)
        
    return h

def _objective(coeffs):
    """
    Wrapper for optimizer.
    coeffs is a 1D array.
    """
    h = _h_from_coeffs(coeffs)
    return _evaluate(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial best is the seed
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    # We will search over the first few low-frequency components.
    # Let's try up to 5 frequency pairs (freq 1 to 5) + DC = 11 variables.
    # This is a small enough space for DE to work well within the time limit.
    K_freqs = 5
    dim = 1 + 2 * K_freqs
    
    # Bounds for coefficients
    # DC term should be positive to keep sum(h) > 0 (or we handle sign in objective)
    # Actually, _evaluate uses sum(h)^2, so sign doesn't matter for the denominator.
    # But we want to avoid sum(h)=0.
    # Let's bound coefficients in [-10, 10].
    bounds = [(-10, 10)] * dim
    
    # Initial population: include the seed (all zeros except DC=1)
    init = np.zeros((10, dim))
    init[0, 0] = 1.0 # Seed
    
    # Use differential_evolution
    # maxiter and popsize are tuned to fit within time budget.
    # DE is stochastic, so we might need multiple restarts if time permits.
    
    try:
        # Run DE
        # seed=42 for reproducibility if needed, but random is fine
        res = differential_evolution(
            _objective, 
            bounds, 
            maxiter=200, 
            popsize=20, 
            tol=1e-4, 
            init='latinhypercube',
            seed=42,
            polish=True
        )
        
        if res.success or res.fun < best_val:
            # Check time
            if time.time() < deadline:
                best_h = _h_from_coeffs(res.x)
                best_val = _evaluate(best_h)
                
    except Exception as e:
        # If DE fails, fall back to seed
        pass
    
    # If we have time left, try a second run with different seed or more iterations?
    # Or try a local refinement from the best found so far.
    if time.time() < deadline - 10:
        # Local refinement using scipy.optimize.minimize
        from scipy.optimize import minimize
        
        # We need to be careful with the non-differentiability of max.
        # However, the max is likely taken at lag 0 (sum h^2) or nearby.
        # If we assume the max is at lag 0, the objective is smooth (ratio of quadratics).
        # But max could be elsewhere.
        # Let's just do a few Nelder-Mead steps on the coefficients.
        
        try:
            res2 = minimize(
                _objective, 
                res.x, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if res2.fun < best_val:
                best_h = _h_from_coeffs(res2.x)
                best_val = _evaluate(best_h)
        except Exception:
            pass

    # Final check
    if _evaluate(best_h) < best_val:
        best_val = _evaluate(best_h)
        
    return best_h

# Test locally if run as main
if __name__ == "__main__":
    h = construct()
    print("Value:", _evaluate(h))
    print("Sum:", np.sum(h))
    print("Max Conv:", np.max(np.convolve(h, h)))
