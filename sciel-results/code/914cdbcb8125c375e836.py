"""
Optimized program for the third autocorrelation inequality task using Zadoff-Chu sequences.

The objective function is:
    value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)

Strategy: Use Zadoff-Chu (CAZAC) sequences as a structural basis. These have
near-zero non-zero autocorrelation sidelobes, which may allow the signed max
of the autoconvolution to be reduced relative to sum(h)^2. We optimize over
real-valued combinations and sign patterns.
"""

import numpy as np
import time
import sys

N = 400
DEADLINE = 145.0  # safety margin before 150s kill

def _zadoff_chu(n, u):
    """Generate a Zadoff-Chu sequence of length n with root u (1 <= u < n, gcd(u,n)=1)."""
    if n % 2 == 0:
        h = np.exp(1j * np.pi * u * np.arange(n) * (np.arange(n) + 1) / n)
    else:
        h = np.exp(1j * np.pi * u * np.arange(n) * (np.arange(n) + 1) / n)
    return h

def _evaluate(h):
    """Compute the objective value."""
    s = np.sum(h)
    if s == 0:
        return float('inf')
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def construct():
    """
    Returns a sequence of 400 signed step heights that minimizes the autocorrelation bound.
    Uses Zadoff-Chu based initialization with local optimization.
    """
    t0 = time.time()
    
    # Start with the known best (constant vector, value=2.0)
    best_h = np.ones(N, dtype=np.float64)
    best_val = _evaluate(best_h)  # should be 2.0
    
    # Try various Zadoff-Chu based real sequences
    # For even N=400, valid roots u must be coprime to N
    # N = 400 = 2^4 * 5^2, so u must be odd and not divisible by 5
    
    import math
    
    candidates = []
    
    # Generate several ZC sequences and take real/imag parts or combinations
    for u in [1, 3, 7, 9, 11, 13, 17, 19, 21, 23, 27, 29, 31, 33, 37, 39, 41, 43, 47, 49]:
        if math.gcd(u, N) != 1:
            continue
        zc = _zadoff_chu(N, u)
        # Real part
        h_real = np.real(zc)
        # Imaginary part
        h_imag = np.imag(zc)
        # Various combinations
        candidates.append(h_real)
        candidates.append(h_imag)
        candidates.append(np.real(zc) + 0.5 * np.imag(zc))
        candidates.append(np.real(zc) - 0.5 * np.imag(zc))
        # Sign-flipped
        candidates.append(-h_real)
        candidates.append(-h_imag)
        
        if time.time() - t0 > DEADLINE:
            break
    
    # Evaluate all candidates
    for h in candidates:
        if not np.all(np.isfinite(h)):
            continue
        if np.sum(h) == 0:
            continue
        val = _evaluate(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        
        if time.time() - t0 > DEADLINE:
            break
    
    # If we found something better, try local refinement with scipy
    if best_val < 2.0 and time.time() - t0 < DEADLINE - 30:
        try:
            from scipy.optimize import minimize
            
            def objective(x):
                h = x
                s = np.sum(h)
                if abs(s) < 1e-10:
                    return 1e10
                conv = np.convolve(h, h)
                return abs(2.0 * N * np.max(conv) / (s * s))
            
            # Try L-BFGS-B from the best candidate
            result = minimize(objective, best_h, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.fun < best_val:
                best_h = result.x
                best_val = result.fun
        except Exception:
            pass
    
    # Final safety checks
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N, dtype=np.float64)
    if np.sum(best_h) == 0:
        best_h = np.ones(N, dtype=np.float64)
    
    return best_h

# EVOLVE-BLOCK-START
# Zadoff-Chu sequences provide CAZAC structure with near-zero sidelobes.
# This structural change from constant vectors may find basins where
# the signed max of autoconvolution is reduced relative to sum(h)^2.
# EVOLVE-BLOCK-END
