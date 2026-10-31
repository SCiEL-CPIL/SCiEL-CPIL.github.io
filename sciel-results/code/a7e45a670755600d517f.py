"""
Minimize the normalized maximum of the autoconvolution of a nonnegative
step function with 600 equal intervals on [-1/4, 1/4].

Approach:
- The autoconvolution of a step function on [-1/4, 1/4] with 600 bins is
  computed via FFT.
- We use a low-rank Gaussian mixture representation to parameterize smooth,
  nonnegative step functions, then optimize continuous parameters via
  differential evolution / L-BFGS-B to escape discrete local minima.
- The objective is the normalized L-infinity norm of the autoconvolution.
- A deadline guard ensures we return within the time limit.
"""

import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.signal import fftconvolve

N = 600
DEADLINE = 120.0  # seconds
START_TIME = time.time()


def time_left():
    return DEADLINE - (time.time() - START_TIME)


def compute_autoconvolution(f):
    """Compute the autoconvolution of step heights f (length N)."""
    # f is defined on [-1/4, 1/4], so each bin has width 1/300
    # Autoconvolution: (f * f)(t) = integral f(s) f(t-s) ds
    # For step functions, this is a discrete convolution scaled by bin width
    conv = fftconvolve(f, f, mode='full')
    # The full convolution has length 2N-1
    # The domain of the autoconvolution is [-1/2, 1/2]
    # We normalize by the maximum possible value or by a standard normalization
    # The "normalized maximum" likely means max(conv) / some reference
    # For a step function with total integral = sum(f) * dw, where dw = 1/300
    # A natural normalization: divide by (sum(f)*dw)^2 to get a probability-like density
    total = np.sum(f) * (1.0 / 300.0)
    if total < 1e-12:
        return np.max(np.abs(conv))
    # Normalize the autoconvolution so that its integral is 1 (if f is a pdf)
    # Actually, if f is a pdf on [-1/4,1/4], then f*f is the autoconvolution pdf
    # on [-1/2,1/2], and its integral is 1.
    # The "normalized maximum" is just max(f*f) since it's already a pdf.
    # But wait, the problem says "normalized maximum of its autoconvolution".
    # Let me think: if f is a step function (not necessarily a pdf), the autoconvolution
    # is a function on [-1/2, 1/2]. The normalization might be by the L1 norm or
    # by the square of the L1 norm of f.
    # 
    # Actually, for the first autocorrelation inequality, C1 is related to the
    # L-infinity norm of the autoconvolution of a function in L2 or L1.
    # 
    # Let me use a standard normalization: max(conv) / max(conv) = 1? No.
    # 
    # The seed program returns [1.0]*N, so let's compute what the score would be.
    # For f = [1]*600, conv = fftconvolve([1]*600, [1]*600) which is a triangle
    # with peak 600 at the center.
    # If the score is just max(conv), then for the seed it's 600.
    # 
    # Let me assume the score is max(conv) / (sum(f) * dw)^2 * something, or
    # simply the peak of the autoconvolution when f is normalized to have unit L1 norm.
    #
    # Let me try: normalize f to have unit L1 norm, then compute max of autoconvolution.
    # For f = [1]*600, L1 norm = 600 * (1/300) = 2. Normalized f = [0.5]*600.
    # Autoconvolution peak = 600 * (0.5)^2 * (1/300) = 600 * 0.25 / 300 = 0.5.
    #
    # I'll use: normalize f to unit L1 norm, compute autoconvolution, return its max.
    return np.max(conv) / (total * total)


def objective_from_params(params):
    """Convert Gaussian mixture params to step heights and compute objective."""
    n_gaussians = len(params) // 3
    f = np.zeros(N)
    dw = 1.0 / 300.0
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + dw / 2
    
    for i in range(n_gaussians):
        amp = params[3 * i]
        center = params[3 * i + 1]
        width = params[3 * i + 2]
        f += amp * np.exp(-0.5 * ((x - center) / width) ** 2)
    
    # Ensure nonnegativity
    f = np.maximum(f, 0.0)
    
    if np.sum(f) < 1e-12:
        return 1e10
    
    # Normalize to unit L1
    f = f / (np.sum(f) * dw)
    
    return compute_autoconvolution(f)


def construct():
    """Construct a step function minimizing the normalized max autoconvolution."""
    best_f = np.ones(N)
    best_score = compute_autoconvolution(best_f / (np.sum(best_f) / 300.0))
    
    # Try a few Gaussian mixture configurations
    configs = [
        1, 2, 3, 4, 5
    ]
    
    for n_gauss in configs:
        if time_left() < 10:
            break
        
        # Parameter bounds: [amp, center, width] for each Gaussian
        # amp > 0, center in [-0.25, 0.25], width in [0.005, 0.25]
        bounds = []
        for _ in range(n_gauss):
            bounds.append((0.01, 10.0))      # amplitude
            bounds.append((-0.25, 0.25))     # center
            bounds.append((0.005, 0.25))     # width
        
        # Use differential evolution for global optimization
        try:
            result = differential_evolution(
                objective_from_params,
                bounds=bounds,
                maxiter=50,
                tol=1e-6,
                seed=42,
                polish=False,
                workers=1
            )
            if result.fun < best_score:
                best_score = result.fun
                # Convert back to step heights
                n_gaussians = len(result.x) // 3
                f = np.zeros(N)
                dw = 1.0 / 300.0
                x = np.linspace(-0.25, 0.25, N, endpoint=False) + dw / 2
                for i in range(n_gaussians):
                    amp = result.x[3 * i]
                    center = result.x[3 * i + 1]
                    width = result.x[3 * i + 2]
                    f += amp * np.exp(-0.5 * ((x - center) / width) ** 2)
                f = np.maximum(f, 0.0)
                if np.sum(f) > 1e-12:
                    f = f / (np.sum(f) * dw)
                    best_f = f
        except Exception:
            pass
    
    # Polish with L-BFGS-B if time allows
    if time_left() > 5:
        try:
            n_gauss = 3
            bounds = []
            for _ in range(n_gauss):
                bounds.append((0.01, 10.0))
                bounds.append((-0.25, 0.25))
                bounds.append((0.005, 0.25))
            
            # Start from current best
            x0 = np.array([1.0, 0.0, 0.15] * n_gauss)
            
            result = minimize(
                objective_from_params,
                x0=x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            if result.fun < best_score:
                best_score = result.fun
                n_gaussians = len(result.x) // 3
                f = np.zeros(N)
                dw = 1.0 / 300.0
                x = np.linspace(-0.25, 0.25, N, endpoint=False) + dw / 2
                for i in range(n_gaussians):
                    amp = result.x[3 * i]
                    center = result.x[3 * i + 1]
                    width = result.x[3 * i + 2]
                    f += amp * np.exp(-0.5 * ((x - center) / width) ** 2)
                f = np.maximum(f, 0.0)
                if np.sum(f) > 1e-12:
                    f = f / (np.sum(f) * dw)
                    best_f = f
        except Exception:
            pass
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    
    return list(best_f)
