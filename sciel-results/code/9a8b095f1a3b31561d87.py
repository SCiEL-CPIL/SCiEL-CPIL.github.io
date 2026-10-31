import numpy as np
from scipy.optimize import minimize
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    # Initial guess: a smooth bump (Gaussian-like) to reduce autoconvolution peak
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    # Start with a raised cosine / smooth window
    h0 = np.maximum(0, np.cos(np.pi * x / 0.25))
    h0 = h0 / np.sum(h0) * np.sqrt(N)  # normalize
    
    def objective(h):
        """Minimize the normalized peak of autoconvolution."""
        # Autoconvolution via FFT
        n_fft = 2 * N
        H = np.fft.rfft(h, n=n_fft)
        AC = np.fft.irfft(H * np.conj(H), n=n_fft)
        # Autoconvolution values: AC[k] for k=0..N-1 (and symmetric)
        # The peak of autoconvolution
        peak = np.max(AC[:N])
        # Normalize: use L1 norm squared as reference, or just return peak
        # For the inequality, we want min of max(AC) / (sum(h))^2 * N or similar
        # Let's use: peak / (sum(h)^2 / N) to make it scale-invariant
        s = np.sum(h)
        if s < 1e-10:
            return 1e10
        norm = s * s / N
        return peak / norm
    
    def jac(h):
        """Gradient approximation via finite differences."""
        n = len(h)
        g = np.zeros(n)
        eps = 1e-7
        f0 = objective(h)
        for i in range(n):
            h[i] += eps
            f1 = objective(h)
            h[i] -= eps
            g[i] = (f1 - f0) / eps
        return g
    
    # Try to optimize
    best_h = h0.copy()
    best_val = objective(h0)
    
    try:
        # Use L-BFGS-B with bounds [0, inf)
        bounds = [(0, None)] * N
        result = minimize(
            objective,
            h0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        if result.success or result.fun < best_val:
            best_h = np.maximum(0, result.x)
            best_val = result.fun
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time - 5:
        # Not enough time, return what we have
        return list(best_h)
    
    # Try a second round with different initial guess if time permits
    if time.time() < deadline - 20:
        # Try a flatter initial guess
        h1 = np.ones(N)
        h1 = h1 / np.sum(h1) * np.sqrt(N)
        
        try:
            result2 = minimize(
                objective,
                h1,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            if result2.success or result2.fun < best_val:
                best_h = np.maximum(0, result2.x)
                best_val = result2.fun
        except Exception:
            pass
    
    # Ensure nonnegativity and return as list
    best_h = np.maximum(0, best_h)
    return list(best_h)
