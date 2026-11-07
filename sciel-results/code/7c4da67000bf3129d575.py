"""Improved program for Erdos minimum overlap task using spectral parameterization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0
START = time.time()

def remaining():
    return DEADLINE - (time.time() - START)

def objective(h):
    """Compute the value to minimize."""
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0

def objective_smooth(h, beta=50.0):
    """Smooth surrogate using log-sum-exp for the max."""
    c = np.correlate(h, 1.0 - h, mode="full")
    c_max = np.max(c)
    shifted = beta * (c - c_max)
    log_sum = np.log(np.sum(np.exp(shifted))) + beta * c_max
    return float(log_sum / N * 2.0)

def project(h):
    """Project to [0,1]^N with sum = N/2."""
    h = np.asarray(h, dtype=np.float64).copy()
    target = N / 2.0
    
    # Bisection to find the right offset
    lo, hi = -1.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        h_trial = np.clip(h + mid, 0.0, 1.0)
        s = h_trial.sum()
        if s < target:
            lo = mid
        else:
            hi = mid
    
    h = np.clip(h + (lo + hi) / 2.0, 0.0, 1.0)
    
    # Final adjustment for precision
    s = h.sum()
    if abs(s - target) > 1e-12:
        diff = target - s
        free = (h > 1e-12) & (h < 1.0 - 1e-12)
        if np.any(free):
            h[free] += diff / np.sum(free)
            h = np.clip(h, 0.0, 1.0)
    
    return h

def construct():
    t0 = time.time()
    i = np.arange(N)
    
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)
    
    # Strategy: Use spectral parameterization with cosine basis functions
    # Frequencies that are divisors of N=100: 1, 2, 4, 5, 10, 20, 25, 50
    # Also try nearby frequencies for more flexibility
    
    # Build basis matrix
    freqs = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25]
    n_basis = len(freqs)
    
    # Basis matrix B where B[i, k] = cos(2*pi*freqs[k]*i/N)
    B = np.zeros((N, n_basis))
    for k, f in enumerate(freqs):
        B[:, k] = np.cos(2 * np.pi * f * i / N)
    
    # Parameterize h = 0.5 + B @ a
    # Sum constraint: sum(h) = N/2 is automatically satisfied since sum of cosines over full periods is ~0
    # Box constraint: 0 <= h <= 1
    
    def h_from_a(a):
        return 0.5 + B @ a
    
    def obj_from_a(a, beta=50.0):
        h = h_from_a(a)
        h = np.clip(h, 0.0, 1.0)  # Soft enforcement during optimization
        return objective_smooth(h, beta)
    
    # Optimize coefficients a using SLSQP
    # Bounds on a: ensure h stays in [0,1]
    # For each basis function, max amplitude is 0.5 to keep h in [0,1]
    # But combined, we need to be more careful. Use conservative bounds.
    bounds_a = [(-0.5, 0.5)] * n_basis
    
    # Multiple starting points
    for trial in range(20):
        if remaining() < 3:
            break
        
        # Random initialization
        np.random.seed(42 + trial)
        a_init = np.random.randn(n_basis) * 0.1
        
        # Also try pure single-frequency starts
        if trial < len(freqs):
            a_init = np.zeros(n_basis)
            a_init[trial] = 0.3 * (1 if trial % 2 == 0 else -1)
        
        def obj_func(a):
            return obj_from_a(a, beta=50.0)
        
        try:
            result = minimize(
                obj_func,
                a_init,
                method='SLSQP',
                bounds=bounds_a,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            
            h_candidate = project(h_from_a(result.x))
            val_candidate = objective(h_candidate)
            
            if val_candidate < best_val:
                best_h = h_candidate
                best_val = val_candidate
                
        except Exception:
            pass
    
    # Strategy 2: Local refinement using L-BFGS-B on h directly
    if remaining() > 10:
        h_curr = best_h.copy()
        
        for k in range(1, 9):
            if remaining() < 5:
                break
            
            # Add small cosine perturbation
            delta = 0.05 * np.cos(2 * np.pi * k * i / N)
            h_mix = h_curr + delta
            h_mix = project(h_mix)
            
            def obj_func(x):
                return objective_smooth(project(x), beta=50.0)
            
            try:
                result = minimize(
                    obj_func,
                    h_mix,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-7}
                )
                
                h_candidate = project(result.x)
                val_candidate = objective(h_candidate)
                
                if val_candidate < best_val:
                    best_h = h_candidate
                    best_val = val_candidate
                    
            except Exception:
                pass
    
    # Strategy 3: Final high-precision polish
    if remaining() > 5:
        # Use the exact (non-smooth) objective for final polish
        def obj_exact(x):
            return objective(project(x))
        
        try:
            result = minimize(
                obj_exact,
                best_h,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-9}
            )
            
            h_candidate = project(result.x)
            val_candidate = objective(h_candidate)
            
            if val_candidate < best_val:
                best_h = h_candidate
                best_val = val_candidate
                
        except Exception:
            pass
    
    # Final projection and return
    best_h = project(best_h)
    return best_h
