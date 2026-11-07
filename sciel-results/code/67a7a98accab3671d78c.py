"""Improved program for Erdos minimum overlap task."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 140.0
START = time.time()

def remaining():
    return DEADLINE - (time.time() - START)

def objective(h):
    """Compute the value to minimize."""
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0

def project(h):
    """Project to [0,1]^N with sum = N/2."""
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(100):
        s = h.sum()
        if abs(s - target) < 1e-12:
            break
        h = h + (target - s) / N
        h = np.clip(h, 0.0, 1.0)
    # Final adjustment
    s = h.sum()
    if abs(s - target) > 1e-12:
        # Distribute the difference
        diff = target - s
        # Find indices not at bounds
        free = (h > 1e-10) & (h < 1.0 - 1e-10)
        if np.any(free):
            h[free] += diff / np.sum(free)
            h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    t0 = time.time()
    
    # Start with cosine initialization
    i = np.arange(N)
    h0 = 0.5 + 0.5 * np.cos(2 * np.pi * i / N)
    h0 = project(h0)
    
    best_h = h0.copy()
    best_val = objective(h0)
    
    # Try multiple frequency components and optimize
    for freq in [1, 2, 3, 4, 5]:
        if remaining() < 10:
            break
        
        h_init = 0.5 + 0.5 * np.cos(2 * np.pi * freq * i / N)
        h_init = project(h_init)
        
        # Use L-BFGS-B to minimize
        def obj_and_grad(x):
            h = project(x)
            val = objective(h)
            # Numerical gradient
            eps = 1e-6
            grad = np.zeros_like(h)
            for j in range(N):
                h_plus = h.copy()
                h_plus[j] += eps
                h_plus = project(h_plus)
                h_minus = h.copy()
                h_minus[j] -= eps
                h_minus = project(h_minus)
                grad[j] = (objective(h_plus) - objective(h_minus)) / (2 * eps)
            return val, grad
        
        result = minimize(
            lambda x: objective(project(x)),
            h_init,
            method='L-BFGS-B',
            bounds=[(0.0, 1.0)] * N,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.success or objective(project(result.x)) < best_val:
            h_candidate = project(result.x)
            val_candidate = objective(h_candidate)
            if val_candidate < best_val:
                best_h = h_candidate
                best_val = val_candidate
    
    # Additional local optimization with smaller steps
    if remaining() > 5:
        # Try random perturbations
        np.random.seed(42)
        for _ in range(50):
            if remaining() < 5:
                break
            h_pert = best_h + 0.1 * np.random.randn(N)
            h_pert = project(h_pert)
            val_pert = objective(h_pert)
            if val_pert < best_val:
                best_h = h_pert
                best_val = val_pert
    
    # Final projection to ensure constraints
    best_h = project(best_h)
    return best_h
# EVOLVE-BLOCK-END
