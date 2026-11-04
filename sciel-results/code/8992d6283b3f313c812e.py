"""Optimized step function to minimize normalized max autoconvolution on [-1/4, 1/4]."""
import numpy as np
import time
from scipy.optimize import minimize

N = 600
# Interval width
h = 1.0 / (2 * N)  # Total length = 1/2, so h = 1/1200


def _autoconvolution_max_step(f):
    """
    Compute the normalized maximum of the autoconvolution of a nonnegative step function
    defined on N equal intervals over [-1/4, 1/4].
    
    The autoconvolution (c = f * f) is computed on a finer grid, then we find its maximum
    and normalize by (sum(f) * h)^2 to get the C1 upper bound.
    """
    f = np.asarray(f, dtype=np.float64)
    if len(f) != N:
        raise ValueError("f must have length N=600")
    if np.any(f < -1e-12):
        # Clamp tiny negatives to zero
        f = np.maximum(f, 0.0)
    
    # Total integral
    total = np.sum(f) * h
    if total <= 1e-15:
        return 0.0
    
    # Autoconvolution of the step function:
    # The function is piecewise constant on intervals [x_i, x_{i+1}) with height f[i]
    # where x_i = -1/4 + i*h for i=0..N.
    # The autoconvolution at a point y is:
    #   (f*f)(y) = ∫ f(y-t) f(t) dt
    # For a step function, this is a piecewise quadratic (or more precisely, the convolution
    # of two step functions gives a piecewise quadratic function).
    # 
    # To compute the maximum efficiently, we can use the fact that the autoconvolution
    # of a nonnegative step function is maximized either at the center or at some specific
    # points. However, the exact maximum can be found by evaluating on a sufficiently fine grid.
    #
    # Alternative: Use FFT-based convolution on a fine grid.
    
    # We'll compute the autoconvolution on a fine grid using FFT.
    # The support of the autoconvolution is [-1/2, 1/2] (since f is supported on [-1/4, 1/4]).
    # We need a grid fine enough to capture the maximum.
    
    # Number of fine grid points
    M = 8 * N  # 4800 points, should be sufficient
    dx = 1.0 / M  # Grid spacing for [-0.5, 0.5]
    # Actually, let's use a grid that covers [-0.5, 0.5] with M points
    x = np.linspace(-0.5 + dx/2, 0.5 - dx/2, M)
    
    # Create the step function on this fine grid
    # Each original interval [x_i, x_{i+1}) maps to a range of fine grid points
    fine_f = np.zeros(M)
    for i in range(N):
        # Interval i: [-1/4 + i*h, -1/4 + (i+1)*h)
        start_idx = int(np.floor((-1/4 + i*h + 0.5) / dx))
        end_idx = int(np.floor((-1/4 + (i+1)*h + 0.5) / dx))
        if end_idx > M:
            end_idx = M
        if start_idx < 0:
            start_idx = 0
        if start_idx < end_idx:
            fine_f[start_idx:end_idx] = f[i]
    
    # Compute autoconvolution using FFT
    # Pad to avoid circular convolution
    L = 1
    while L < 2 * M:
        L *= 2
    fine_f_pad = np.zeros(L)
    fine_f_pad[:M] = fine_f
    
    F = np.fft.fft(fine_f_pad, n=L)
    conv = np.fft.ifft(F * F).real
    conv = conv[:2*M]  # Keep only the valid part
    conv = conv * dx  # Scale by grid spacing
    
    max_conv = np.max(conv)
    return max_conv / (total * total)


def construct():
    """Construct a nonnegative step function to minimize the normalized max autoconvolution."""
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second budget
    
    # Initial guess: uniform function
    x0 = np.ones(N)
    
    # Try a few different initializations
    initializations = [
        np.ones(N),
        np.cos(np.pi * (np.arange(N) / N - 0.5)),  # Cosine window
        np.maximum(0, np.cos(2 * np.pi * (np.arange(N) / N - 0.5))),  # Positive part of cosine
    ]
    
    best_x = x0
    best_val = _autoconvolution_max_step(x0)
    
    for init in initializations:
        init = np.maximum(init, 0)  # Ensure nonnegative
        init = init / (np.sum(init) + 1e-15) * np.sum(x0)  # Scale to same total
        
        # Optimize
        def objective(x):
            if time.time() > deadline - 5:
                return 1e10
            return _autoconvolution_max_step(x)
        
        def gradient(x):
            if time.time() > deadline - 5:
                return np.zeros_like(x)
            # Numerical gradient
            eps = 1e-6
            grad = np.zeros_like(x)
            for i in range(N):
                xp = x.copy()
                xm = x.copy()
                xp[i] += eps
                xm[i] -= eps
                grad[i] = (objective(xp) - objective(xm)) / (2 * eps)
            return grad
        
        try:
            res = minimize(
                objective,
                init,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res.success and res.fun < best_val and time.time() < deadline - 10:
                best_x = res.x
                best_val = res.fun
        except Exception:
            pass
    
    # Final polish: coordinate descent
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline - 5:
                break
            old_val = best_val
            # Try small perturbations
            for delta in [1e-4, 1e-3, 1e-2, 0.1]:
                for sign in [1, -1]:
                    x_trial = best_x.copy()
                    x_trial[i] = max(0, x_trial[i] + sign * delta)
                    trial_val = _autoconvolution_max_step(x_trial)
                    if trial_val < old_val:
                        best_x = x_trial
                        best_val = trial_val
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            break
    
    # Return as list
    return [float(max(0, v)) for v in best_x]
