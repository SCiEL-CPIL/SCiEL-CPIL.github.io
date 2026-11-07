"""
Strategy: Fourier-based state-space model for a nonlinear oscillator.

The target is the acceleration y = dv/dt of a nonlinear oscillator with inputs
t (time), x (position), v (velocity).

We model the dynamics as:
    y = -k(t) * x - c(t) * v + f(t)

where k(t), c(t), f(t) are time-varying coefficients expressed as low-order
Fourier series. This captures periodic changes in effective stiffness, damping,
and forcing without introducing ill-conditioned high-degree polynomials.

Parameters (10 total):
    p0: k0  (constant stiffness)
    p1: k1  (cos coefficient for stiffness)
    p2: k2  (sin coefficient for stiffness)
    p3: c0  (constant damping)
    p4: c1  (cos coefficient for damping)
    p5: c2  (sin coefficient for damping)
    p6: f0  (constant forcing)
    p7: f1  (cos coefficient for forcing)
    p8: f2  (sin coefficient for forcing)
    p9: w   (angular frequency)

All Fourier basis terms are bounded in [-1, 1], which helps with numerical
conditioning for BFGS optimization starting from all-ones initialization.
"""
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using time-varying state-space model with
    low-order Fourier series coefficients.
    
    Inputs:
    t: time
    x: position
    v: velocity
    params: array of 10 floats
    
    Returns:
    prediction: array of shape (N,)
    """
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Angular frequency - ensure it's reasonable to avoid numerical issues
    # Use absolute value to keep it positive, with a small floor to avoid zero
    w = np.abs(p9) + 1e-6
    
    # Compute Fourier basis terms
    # These are bounded in [-1, 1] which helps with conditioning
    cos_wt = np.cos(w * t)
    sin_wt = np.sin(w * t)
    
    # Time-varying coefficients
    # k(t): stiffness
    k_t = p0 + p1 * cos_wt + p2 * sin_wt
    
    # c(t): damping
    c_t = p3 + p4 * cos_wt + p5 * sin_wt
    
    # f(t): forcing
    f_t = p6 + p7 * cos_wt + p8 * sin_wt
    
    # Acceleration: y = -k(t)*x - c(t)*v + f(t)
    result = -k_t * x - c_t * v + f_t
    
    return result
