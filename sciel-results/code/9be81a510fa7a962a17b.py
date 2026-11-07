"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: evaluator kills after 150s wall-clock. Keep a safety margin.
_DEADLINE_SECONDS = 145.0
_START = time.time()


def _elapsed_ok() -> bool:
    return (time.time() - _START) < _DEADLINE_SECONDS


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF (Radial Basis Function) Model for Nonlinear Oscillator.
    
    Strategy:
    Instead of assuming a specific functional form (sinusoids, polynomials), 
    we model the target y as a linear combination of 10 Gaussian basis functions.
    The centers of these Gaussians are fixed on a grid in the (t, x, v) space.
    The 10 parameters are the weights for these basis functions.
    
    This allows the BFGS optimizer to fit a smooth, flexible surface that 
    can capture complex nonlinear dependencies between time, position, and velocity.
    
    Inputs:
    t: time (range [30, 50])
    x: position (range [-0.2056, 0.1749])
    v: velocity (range [-0.3391, 0.3624])
    
    Parameters (10):
    w0...w9: Weights for the 10 RBF centers.
    """
    # Safety margin guard
    if not _elapsed_ok():
        return np.zeros(np.shape(t), dtype=float)

    # Ensure float arrays and consistent shape
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)

    p = np.asarray(params, dtype=float)
    if p.shape[0] < 10:
        p = np.pad(p, (0, 10 - p.shape[0]))
    p = p[:10]

    # Define fixed centers for the RBFs.
    # We need to cover the input ranges:
    # t: [30, 50] -> midpoints: 35, 45
    # x: [-0.2, 0.17] -> midpoints: -0.1, 0.0, 0.1 (approx)
    # v: [-0.34, 0.36] -> midpoints: -0.15, 0.0, 0.15 (approx)
    
    # To get 10 distinct centers, we can sample a small grid.
    # Let's use 2 t-values, 2 x-values, 2 v-values = 8 centers. 
    # Plus 2 more for specific "edge" cases or just duplicate/shift.
    # Actually, let's just pick 10 well-distributed points.
    
    # t centers
    t_centers = np.array([32.0, 37.0, 42.0, 48.0])
    # x centers
    x_centers = np.array([-0.15, 0.0, 0.1])
    # v centers
    v_centers = np.array([-0.2, 0.0, 0.2])

    # Create a list of 10 centers
    # We'll mix them up to ensure coverage.
    # 0: (t=32, x=-0.15, v=-0.2)
    # 1: (t=32, x=0.0,  v=0.0)
    # 2: (t=32, x=0.1,  v=0.2)
    # 3: (t=37, x=-0.15, v=0.2)
    # 4: (t=37, x=0.0,  v=-0.2)
    # 5: (t=37, x=0.1,  v=0.0)
    # 6: (t=42, x=-0.15, v=0.0)
    # 7: (t=42, x=0.0,  v=0.2)
    # 8: (t=42, x=0.1,  v=-0.2)
    # 9: (t=48, x=0.0,  v=0.0)
    
    centers_t = np.array([32.0, 32.0, 32.0, 37.0, 37.0, 37.0, 42.0, 42.0, 42.0, 48.0])
    centers_x = np.array([-0.15, 0.0, 0.1, -0.15, 0.0, 0.1, -0.15, 0.0, 0.1, 0.0])
    centers_v = np.array([-0.2, 0.0, 0.2, 0.2, -0.2, 0.0, 0.0, 0.2, -0.2, 0.0])
    
    # Bandwidths (sigma) for the Gaussians.
    # We want them to be wide enough to cover the local neighborhood but distinct.
    # Input ranges: t~20, x~0.4, v~0.7
    # Let's use normalized distances or fixed sigmas.
    # Fixed sigmas are safer for BFGS stability if centers are fixed.
    # sigma_t ~ 5.0, sigma_x ~ 0.1, sigma_v ~ 0.15
    # Distance metric: sqrt((t-tc)^2/sigma_t^2 + (x-xc)^2/sigma_x^2 + (v-vc)^2/sigma_v^2)
    
    sigma_t = 5.0
    sigma_x = 0.1
    sigma_v = 0.15
    
    # Compute squared distances for each input point to each center
    # t shape: (N,), centers_t shape: (10,)
    # We want output shape (N, 10)
    
    dt = (t[:, np.newaxis] - centers_t[np.newaxis, :]) / sigma_t
    dx = (x[:, np.newaxis] - centers_x[np.newaxis, :]) / sigma_x
    dv = (v[:, np.newaxis] - centers_v[np.newaxis, :]) / sigma_v
    
    sq_dist = dt**2 + dx**2 + dv**2
    
    # Gaussian basis: exp(-0.5 * sq_dist)
    # We use 0.5 factor for standard Gaussian kernel.
    basis = np.exp(-0.5 * sq_dist)
    
    # Linear combination: basis (N, 10) @ p (10,) -> (N,)
    out = basis @ p
    
    # Guarantee finiteness and correct shape
    out = np.where(np.isfinite(out), out, 0.0)
    return np.asarray(out, dtype=float)

# EVOLVE-BLOCK-END
