# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def _score(points):
    """Compute R = (d_min/d_max)^2 from 120 pairwise distances."""
    d = pdist(points)
    d_min = d.min()
    d_max = d.max()
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _smooth_min_logsumexp(distances, beta=50.0):
    """Soft-min via -log-sum-exp of negatives, scaled by beta."""
    # For soft-min: -1/beta * log(sum(exp(-beta * d_i)))
    # But we want to maximize d_min, so we minimize -soft_min
    # soft_min ≈ -1/beta * log(sum(exp(-beta * d)))
    # To make it smooth and bounded, use:
    x = -beta * distances
    x_max = np.max(x)
    if not np.isfinite(x_max):
        return 0.0
    s = np.sum(np.exp(x - x_max))
    return -(x_max + np.log(s)) / beta

def _smooth_max_logsumexp(distances, beta=50.0):
    """Soft-max via log-sum-exp, scaled by beta."""
    x = beta * distances
    x_max = np.max(x)
    if not np.isfinite(x_max):
        return 0.0
    s = np.sum(np.exp(x - x_max))
    return (x_max + np.log(s)) / beta

def _objective_smooth(params, beta=50.0):
    """
    Objective: minimize -log(soft_min/soft_max) which is equivalent to
    minimizing -log(soft_min) + log(soft_max).
    This is smooth and differentiable.
    """
    # params: complex numbers for 16 points (32 real values)
    # First point fixed at origin (scale invariance)
    z = params[2:] + 1j * params[3::2]  # points 1..15
    z = np.concatenate([[0.0 + 0.0j], z])  # prepend origin
    
    # Compute all pairwise distances
    n = len(z)
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = abs(z[i] - z[j])
            dists[i, j] = d
            dists[j, i] = d
    
    # Get unique distances (upper triangle)
    iu = np.triu_indices(n, 1)
    d_vals = dists[iu]
    
    if len(d_vals) == 0:
        return 0.0
    
    sm = _smooth_min_logsumexp(d_vals, beta)
    sx = _smooth_max_logsumexp(d_vals, beta)
    
    if sm <= 0 or sx <= 0 or not np.isfinite(sm) or not np.isfinite(sx):
        return 1e10
    
    # Minimize -log(sm/sx) = -log(sm) + log(sx)
    return -np.log(sm) + np.log(sx)

def _objective_grad(params, beta=50.0):
    """Numerical gradient for L-BFGS-B."""
    eps = 1e-7
    f0 = _objective_smooth(params, beta)
    grad = np.zeros_like(params)
    for i in range(len(params)):
        p_plus = params.copy()
        p_plus[i] += eps
        p_minus = params.copy()
        p_minus[i] -= eps
        grad[i] = (_objective_smooth(p_plus, beta) - _objective_smooth(p_minus, beta)) / (2*eps)
    return f0, grad

def _optimize_from_init(z_init, beta=50.0, maxiter=2000):
    """Run L-BFGS-B from a given initialization."""
    # z_init: array of 16 complex numbers, first at origin
    params = np.zeros(32)
    for i in range(1, 16):
        params[2*i] = z_init[i].real
        params[2*i+1] = z_init[i].imag
    
    result = minimize(
        _objective_smooth,
        params,
        args=(beta,),
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
    )
    
    # Reconstruct points
    z = np.zeros(16, dtype=complex)
    z[0] = 0.0
    for i in range(1, 16):
        z[i] = result.x[2*i] + 1j * result.x[2*i+1]
    
    points = np.column_stack([z.real, z.imag])
    return points, result.fun

def _refine_with_nm(points, maxiter=5000):
    """Refine using Nelder-Mead on the raw (nonsmooth) objective."""
    def nm_obj(p):
        pts = p.reshape(16, 2)
        return -_score(pts)
    
    x0 = points.flatten()
    result = minimize(
        nm_obj,
        x0,
        method='Nelder-Mead',
        options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
    )
    return result.x.reshape(16, 2)

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    deadline = time.time() + 600.0
    
    best_points = None
    best_score = 0.0
    
    # Strategy: multiple random initializations on/near the unit circle,
    # optimize with smooth L-BFGS-B, then polish with Nelder-Mead.
    
    rng = np.random.RandomState(42)
    
    # Try several different initial configurations
    n_trials = 12
    
    for trial in range(n_trials):
        if time.time() > deadline - 60:
            break
        
        # Initialize: first point at origin, others scattered
        # Use different radial/angular patterns
        if trial == 0:
            # Uniform circle
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            radii = np.ones(n)
        elif trial == 1:
            # Two concentric circles
            angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
            angles2 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
            radii = np.concatenate([np.ones(8), 0.5*np.ones(8)])
            angles = np.concatenate([angles1, angles2])
        elif trial == 2:
            # Random angles, unit radius
            angles = rng.uniform(0, 2*np.pi, n)
            radii = np.ones(n)
        elif trial == 3:
            # Random radii and angles
            angles = rng.uniform(0, 2*np.pi, n)
            radii = rng.uniform(0.3, 1.0, n)
        elif trial == 4:
            # Fibonacci-like spiral
            i = np.arange(n)
            radii = np.sqrt(i / (n - 1))
            angles = i * 2.399963229728653  # golden angle
        elif trial == 5:
            # Random with some structure
            angles = rng.uniform(0, 2*np.pi, n)
            radii = 0.8 + 0.2 * rng.uniform(0, 1, n)
        elif trial == 6:
            # Grid-like
            cols = 4
            rows = 4
            xs = np.linspace(-1, 1, cols)
            ys = np.linspace(-1, 1, rows)
            X, Y = np.meshgrid(xs, ys)
            radii = np.sqrt(X.flatten()**2 + Y.flatten()**2)
            angles = np.arctan2(Y.flatten(), X.flatten())
        elif trial == 7:
            # Random perturbation of circle
            angles = np.linspace(0, 2*np.pi, n, endpoint=False) + rng.normal(0, 0.1, n)
            radii = 1.0 + rng.normal(0, 0.05, n)
        elif trial == 8:
            # Three rings
            ring_sizes = [6, 5, 5]
            radii_list = [1.0, 0.6, 0.3]
            radii = np.concatenate([np.full(s, r) for s, r in zip(ring_sizes, radii_list)])
            angles = np.concatenate([np.linspace(0, 2*np.pi, s, endpoint=False) + rng.uniform(0, 0.5, 1) 
                                     for s in ring_sizes])
        elif trial == 9:
            # Random
            angles = rng.uniform(0, 2*np.pi, n)
            radii = rng.uniform(0.5, 1.0, n)
        elif trial == 10:
            # Hexagonal-ish
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            radii = np.ones(n)
            # Perturb some
            for i in [2, 5, 8, 11, 14]:
                radii[i] = 0.7
        else:
            # Pure random
            angles = rng.uniform(0, 2*np.pi, n)
            radii = rng.uniform(0.2, 1.0, n)
        
        # Ensure first point at origin
        radii[0] = 0.0
        angles[0] = 0.0
        
        z_init = radii * np.exp(1j * angles)
        
        # Optimize with smooth surrogate
        try:
            points, smooth_val = _optimize_from_init(z_init, beta=50.0, maxiter=2000)
            if time.time() > deadline - 30:
                s = _score(points)
                if s > best_score:
                    best_score = s
                    best_points = points.copy()
                continue
            s = _score(points)
            if s > best_score:
                best_score = s
                best_points = points.copy()
        except Exception:
            continue
        
        # Polish with Nelder-Mead
        if best_points is not None and time.time() < deadline - 20:
            try:
                refined = _refine_with_nm(best_points, maxiter=3000)
                s_ref = _score(refined)
                if s_ref > best_score:
                    best_score = s_ref
                    best_points = refined.copy()
            except Exception:
                pass
    
    # Final check: if we have no valid result, fall back to circle
    if best_points is None or best_score <= 0:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Ensure shape is exactly (16, 2) and finite
    best_points = np.asarray(best_points, dtype=np.float64).reshape(16, 2)
    if not np.all(np.isfinite(best_points)):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points

# EVOLVE-BLOCK-END
