# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

_N = 14


def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    n = len(points)
    if n < 2:
        return 0.0
    dists = pdist(points, metric='euclidean')
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    return (d_min / d_max) ** 2


def _spherical_to_cartesian(theta_phi, n=_N):
    """Convert (theta, phi) pairs to 3D unit sphere points."""
    thetas = theta_phi[0::2]
    phis = theta_phi[1::2]
    x = np.sin(thetas) * np.cos(phis)
    y = np.sin(thetas) * np.sin(phis)
    z = np.cos(thetas)
    return np.column_stack([x, y, z])


def _neg_score_spherical(theta_phi, n=_N):
    """Objective: negative of score for minimization."""
    points = _spherical_to_cartesian(theta_phi, n)
    if not np.all(np.isfinite(points)):
        return 1.0
    return -_score(points)


def _neg_score_spherical_soft(theta_phi, n=_N, sigma=0.1):
    """Soft surrogate: smooth approximation using log-sum-exp on distances.
    
    Instead of min/max which are non-smooth, we use a soft version.
    """
    points = _spherical_to_cartesian(theta_phi, n)
    if not np.all(np.isfinite(points)):
        return 1.0
    dists = pdist(points, metric='euclidean')
    if len(dists) == 0:
        return 1.0
    
    # Soft min: -logsumexp(-d/sigma) * sigma
    # Soft max: logsumexp(d/sigma) * sigma
    d_scaled = dists / sigma
    # Clip for numerical stability
    d_scaled = np.clip(d_scaled, -500, 500)
    
    # Soft min
    max_neg = np.max(-d_scaled)
    soft_min = -sigma * (max_neg + np.log(np.sum(np.exp(-d_scaled - max_neg))))
    
    # Soft max
    max_d = np.max(d_scaled)
    soft_max = sigma * (max_d + np.log(np.sum(np.exp(d_scaled - max_d))))
    
    if soft_max <= 0:
        return 1.0
    
    ratio = soft_min / soft_max
    return -(ratio ** 2)


def _coordinate_descent_polish(theta_phi, n=_N, iterations=200, time_limit=30.0):
    """Polish using coordinate descent on spherical coordinates."""
    start = time.time()
    x = theta_phi.copy()
    current_score = _score(_spherical_to_cartesian(x, n))
    
    for it in range(iterations):
        if time.time() - start > time_limit:
            break
        improved = False
        for i in range(n):
            # Optimize theta_i
            best_theta = x[2*i]
            for delta in [0.1, 0.05, 0.01, 0.005, 0.001]:
                for sign in [1, -1]:
                    new_x = x.copy()
                    new_x[2*i] = best_theta + sign * delta
                    new_score = _score(_spherical_to_cartesian(new_x, n))
                    if new_score > current_score:
                        current_score = new_score
                        best_theta = new_x[2*i]
                        improved = True
            
            x[2*i] = best_theta
            
            # Optimize phi_i
            best_phi = x[2*i+1]
            for delta in [0.1, 0.05, 0.01, 0.005, 0.001]:
                for sign in [1, -1]:
                    new_x = x.copy()
                    new_x[2*i+1] = best_phi + sign * delta
                    new_score = _score(_spherical_to_cartesian(new_x, n))
                    if new_score > current_score:
                        current_score = new_score
                        best_phi = new_x[2*i+1]
                        improved = True
            
            x[2*i+1] = best_phi
        
        if not improved:
            break
    
    return x


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points as initial guess."""
    golden_angle = np.pi * (3 - np.sqrt(5))
    thetas = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
    phis = golden_angle * np.arange(n)
    theta_phi = np.empty(2 * n)
    theta_phi[0::2] = thetas
    theta_phi[1::2] = phis
    return theta_phi


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min / d_max)^2.
    
    Uses two-phase optimization:
    1. Soft surrogate to guide Nelder-Mead to a good region
    2. Raw score optimization with coordinate descent polish
    """
    n = _N
    deadline = time.time() + 600.0
    
    best_points = None
    best_score = -np.inf
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Multi-start with soft surrogate, then polish with raw score
    max_restarts = 40
    
    for i in range(max_restarts):
        if time.time() > deadline - 45:
            break
        
        # Generate initial spherical coordinates
        if i == 0:
            # Fibonacci sphere
            theta_phi = _fibonacci_sphere(n)
        elif i == 1:
            # Uniform random
            thetas = rng.uniform(0.05, np.pi - 0.05, n)
            phis = rng.uniform(0, 2 * np.pi, n)
            theta_phi = np.empty(2 * n)
            theta_phi[0::2] = thetas
            theta_phi[1::2] = phis
        elif i == 2:
            # Slightly biased toward equator
            thetas = rng.uniform(np.pi/5, 4*np.pi/5, n)
            phis = rng.uniform(0, 2 * np.pi, n)
            theta_phi = np.empty(2 * n)
            theta_phi[0::2] = thetas
            theta_phi[1::2] = phis
        elif i == 3:
            # Two rings (like a dumbbell)
            thetas = np.concatenate([
                np.full(n//2, np.pi/4 + rng.normal(0, 0.1, n//2)),
                np.full(n - n//2, 3*np.pi/4 + rng.normal(0, 0.1, n - n//2))
            ])
            phis = rng.uniform(0, 2 * np.pi, n)
            theta_phi = np.empty(2 * n)
            theta_phi[0::2] = thetas
            theta_phi[1::2] = phis
        else:
            # Random
            thetas = rng.uniform(0.05, np.pi - 0.05, n)
            phis = rng.uniform(0, 2 * np.pi, n)
            theta_phi = np.empty(2 * n)
            theta_phi[0::2] = thetas
            theta_phi[1::2] = phis
        
        # Phase 1a: Soft surrogate optimization (shorter)
        try:
            result_soft = minimize(
                _neg_score_spherical_soft,
                theta_phi,
                args=(n, 0.05),
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'maxfev': 1000,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                }
            )
        except Exception:
            continue
        
        if time.time() > deadline - 30:
            break
        
        # Phase 1b: Raw score optimization from soft result
        try:
            result_raw = minimize(
                _neg_score_spherical,
                result_soft.x,
                args=(n,),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'maxfev': 2000,
                    'xatol': 1e-10,
                    'fatol': 1e-10,
                }
            )
        except Exception:
            # Fall back to soft result
            result_raw = result_soft
        
        if time.time() > deadline - 20:
            break
        
        # Phase 1c: Coordinate descent polish
        polished_x = _coordinate_descent_polish(result_raw.x, n, iterations=100, time_limit=15.0)
        
        # Evaluate
        points = _spherical_to_cartesian(polished_x, n)
        if not np.all(np.isfinite(points)):
            continue
        
        score = _score(points)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        # Also try polishing the soft result directly
        if time.time() > deadline - 15:
            break
        polished_soft_x = _coordinate_descent_polish(result_soft.x, n, iterations=100, time_limit=10.0)
        points_soft = _spherical_to_cartesian(polished_soft_x, n)
        if np.all(np.isfinite(points_soft)):
            score_soft = _score(points_soft)
            if score_soft > best_score:
                best_score = score_soft
                best_points = points_soft.copy()
    
    # If no valid result, return Fibonacci sphere
    if best_points is None:
        theta_phi = _fibonacci_sphere(n)
        best_points = _spherical_to_cartesian(theta_phi, n)
    
    # Final check
    if not np.all(np.isfinite(best_points)):
        best_points = rng.randn(n, 3)
    
    # Ensure shape is correct
    if best_points.shape != (n, 3):
        best_points = rng.randn(n, 3)
    
    return best_points


# EVOLVE-BLOCK-END
