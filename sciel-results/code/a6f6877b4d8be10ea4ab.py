"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math


def _log_energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = points.shape[0]
    # Compute all pairwise distances
    # points is (n, 3)
    # dist^2 = sum(x_i^2) + sum(x_j^2) - 2*x_i.x_j = 2 - 2*x_i.x_j for unit vectors
    # dist = sqrt(2 - 2*x_i.x_j)
    # log(1/dist) = -log(dist) = -0.5*log(2 - 2*x_i.x_j)
    
    # Use matrix form for efficiency
    # G = points @ points.T  (n x n)
    G = points @ points.T
    # dist_sq = 2 - 2*G, diagonal is 0
    dist_sq = 2.0 - 2.0 * G
    # Set diagonal to 1 to avoid log(0) issues, then we'll mask it
    np.fill_diagonal(dist_sq, 1.0)
    
    # We need sum over i<j of log(1/sqrt(dist_sq)) = sum over i<j of -0.5*log(dist_sq)
    # = -0.5 * sum over i<j of log(dist_sq)
    # = -0.25 * sum over all i!=j of log(dist_sq)  [since each pair counted twice]
    
    # But we must be careful with very small distances
    # Use math.fsum for precision as specified
    # Extract upper triangle
    iu = np.triu_indices(n, k=1)
    d_sq_upper = dist_sq[iu]
    
    # Compute log(1/d) = -0.5 * log(d_sq)
    # For numerical stability, clip d_sq to a minimum
    d_sq_upper = np.maximum(d_sq_upper, 1e-30)
    terms = -0.5 * np.log(d_sq_upper)
    
    # Use math.fsum for high precision summation
    return math.fsum(terms)


def _project_to_sphere(points_flat):
    """Project flat array to unit sphere points."""
    points = points_flat.reshape(-1, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    # Start with a good initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Flatten for optimization
    x0_flat = x0.flatten()
    
    # Use L-BFGS-B optimizer with many iterations
    # The energy function is smooth except when points coincide
    
    def objective(x_flat):
        pts = _project_to_sphere(x_flat)
        return _log_energy(pts)
    
    # Try multiple restarts with different initial conditions
    best_x = x0_flat
    best_e = objective(x0_flat)
    
    # Also try a few random starts
    rng = np.random.RandomState(42)
    for restart in range(3):
        # Random points on sphere
        rand_pts = rng.randn(N, 3)
        rand_pts /= np.linalg.norm(rand_pts, axis=1, keepdims=True)
        rand_flat = rand_pts.flatten()
        
        res = minimize(
            objective,
            rand_flat,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
        )
        if res.success or res.fun < best_e:
            best_x = res.x
            best_e = res.fun
    
    # Refine the best solution with more iterations
    res = minimize(
        objective,
        best_x,
        method='L-BFGS-B',
        options={'maxiter': 20000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
    )
    if res.fun < best_e:
        best_x = res.x
        best_e = res.fun
    
    # Final projection to ensure unit sphere
    final_pts = _project_to_sphere(best_x)
    
    # Verify validity
    assert final_pts.shape == (N, 3)
    assert np.all(np.isfinite(final_pts))
    norms = np.linalg.norm(final_pts, axis=1)
    assert np.all(norms > 1e-9)
    
    # Check no two points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(final_pts[i] - final_pts[j])
            if d < 1e-9:
                # Nudge apart
                final_pts[j] += 1e-8 * (final_pts[j] - final_pts[i]) / max(d, 1e-15)
                final_pts[j] /= np.linalg.norm(final_pts[j])
    
    return final_pts
# EVOLVE-BLOCK-END
