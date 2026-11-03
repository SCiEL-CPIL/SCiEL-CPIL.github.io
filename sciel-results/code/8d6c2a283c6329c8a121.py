"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, LinearConstraint
import warnings

def _energy_and_grad(pts):
    """Compute log energy and gradient for points on unit sphere.
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    Gradient on sphere (tangent): for each i, grad_i = sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2
    but projected to tangent plane: grad_i -= (grad_i . x_i) x_i
    """
    n = pts.shape[0]
    # Compute pairwise distances
    # pts: (n, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(dist_sq, 1.0)  # avoid log(0)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of log(1/d_ij) = -sum_{i<j} log(d_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each point i, the force from point j is (x_i - x_j)/|x_i - x_j|^2
    # This is the gradient of -log|d_ij| with respect to x_i
    # grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # But we need to project to tangent plane of sphere at x_i
    
    # Compute forces: F[i] = sum_j (x_i - x_j)/|d_ij|^2
    # dist_sq[i,j] = |x_i - x_j|^2
    inv_dist_sq = 1.0 / dist_sq  # (n, n)
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # F[i] = sum_j inv_dist_sq[i,j] * (x_i - x_j)
    # = sum_j inv_dist_sq[i,j] * x_i - sum_j inv_dist_sq[i,j] * x_j
    # = x_i * sum_j inv_dist_sq[i,j] - sum_j inv_dist_sq[i,j] * x_j
    
    row_sum = np.sum(inv_dist_sq, axis=1)  # (n,)
    F = row_sum[:, np.newaxis] * pts - inv_dist_sq @ pts  # (n, 3)
    
    # Project to tangent plane: F_tangent[i] = F[i] - (F[i] . x_i) x_i
    dot = np.sum(F * pts, axis=1)  # (n,)
    F_tangent = F - dot[:, np.newaxis] * pts
    
    return energy, F_tangent


def _energy_fast(pts):
    """Just the energy, for quick evaluation."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu]))


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms


def _lbfgs_on_sphere(pts0, max_iter=2000, ftol=1e-14, gtol=1e-10):
    """Run L-BFGS-B on the sphere using a parameterization that stays on the sphere.
    We use the fact that for small moves, we can work in tangent space and re-project.
    Actually, let's use a simpler approach: optimize in R^3 with a penalty, or use
    the fact that scipy's L-BFGS-B can handle equality constraints via penalty.
    
    Better approach: use the tangent space parameterization.
    For each point x_i on sphere, a tangent vector v_i satisfies v_i . x_i = 0.
    A move is x_i_new = (x_i + alpha * v_i) / |x_i + alpha * v_i|
    
    We'll define a function that takes a flat vector of tangent components and
    evaluates the energy.
    """
    n = pts0.shape[0]
    
    def _tangent_to_sphere(pts, tvec):
        """Given points on sphere and tangent vectors (flat array of size 3n),
        produce new points on sphere via exponential map (approximate for small steps)."""
        tv = tvec.reshape(n, 3)
        # Project tv to tangent space (should already be tangent, but ensure)
        dots = np.sum(tv * pts, axis=1)
        tv = tv - dots[:, np.newaxis] * pts
        # New points: pts + tv, then normalize
        new_pts = pts + tv
        new_pts = _normalize(new_pts)
        return new_pts
    
    def _obj_and_grad(tvec):
        pts = _tangent_to_sphere(pts0, tvec)
        e, g = _energy_and_grad(pts)
        # Chain rule: dE/dtvec = sum_i (dE/dx_i) . (dx_i/dtvec_i)
        # dx_i/dtvec_i is the projection matrix: I - x_i x_i^T
        # So grad_tvec_i = (I - x_i x_i^T) g_i = g_i (already projected)
        # But we need to account for the normalization in _tangent_to_sphere.
        # For small tvec, dx_i/dtvec_i ≈ I - x_i x_i^T
        # So grad_tvec ≈ g (the tangent gradient)
        return e, g.reshape(-1)
    
    res = minimize(_obj_and_grad, np.zeros(3 * n), method='L-BFGS-B',
                   jac=True, options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol,
                                       'maxls': 50, 'maxfun': 5000})
    
    pts_opt = _tangent_to_sphere(pts0, res.x)
    return pts_opt, res.fun, res


def _local_optimize(pts, time_budget=30.0):
    """Local optimization with L-BFGS on the sphere, with time budget."""
    start = time.time()
    pts = _normalize(pts.copy())
    best_pts = pts.copy()
    best_energy = _energy_fast(best_pts)
    
    # Run L-BFGS with tight tolerances
    try:
        pts_opt, e_opt, res = _lbfgs_on_sphere(pts, max_iter=3000, ftol=1e-15, gtol=1e-12)
        if e_opt < best_energy:
            best_pts = pts_opt
            best_energy = e_opt
    except Exception:
        pass
    
    elapsed = time.time() - start
    if elapsed > time_budget:
        return best_pts, best_energy
    
    # Polish: run again from the result with even tighter tolerances
    try:
        pts_opt2, e_opt2, res2 = _lbfgs_on_sphere(best_pts, max_iter=5000, ftol=1e-16, gtol=1e-13)
        if e_opt2 < best_energy:
            best_pts = pts_opt2
            best_energy = e_opt2
    except Exception:
        pass
    
    return best_pts, best_energy


def _fibonacci_sphere(n):
    """Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)


def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    rng = np.random.default_rng(42)
    
    # Start with Fibonacci spiral
    best_pts = _fibonacci_sphere(N)
    best_energy = _energy_fast(best_pts)
    
    # First, do a local optimization from the Fibonacci start
    best_pts, best_energy = _local_optimize(best_pts, time_budget=20.0)
    
    # Now try multiple random starts with local optimization
    n_random_starts = 10
    for k in range(n_random_starts):
        if time.time() > deadline - 15.0:
            break
        
        # Generate a random start, but perturb the Fibonacci points to avoid
        # getting stuck in the same local minimum
        pts_rand = _random_sphere(N, rng)
        # Mix with Fibonacci to create diverse starts
        mix = rng.uniform(0.3, 0.7)
        pts_mix = mix * pts_rand + (1 - mix) * best_pts
        pts_mix = _normalize(pts_mix)
        
        # Local optimize
        pts_opt, e_opt = _local_optimize(pts_mix, time_budget=10.0)
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt
    
    # Final polish with very tight tolerances
    if time.time() < deadline - 5.0:
        best_pts, best_energy = _local_optimize(best_pts, time_budget=15.0)
    
    # Ensure points are well-separated and on the sphere
    best_pts = _normalize(best_pts)
    
    # Validate: check no points closer than 1e-9
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    min_dist = np.sqrt(np.min(dist_sq))
    if min_dist < 1e-9:
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N)
    
    # Ensure all finite
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
