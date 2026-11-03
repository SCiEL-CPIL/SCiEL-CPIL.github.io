"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

_DEADLINE = 150.0  # seconds
_START_TIME = None


def _energy_and_grad(pts):
    """
    Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) and its gradient.
    
    Uses dot-product based computation for speed.
    Energy = -sum_{i<j} log(|x_i - x_j|)
    Since |x_i - x_j|^2 = 2 - 2*x_i.x_j for unit vectors,
    log(1/|x_i - x_j|) = -0.5 * log(2 - 2*x_i.x_j)
    
    Gradient w.r.t. x_i: sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    # pts has shape (N, 3), rows are unit vectors
    # Compute pairwise dot products
    dots = pts @ pts.T  # (N, N)
    
    # Ensure dots are in valid range
    dots = np.clip(dots, -1.0, 1.0 - 1e-15)
    
    # Squared distances: 2 - 2*dots
    sq_dist = 2.0 - 2.0 * dots
    np.fill_diagonal(sq_dist, 1.0)  # avoid log(0) issues on diagonal
    
    # Energy = -0.5 * sum_{i<j} log(sq_dist_ij)
    # We need sum over i<j only
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    sq_dist_upper = sq_dist[iu]
    
    # Filter out any non-positive values (shouldn't happen for valid points)
    mask = sq_dist_upper > 1e-15
    if not np.all(mask):
        sq_dist_upper = sq_dist_upper[mask]
    
    energy = -0.5 * math.fsum(math.log(v) for v in sq_dist_upper)
    
    # Gradient: grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # = sum_{j != i} (x_i - x_j) / (2 - 2*x_i.x_j)
    # 
    # Let's compute this efficiently.
    # For each i, grad_i = sum_j (x_i - x_j) / sq_dist_ij (j != i)
    # = sum_j x_i / sq_dist_ij - sum_j x_j / sq_dist_ij
    # = x_i * sum_j 1/sq_dist_ij - sum_j x_j / sq_dist_ij  (j != i)
    
    # Build 1/sq_dist matrix, with 0 on diagonal
    inv_sq_dist = np.zeros_like(sq_dist)
    np.fill_diagonal(sq_dist, 1.0)
    inv_sq_dist = 1.0 / sq_dist
    np.fill_diagonal(inv_sq_dist, 0.0)
    
    # grad_i = x_i * (sum_j inv_sq_dist_ij) - (pts @ inv_sq_dist)_i
    row_sums = inv_sq_dist.sum(axis=1)
    grad = pts * row_sums[:, np.newaxis] - (pts @ inv_sq_dist)
    
    # The energy is -0.5 * sum log(sq_dist), so the gradient of energy is:
    # dE/dx_i = -0.5 * sum_j (1/sq_dist_ij) * d(sq_dist_ij)/dx_i
    # d(sq_dist_ij)/dx_i = -2*(x_i - x_j)
    # So dE/dx_i = -0.5 * sum_j (1/sq_dist_ij) * (-2)*(x_i - x_j)
    #             = sum_j (x_i - x_j) / sq_dist_ij
    #             = grad as computed above
    # 
    # Wait, let me re-derive. E = -0.5 * sum_{i<j} log(2 - 2*x_i.x_j)
    # dE/dx_k for a given pair (i,j): 
    #   if k == i: -0.5 * (1/(2-2*x_i.x_j)) * (-2*x_j) = x_j / (2-2*x_i.x_j)
    #   if k == j: -0.5 * (1/(2-2*x_i.x_j)) * (-2*x_i) = x_i / (2-2*x_i.x_j)
    # 
    # So for point k: grad_k = sum_{j != k} x_j / (2 - 2*x_k.x_j) ... no wait
    # 
    # Let me be more careful. E = -0.5 * sum_{i<j} log(2 - 2 x_i · x_j)
    # 
    # ∂E/∂x_k = -0.5 * sum_{i<j, i or j = k} (1/(2-2 x_i·x_j)) * (-2 * ∂(x_i·x_j)/∂x_k)
    # 
    # If k = i (and i < j): term is -0.5 * (1/(2-2 x_i·x_j)) * (-2 * x_j) = x_j / (2-2 x_i·x_j)
    # If k = j (and i < j): term is -0.5 * (1/(2-2 x_i·x_j)) * (-2 * x_i) = x_i / (2-2 x_i·x_j)
    # 
    # So grad_k = sum_{j: j<k} x_j / (2-2 x_k·x_j) + sum_{j: j>k} x_j / (2-2 x_k·x_j)
    #           = sum_{j != k} x_j / (2 - 2 x_k·x_j)
    # 
    # Hmm, that's different from what I had before. Let me check with a simple case.
    # 
    # Actually wait. Let me reconsider. The standard result for log energy on sphere:
    # E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|x_i - x_j|
    # 
    # ∂/∂x_k log|x_i - x_j| for i,j != k is 0.
    # For pair (k, j), j != k: ∂/∂x_k log|x_k - x_j| = (x_k - x_j)/|x_k - x_j|^2
    # 
    # So ∂E/∂x_k = -sum_{j != k} (x_k - x_j)/|x_k - x_j|^2
    # 
    # This is the negative of what I computed. Let me verify the sign.
    # E = -sum_{i<j} log|x_i - x_j|
    # ∂E/∂x_k = -sum_{j != k} ∂/∂x_k log|x_k - x_j|
    #          = -sum_{j != k} (x_k - x_j)/|x_k - x_j|^2
    # 
    # So grad_k = -sum_{j != k} (x_k - x_j)/|x_k - x_j|^2
    #           = sum_{j != k} (x_j - x_k)/|x_k - x_j|^2
    
    # Recompute gradient with correct sign
    # grad_k = sum_{j != k} (x_j - x_k) / sq_dist_kj
    #         = sum_j x_j / sq_dist_kj - x_k * sum_j 1/sq_dist_kj  (j != k)
    #         = (inv_sq_dist @ pts)_k - x_k * row_sums_k
    
    grad = (inv_sq_dist @ pts) - pts * row_sums[:, np.newaxis]
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_pts = None
    best_energy = float('inf')
    
    def _update_best(pts):
        nonlocal best_pts, best_energy
        pts = _project_to_sphere(pts)
        # Check for close points
        dots = pts @ pts.T
        np.fill_diagonal(dots, -2.0)
        min_dist_sq = 2.0 - 2.0 * dots.max()
        if min_dist_sq < 1e-18:
            return
        e, _ = _energy_and_grad(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
    
    # Try multiple random initializations and optimize each
    rng = np.random.default_rng(42)
    
    n_restarts = 0
    max_restarts = 50
    
    while True:
        elapsed = time.time() - _START_TIME
        # Need about 20s per optimization, leave 10s margin
        if elapsed > _DEADLINE - 25.0:
            break
        
        # Generate random initial points
        pts = rng.standard_normal((N, 3))
        pts = _project_to_sphere(pts)
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                _energy_and_grad,
                pts,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            _update_best(result.x)
        except Exception:
            pass
        
        n_restarts += 1
        if n_restarts >= max_restarts:
            break
    
    # Also try the Fibonacci spiral as a starting point
    if time.time() - _START_TIME < _DEADLINE - 25.0:
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        try:
            result = minimize(
                _energy_and_grad,
                pts_fib,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            _update_best(result.x)
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Final projection
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    dots = best_pts @ best_pts.T
    np.fill_diagonal(dots, -2.0)
    max_dot = dots.max()
    if max_dot > 1.0 - 1e-9:
        # Perturb slightly
        best_pts += rng.standard_normal(best_pts.shape) * 1e-5
        best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
