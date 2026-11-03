"""New program: Riemannian Gradient Descent with Multiple Restarts for Logarithmic Energy Minimization."""
import numpy as np
import math

N = 100

# EVOLVE-BLOCK-START
def _log_energy(pts):
    """Compute sum_{i<j} log(1/||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)."""
    # pts shape (N, 3)
    # Use broadcasting for vectorized distance calculation
    # dist^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i . x_j = 2 - 2 x_i . x_j
    # Since ||x_i|| = 1
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    # Avoid log(0) and negative due to numerical error
    dist_sq = np.maximum(dist_sq, 1e-15)
    # We only need i < j
    iu = np.triu_indices(N, k=1)
    d_sq = dist_sq[iu]
    # log(1/d) = -log(d) = -0.5 log(d^2)
    return -0.5 * np.sum(np.log(d_sq))

def _log_energy_grad(pts):
    """Compute the Riemannian gradient of the log energy on the unit sphere.
    
    The gradient of E = -sum_{i<j} log(||x_i - x_j||) w.r.t x_i is:
    grad_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    
    This is the Euclidean gradient. The Riemannian gradient is the projection
    of this vector onto the tangent space at x_i:
    grad_riem_i = grad_i - (grad_i . x_i) x_i
    """
    # Compute all pairwise squared distances
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    # Set diagonal to 1 to avoid division by zero, we'll mask it later
    np.fill_diagonal(dist_sq, 1.0)
    
    # Compute weights w_ij = 1 / dist_sq_ij
    # Note: For i=j, w_ii is 1/1 = 1, but we will zero out the diagonal contribution
    w = 1.0 / dist_sq
    
    # The Euclidean gradient for point i is sum_{j != i} w_ij (x_i - x_j)
    # = x_i * sum_{j != i} w_ij - sum_{j != i} w_ij x_j
    
    # sum_{j != i} w_ij is the row sum of w, minus the diagonal element (which is 1)
    row_sums = np.sum(w, axis=1)
    # Subtract the self-term (w_ii = 1)
    row_sums_no_self = row_sums - 1.0
    
    # sum_{j != i} w_ij x_j is the matrix product W @ X, where W has 0 on diagonal
    W_no_diag = w.copy()
    np.fill_diagonal(W_no_diag, 0.0)
    weighted_sum = W_no_diag @ pts
    
    # Euclidean Gradient
    grad_euclid = row_sums_no_self[:, np.newaxis] * pts - weighted_sum
    
    # Project to Tangent Space
    # grad_riem = grad_euclid - (grad_euclid . x_i) x_i
    dot_prod = np.sum(grad_euclid * pts, axis=1)
    grad_riem = grad_euclid - dot_prod[:, np.newaxis] * pts
    
    return grad_riem

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _riemannian_step(pts, grad, step_size):
    """Perform a Riemannian gradient step.
    
    Move along the geodesic in the direction of -grad.
    x_new = cos(theta) x + sin(theta) * (-grad / ||grad||)
    where theta = step_size * ||grad|| (approximation for small steps)
    Or more simply, for small steps:
    x_new = (x - step_size * grad) / ||x - step_size * grad||
    """
    # Simple retraction: x_new = x - step_size * grad, then normalize
    x_new = pts - step_size * grad
    return _project_to_sphere(x_new)

def _minimize_local(pts0, max_iter=5000, tol=1e-10):
    """Riemannian Gradient Descent with backtracking line search."""
    pts = pts0.copy()
    energy = _log_energy(pts)
    grad = _log_energy_grad(pts)
    
    # Initial step size
    step_size = 1e-2
    
    for _ in range(max_iter):
        # Check convergence
        if np.linalg.norm(grad) < tol:
            break
        
        # Backtracking line search
        alpha = step_size
        decrease = False
        for _ in range(20):
            pts_new = _riemannian_step(pts, grad, alpha)
            energy_new = _log_energy(pts_new)
            if energy_new < energy:
                decrease = True
                break
            alpha *= 0.5
        
        if not decrease:
            # If we couldn't find a decrease, reduce step size permanently
            step_size *= 0.1
            if step_size < 1e-12:
                break
            continue
            
        # Accept step
        pts = pts_new
        energy = energy_new
        grad = _log_energy_grad(pts)
        
        # Increase step size slightly if we took a full step
        if alpha == step_size:
            step_size = min(step_size * 1.5, 1e-1)
            
    return pts, energy

def _fibonacci_sphere(n):
    """Generate a Fibonacci sphere distribution."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=0):
    """Generate random points on the sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3))
    return _project_to_sphere(pts)

def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    
    # List of initial configurations
    inits = []
    
    # 1. Fibonacci sphere
    inits.append(_fibonacci_sphere(N))
    
    # 2. Random starts with different seeds
    for seed in [1, 2, 3, 4, 5]:
        inits.append(_random_sphere(N, seed=seed))
        
    # 3. Grid-based perturbation
    # Create a grid on the sphere
    # This is a bit complex, so we'll stick to random and Fibonacci for now,
    # but add a "structured" random start
    rng = np.random.default_rng(42)
    inits.append(_project_to_sphere(rng.normal(size=(N, 3))))
    
    best_pts = None
    best_energy = float('inf')
    
    # Run local minimization from each start
    for i, init_pts in enumerate(inits):
        try:
            pts, energy = _minimize_local(init_pts, max_iter=2000, tol=1e-12)
            if energy < best_energy:
                best_energy = energy
                best_pts = pts
        except Exception as e:
            pass
            
    # Refine the best result with a tighter tolerance and more iterations
    if best_pts is not None:
        try:
            pts, energy = _minimize_local(best_pts, max_iter=5000, tol=1e-14)
            if energy < best_energy:
                best_energy = energy
                best_pts = pts
        except Exception:
            pass
            
    # Final polish: very small steps to ensure high precision
    # Perform a few iterations with a very small fixed step size
    pts = best_pts
    for _ in range(100):
        grad = _log_energy_grad(pts)
        if np.linalg.norm(grad) < 1e-15:
            break
        step = 1e-5
        pts_new = _riemannian_step(pts, grad, step)
        energy_new = _log_energy(pts_new)
        energy_old = _log_energy(pts)
        if energy_new < energy_old:
            pts = pts_new
        else:
            break
            
    best_pts = _project_to_sphere(pts)
    
    # Check for close points
    dists = np.linalg.norm(best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :], axis=2)
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dists[iu])
    
    if min_dist < 1e-9:
        # Perturb
        rng = np.random.default_rng(123)
        best_pts = best_pts + rng.normal(0, 1e-6, best_pts.shape)
        best_pts = _project_to_sphere(best_pts)
        # Re-minimize
        pts, energy = _minimize_local(best_pts, max_iter=1000, tol=1e-12)
        best_pts = pts
        
    return best_pts
# EVOLVE-BLOCK-END
