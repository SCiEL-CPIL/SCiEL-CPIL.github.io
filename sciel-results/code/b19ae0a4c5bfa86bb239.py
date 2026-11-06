"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

N = 150
TIME_LIMIT = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _log_energy(points):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dists[iu])
    return -math.fsum(log_dists.tolist())

def _fibonacci_sphere(n, seed=0.0):
    """Generate n points on the sphere using a Fibonacci spiral with an offset."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + seed)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng=None):
    """Generate n random points on the unit sphere."""
    if rng is None:
        rng = np.random.default_rng()
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _energy_and_grad_riemannian(points):
    """
    Compute energy and Riemannian gradient (tangential component) on the sphere.
    Energy: E = -sum_{i<j} log(|x_i - x_j|)
    Euclidean Gradient: grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    Riemannian Gradient: proj_{T_x S^2} (grad_i) = grad_i - (grad_i . x_i) x_i
    """
    n = points.shape[0]
    # Pairwise differences
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, 3)
    dists_sq = np.sum(diff * diff, axis=-1)  # (n, n)
    np.fill_diagonal(dists_sq, 1.0)
    
    # Energy
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(np.sqrt(dists_sq[iu]))
    energy = -math.fsum(log_dists.tolist())
    
    # Euclidean Gradient
    # grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # Note: dists_sq has 1.0 on diagonal, so 1/dists_sq is 0 effectively for j=i if we mask, 
    # but here we sum over all j. The term j=i is (x_i-x_i)/1 = 0.
    inv_dists_sq = 1.0 / dists_sq  # (n, n)
    grad_euclidean = np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)  # (n, 3)
    
    # Project to tangent space
    # grad_riemannian_i = grad_euclidean_i - (grad_euclidean_i . x_i) x_i
    dot_products = np.sum(grad_euclidean * points, axis=1, keepdims=True)  # (n, 1)
    grad_riemannian = grad_euclidean - dot_products * points  # (n, 3)
    
    return energy, grad_riemannian

def _exp_map(points, tangent_vecs, step_size=1.0):
    """
    Exponential map on the sphere.
    moves points along the geodesic defined by tangent_vecs scaled by step_size.
    x_new = sin(theta)/||v|| * v + cos(theta) * x  where v = step_size * tangent_vec
    """
    v = tangent_vecs * step_size
    norms_v = np.linalg.norm(v, axis=1, keepdims=True)
    
    # Handle zero vectors
    mask = norms_v > 1e-12
    norms_v_safe = np.where(mask, norms_v, 1.0)
    
    theta = norms_v_safe
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # x_new = x * cos(theta) + (v / ||v||) * sin(theta)
    # v / ||v|| is the direction
    dir_v = v / norms_v_safe
    
    new_points = points * cos_theta + dir_v * sin_theta
    
    # For zero vectors, new_points = points * 1 + 0 = points
    # But we must ensure normalization just in case of numerical drift
    return _normalize(new_points)

def _riemannian_lbfgs(points, maxiter=100, tol=1e-8):
    """
    Perform Riemannian L-BFGS optimization.
    We use scipy's L-BFGS-B on the flat representation but with a custom 
    retraction (normalization) and gradient (tangential).
    Actually, to be strictly Riemannian, we should update on the manifold.
    Since scipy.optimize.minimize works on flat vectors, we can:
    1. Map flat -> sphere (normalize)
    2. Compute Riemannian gradient
    3. Map Riemannian gradient back to flat gradient for L-BFGS? 
       No, L-BFGS assumes Euclidean metric.
    
    Better approach: Implement a simple Riemannian gradient descent with momentum,
    or use a custom loop that projects steps.
    
    Given the constraints, let's implement a custom Riemannian Gradient Descent 
    with adaptive step size (Barzilai-Borwein or simple backtracking).
    """
    current_pts = points.copy()
    current_energy, current_grad = _energy_and_grad_riemannian(current_pts)
    
    step_size = 0.01
    momentum = 0.9
    v = np.zeros_like(current_pts)
    
    for i in range(maxiter):
        # Update velocity (momentum)
        v = momentum * v - step_size * current_grad
        # Project velocity to tangent space to ensure we stay on manifold
        # v_tangent = v - (v . x) x
        dot_v_x = np.sum(v * current_pts, axis=1, keepdims=True)
        v_tangent = v - dot_v_x * current_pts
        
        # Take step using exponential map
        next_pts = _exp_map(current_pts, v_tangent, step_size=1.0)
        
        next_energy, next_grad = _energy_and_grad_riemannian(next_pts)
        
        if next_energy < current_energy:
            current_pts = next_pts
            current_energy = next_energy
            current_grad = next_grad
            # Slowly decay step size or keep it
            step_size = max(0.001, step_size * 0.99)
        else:
            # Backtrack
            step_size *= 0.5
            if step_size < 1e-6:
                break
            
    return current_pts, current_energy

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_energy = float('inf')
    best_points = _fibonacci_sphere(N)
    
    rng = np.random.default_rng(42)

    # 1. Evaluate initial Fibonacci
    try:
        e = _log_energy(best_points)
        if e < best_energy:
            best_energy = e
    except Exception:
        pass

    # 2. Refine Fibonacci start with Riemannian GD
    try:
        refined_pts, e = _riemannian_lbfgs(best_points, maxiter=200)
        if e < best_energy:
            best_energy = e
            best_points = refined_pts
    except Exception:
        pass

    # 3. Simulated Annealing / Basin Hopping with Riemannian refinement
    #    We perform a random walk on the sphere, accepting worse moves with probability exp(-dE/T)
    #    Periodically, we refine the current state with Riemannian GD.
    
    current_pts = best_points.copy()
    current_energy = best_energy
    T = 1.0  # Temperature
    cooling_rate = 0.99
    
    # Pre-allocate for speed
    num_sa_steps = 5000
    
    for step in range(num_sa_steps):
        if time.time() > deadline - 30:
            break
            
        # Perturb current points
        # Add random noise in tangent space
        noise = rng.standard_normal((N, 3))
        # Project noise to tangent space
        dot_n_x = np.sum(noise * current_pts, axis=1, keepdims=True)
        noise_tangent = noise - dot_n_x * current_pts
        noise_tangent *= 0.1  # Scale of perturbation
        
        next_pts = _exp_map(current_pts, noise_tangent, step_size=1.0)
        next_energy = _log_energy(next_pts)
        
        delta_e = next_energy - current_energy
        
        # Accept if better, or with probability exp(-delta_e / T)
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            current_pts = next_pts
            current_energy = next_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_pts.copy()
        
        # Cool down
        T *= cooling_rate
        
        # Every 50 steps, do a quick Riemannian refinement to escape local minima
        if step % 50 == 0 and time.time() < deadline - 40:
            try:
                refined_pts, e = _riemannian_lbfgs(current_pts, maxiter=50)
                if e < current_energy:
                    current_pts = refined_pts
                    current_energy = e
                    if e < best_energy:
                        best_energy = e
                        best_points = current_pts.copy()
            except Exception:
                pass

    # 4. Final polish
    if time.time() < deadline - 5:
        try:
            final_pts, e = _riemannian_lbfgs(best_points, maxiter=500)
            if e < best_energy:
                best_energy = e
                best_points = final_pts
        except Exception:
            pass

    return _normalize(best_points)

# EVOLVE-BLOCK-END
