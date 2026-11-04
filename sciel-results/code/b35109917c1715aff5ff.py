"""Optimize 150 points on unit sphere to minimize logarithmic energy."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient (fast vectorized)."""
    n = len(pts)
    # Use cdist for speed, but manual diff is also fine for N=150
    # cdist is often faster for large N, but for N=150, manual might be competitive
    # Let's stick to the manual diff method which is very clean for N=150
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    
    # Avoid log(0) and division by zero
    np.fill_diagonal(dist2, 1.0)
    
    iu = np.triu_indices(n, k=1)
    # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij) = -0.5 * sum_{i!=j} 0.5 log(dist2_ij)
    # Wait, log(1/d) = -log(d).
    # d = sqrt(dist2). log(d) = 0.5 log(dist2).
    # So term is -0.5 log(dist2).
    # Sum over i<j is half the sum over i!=j.
    # So Energy = sum_{i<j} -log(d_ij) = -0.5 * sum_{i!=j} log(d_ij) = -0.5 * sum_{i!=j} 0.5 log(dist2_ij) = -0.25 * sum_{i!=j} log(dist2_ij).
    # Alternatively: Energy = -sum_{i<j} log(d_ij).
    # Let's just sum the upper triangle of -log(d_ij).
    # d_ij = sqrt(dist2_ij).
    # log(d_ij) = 0.5 * log(dist2_ij).
    # So term is -0.5 * log(dist2_ij).
    
    log_dist2_upper = np.log(dist2[iu])
    energy = -0.5 * np.sum(log_dist2_upper)

    # Gradient:
    # E = -0.5 * sum_{i<j} log(dist2_ij)  (This is equivalent to -sum_{i<j} log(d_ij))
    # dE/dx_i = -0.5 * sum_{j>i} (1/dist2_ij) * 2*(x_i - x_j)  (from j>i terms)
    #          + (-0.5) * sum_{j<i} (1/dist2_ij) * 2*(x_i - x_j) (from i>j terms, where derivative wrt x_i of log(dist2_ji) is 2(x_i-x_j)/dist2)
    # Actually, simpler:
    # E = - sum_{i<j} log(||x_i - x_j||)
    # dE/dx_i = - sum_{j!=i} (1/||x_i - x_j||) * (x_i - x_j)/||x_i - x_j||
    #          = - sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    #          = sum_{j!=i} (x_j - x_i) / dist2_ij
    
    inv_dist2 = np.zeros_like(dist2)
    mask = dist2 > 1e-30
    inv_dist2[mask] = 1.0 / dist2[mask]
    np.fill_diagonal(inv_dist2, 0.0)

    # grad_i = sum_j (x_j - x_i) * inv_dist2[i,j]
    #        = (sum_j x_j * inv_dist2[i,j]) - x_i * (sum_j inv_dist2[i,j])
    weighted_sum = inv_dist2 @ pts
    row_sums = np.sum(inv_dist2, axis=1)
    grad = weighted_sum - pts * row_sums[:, np.newaxis]

    return energy, grad


def _energy_only(pts):
    """Energy only, for fast inner loops."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    iu = np.triu_indices(n, k=1)
    return -0.5 * np.sum(np.log(dist2[iu]))


def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _tangent_project(pts, grad):
    """Project gradient onto tangent space of sphere at each point."""
    g_dot_n = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - g_dot_n * pts


def _fibonacci_sphere(n, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
        offset = rng.uniform(0, 1)
    else:
        offset = 0.0
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def _random_sphere(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _project_to_sphere(pts)


def _lbfgs_optimize(pts, time_limit=5.0, ftol=1e-12, gtol=1e-10, maxiter=2000):
    """L-BFGS-B optimization with time limit."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())

    def objective_flat(x):
        if time.time() - start_time > time_limit:
            return np.inf
        p = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _energy_and_grad(p)
        return e

    def grad_flat(x):
        if time.time() - start_time > time_limit:
            return np.zeros(len(x))
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _energy_and_grad(p)
        g_tangent = _tangent_project(p, g)
        return g_tangent.flatten()

    x0 = pts.flatten()

    try:
        result = minimize(
            objective_flat,
            x0,
            jac=grad_flat,
            method='L-BFGS-B',
            options={
                'maxiter': maxiter,
                'maxfun': 5000,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': 50
            }
        )
    except Exception:
        return pts, _energy_only(pts)

    final_pts = _project_to_sphere(result.x.reshape(-1, 3))
    energy, _ = _energy_and_grad(final_pts)
    if not np.all(np.isfinite(energy)):
        return pts, _energy_only(pts)
    return final_pts, energy


def _riemannian_gradient_flow(pts, time_limit=10.0, max_steps=5000):
    """Riemannian gradient flow with Armijo line search."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)
    
    current_energy, _ = _energy_and_grad(pts)
    step_size = 0.1  # Initial step size
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        _, grad = _energy_and_grad(pts)
        grad_tangent = _tangent_project(pts, grad)
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < 1e-10:
            break
            
        # Armijo Line Search
        alpha = step_size
        c1 = 1e-4  # Armijo constant
        max_ls = 20
        
        accepted = False
        for _ in range(max_ls):
            new_pts = pts - alpha * grad_tangent
            new_pts = _project_to_sphere(new_pts)
            new_energy, _ = _energy_and_grad(new_pts)
            
            # Armijo condition: f(x + alpha*d) <= f(x) + c1 * alpha * grad^T * d
            # Here d = -grad_tangent, so grad^T * d = -||grad_tangent||^2
            # Condition: new_energy <= current_energy - c1 * alpha * grad_norm^2
            if new_energy <= current_energy - c1 * alpha * grad_norm * grad_norm:
                pts = new_pts
                current_energy = new_energy
                accepted = True
                break
            else:
                alpha *= 0.5
                
        if not accepted:
            # If line search fails, try a very small step or break
            # If we can't find a descent direction, we might be at a minimum or stuck
            break
            
    final_pts = _project_to_sphere(pts)
    energy, _ = _energy_and_grad(final_pts)
    return final_pts, energy


def _basin_hopping(pts0, time_limit=60.0, seed=42):
    """
    Basin Hopping algorithm.
    1. Perturb current configuration.
    2. Polish with L-BFGS.
    3. Accept if better.
    """
    start_time = time.time()
    current = _project_to_sphere(pts0.copy())
    current_energy = _energy_only(current)
    best = current.copy()
    best_energy = current_energy
    
    T = 0.5  # Initial temperature for perturbation scale
    T_min = 0.05
    cooling = 0.98
    
    rng = np.random.RandomState(seed)
    
    while time.time() - start_time < time_limit:
        # Perturb
        # Move a random subset of points
        n_perturb = rng.randint(5, 15)
        idxs = rng.choice(N, n_perturb, replace=False)
        
        new_pts = current.copy()
        for idx in idxs:
            # Generate random tangent vector
            rand_vec = rng.randn(3)
            tangent = rand_vec - np.dot(rand_vec, new_pts[idx]) * new_pts[idx]
            norm = np.linalg.norm(tangent)
            if norm < 1e-15:
                continue
            tangent /= norm
            new_pts[idx] = new_pts[idx] + T * tangent
        
        new_pts = _project_to_sphere(new_pts)
        
        # Polish
        # Use a time limit for the polish to ensure we don't exceed the total budget
        polish_time = min(2.0, time_limit - (time.time() - start_time) - 1.0)
        if polish_time < 0.5:
            break
            
        polished_pts, polished_energy = _lbfgs_optimize(new_pts, time_limit=polish_time, maxiter=100)
        
        # Accept if better
        if polished_energy < current_energy:
            current = polished_pts
            current_energy = polished_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best = current.copy()
                
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin from 150s limit

    rng = np.random.default_rng(42)

    # Generate diverse starting points
    starts = []

    # 1. Fibonacci spiral
    starts.append(_fibonacci_sphere(N))

    # 2. Perturbed Fibonacci (multiple seeds)
    for seed in [1, 7, 13, 42, 123]:
        fib = _fibonacci_sphere(N, seed=seed)
        noise = 0.1 * rng.standard_normal((N, 3))
        perturbed = fib + noise
        starts.append(_project_to_sphere(perturbed))

    # 3. Random starts
    for _ in range(3):
        starts.append(_random_sphere(N, rng))

    best_pts = None
    best_energy = np.inf

    # Phase 1: Optimize each start with L-BFGS-B, keep best
    # Allocate ~15s
    phase1_deadline = start_time + 15.0
    for i, start_pts in enumerate(starts):
        if time.time() > phase1_deadline:
            break
        candidate_pts, candidate_energy = _lbfgs_optimize(start_pts, time_limit=2.0, maxiter=100)
        if candidate_energy < best_energy:
            best_energy = candidate_energy
            best_pts = candidate_pts

    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
        best_energy = _energy_only(best_pts)

    # Phase 2: Basin Hopping
    # Allocate ~60s
    phase2_deadline = start_time + 75.0
    if time.time() < phase2_deadline:
        time_limit = phase2_deadline - time.time()
        candidate_pts, candidate_energy = _basin_hopping(best_pts, time_limit=time_limit, seed=42)
        if candidate_energy < best_energy:
            best_pts = candidate_pts
            best_energy = candidate_energy

    # Phase 3: Riemannian gradient flow with line search
    # Allocate ~20s
    phase3_deadline = start_time + 95.0
    if time.time() < phase3_deadline:
        time_limit = phase3_deadline - time.time()
        candidate_pts, candidate_energy = _riemannian_gradient_flow(best_pts, time_limit=time_limit)
        if candidate_energy < best_energy:
            best_pts = candidate_pts
            best_energy = candidate_energy

    # Phase 4: Another round of Basin Hopping with different seed
    # Allocate ~15s
    phase4_deadline = start_time + 110.0
    if time.time() < phase4_deadline:
        time_limit = phase4_deadline - time.time()
        candidate_pts, candidate_energy = _basin_hopping(best_pts, time_limit=time_limit, seed=123)
        if candidate_energy < best_energy:
            best_pts = candidate_pts
            best_energy = candidate_energy

    # Phase 5: Final high-precision polish with L-BFGS
    # Allocate ~10s
    phase5_deadline = deadline - 2.0
    if time.time() < phase5_deadline:
        time_limit = phase5_deadline - time.time()
        candidate_pts, candidate_energy = _lbfgs_optimize(best_pts, time_limit=time_limit, ftol=1e-15, gtol=1e-12, maxiter=5000)
        if candidate_energy < best_energy:
            best_pts = candidate_pts
            best_energy = candidate_energy

    best_pts = _project_to_sphere(best_pts)

    # Validate: check for very close points
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)

    if min_dist < 1e-9:
        return _fibonacci_sphere(N)

    return best_pts
# EVOLVE-BLOCK-END
