"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
TIME_LIMIT = 115.0  # safety margin before 120s limit

# EVOLVE-BLOCK-START

def _pairwise_distances_sq(pts):
    """Compute squared pairwise chordal distances (N, N)."""
    # |xi - xj|^2 = 2 - 2*xi.dot(xj)
    dot = pts @ pts.T
    sq_d = 2.0 - 2.0 * dot
    np.fill_diagonal(sq_d, 0.0)
    # Clamp to avoid log(0) or negative values from numerical error
    sq_d = np.maximum(sq_d, 1e-18)
    return sq_d


def _log_energy_and_grad(pts):
    """
    Compute logarithmic energy sum_{i<j} log(1/|xi-xj|) and its gradient.
    pts: (N, 3) array on unit sphere.
    Returns (energy, grad) where grad is (N, 3).
    """
    n = pts.shape[0]
    sq_d = _pairwise_distances_sq(pts)
    d = np.sqrt(sq_d)
    
    # Energy: sum over i<j of log(1/d_ij) = -sum_{i<j} log(d_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    log_d = np.log(d[iu])
    energy = -np.sum(log_d)
    
    # Gradient of log(1/d_ij) w.r.t. xi = -(xi - xj) / |xi-xj|^2 * d_ij... 
    # Actually: d/dxi [log(1/|xi-xj|)] = d/dxi [-log|xi-xj|] = -(xi-xj)/|xi-xj|^2
    # So grad_i contribution from pair (i,j) = -(xi-xj)/|xi-xj|^2
    # And grad_j contribution = -(-(xi-xj))/|xi-xj|^2 = (xi-xj)/|xi-xj|^2
    
    # Vectorized: for each pair (i,j), contribution to grad[i] is -(xi-xj)/sq_d_ij
    #               contribution to grad[j] is (xi-xj)/sq_d_ij
    
    grad = np.zeros_like(pts)
    # (i,j) pairs
    xi = pts[iu[0]]
    xj = pts[iu[1]]
    diff = xi - xj  # (n_pairs, 3)
    inv_sq_d = 1.0 / sq_d[iu]  # (n_pairs,)
    
    # grad[i] += -diff / sq_d
    # grad[j] += diff / sq_d
    contrib = diff * inv_sq_d[:, np.newaxis]
    
    np.add.at(grad, iu[0], -contrib)
    np.add.at(grad, iu[1], contrib)
    
    return energy, grad


def _project_to_sphere(pts):
    """Normalize each point to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_spiral(n, seed_offset=0.0):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + seed_offset)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _lbfgs_optimize(pts, max_time):
    """
    Optimize using L-BFGS-B on the sphere via penalty method.
    We parameterize by free 3D coordinates and penalize deviation from unit sphere.
    """
    start_time = time.time()
    n = pts.shape[0]
    
    best_pts = pts.copy()
    best_energy, _ = _log_energy_and_grad(best_pts)
    
    # Use penalty method: minimize E(x) + mu * sum(||xi||^2 - 1)^2
    # Start with moderate penalty
    mu = 100.0
    
    def objective(x):
        pts = x.reshape(n, 3)
        e, g = _log_energy_and_grad(pts)
        # Penalty for non-unit norm
        norms_sq = np.sum(pts * pts, axis=1)
        pen = mu * np.sum((norms_sq - 1.0) ** 2)
        # Gradient of penalty: 2*mu*(norms_sq - 1) * 2*pts = 4*mu*(norms_sq-1)*pts
        pen_grad = 4.0 * mu * (norms_sq - 1.0)[:, np.newaxis] * pts
        return e + pen, g + pen_grad
    
    x0 = pts.flatten()
    
    # Try L-BFGS-B with increasing penalty
    for mu_try in [10.0, 100.0, 1000.0, 10000.0]:
        if time.time() - start_time > max_time - 5:
            break
        
        def obj2(x, mu=mu_try):
            pts = x.reshape(n, 3)
            e, g = _log_energy_and_grad(pts)
            norms_sq = np.sum(pts * pts, axis=1)
            pen = mu * np.sum((norms_sq - 1.0) ** 2)
            pen_grad = 4.0 * mu * (norms_sq - 1.0)[:, np.newaxis] * pts
            return e + pen, g + pen_grad
        
        res = minimize(obj2, x0, method='L-BFGS-B', 
                       options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000})
        
        if time.time() - start_time > max_time - 3:
            break
        
        pts_new = res.x.reshape(n, 3)
        pts_new = _project_to_sphere(pts_new)
        
        e_new, _ = _log_energy_and_grad(pts_new)
        if e_new < best_energy:
            best_energy = e_new
            best_pts = pts_new.copy()
        
        x0 = pts_new.flatten()
    
    return best_pts, best_energy


def _gradient_descent_sphere(pts, max_time, lr=0.01):
    """
    Gradient descent on the sphere using retraction (normalize after each step).
    """
    start_time = time.time()
    n = pts.shape[0]
    best_pts = pts.copy()
    best_energy, _ = _log_energy_and_grad(best_pts)
    
    # Adaptive learning rate
    lr_cur = lr
    stall_count = 0
    prev_energy = best_energy
    
    while time.time() - start_time < max_time:
        e, g = _log_energy_and_grad(pts)
        
        # Project gradient to tangent space: g_tan = g - (g . x) * x
        g_dot_x = np.sum(g * pts, axis=1, keepdims=True)
        g_tan = g - g_dot_x * pts
        
        # Take step
        pts_new = pts - lr_cur * g_tan
        pts_new = _project_to_sphere(pts_new)
        
        e_new, _ = _log_energy_and_grad(pts_new)
        
        if e_new < best_energy:
            best_energy = e_new
            best_pts = pts_new.copy()
        
        if e_new < prev_energy - 1e-15:
            # Improving
            if e_new < prev_energy - 1e-8:
                lr_cur = min(lr_cur * 1.1, 0.1)
            stall_count = 0
        else:
            # Stalling
            lr_cur *= 0.5
            stall_count += 1
            if stall_count > 50:
                break
        
        prev_energy = e_new
        pts = pts_new
    
    return best_pts, best_energy


def _simulated_annealing(pts, max_time, n_iter_max=50000):
    """
    Simple simulated annealing with random perturbations.
    """
    start_time = time.time()
    n = pts.shape[0]
    best_pts = pts.copy()
    best_energy, _ = _log_energy_and_grad(best_pts)
    
    current_pts = pts.copy()
    current_energy = best_energy
    
    T = 1.0  # initial temperature
    T_min = 1e-8
    cool_rate = 0.999
    
    rng = np.random.default_rng(42)
    
    for it in range(n_iter_max):
        if time.time() - start_time > max_time:
            break
        
        # Random perturbation: move one point randomly
        idx = rng.integers(0, n)
        noise = rng.standard_normal(3) * 0.1
        new_pts = current_pts.copy()
        new_pts[idx] += noise
        new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])
        
        e_new, _ = _log_energy_and_grad(new_pts)
        
        delta = e_new - current_energy
        if delta < 0 or rng.random() < math.exp(-delta / max(T, T_min)):
            current_pts = new_pts
            current_energy = e_new
            if e_new < best_energy:
                best_energy = e_new
                best_pts = current_pts.copy()
        
        T = max(T * cool_rate, T_min)
    
    return best_pts, best_energy


def construct():
    """
    Construct 150 points on unit sphere minimizing logarithmic energy.
    Returns array of shape (150, 3).
    """
    t_start = time.time()
    deadline = t_start + TIME_LIMIT
    
    # Start with Fibonacci spiral
    rng = np.random.default_rng(12345)
    
    # Multiple starting points
    starts = []
    # Fibonacci spiral
    starts.append(_fibonacci_spiral(N))
    # Random starts
    for _ in range(3):
        starts.append(_random_points(N, rng))
    
    best_pts = None
    best_energy = float('inf')
    
    for si, start in enumerate(starts):
        if time.time() > deadline - 30:
            break
        
        # Phase 1: L-BFGS with penalty
        t_phase1 = min(40.0, max(10.0, (deadline - time.time()) * 0.5))
        pts1, e1 = _lbfgs_optimize(start, t_phase1)
        
        if e1 < best_energy:
            best_energy = e1
            best_pts = pts1.copy()
        
        # Phase 2: Fine-tune with gradient descent on sphere
        if time.time() > deadline - 20:
            break
        t_phase2 = min(15.0, max(5.0, (deadline - time.time()) * 0.4))
        pts2, e2 = _gradient_descent_sphere(pts1, t_phase2, lr=0.005)
        
        if e2 < best_energy:
            best_energy = e2
            best_pts = pts2.copy()
        
        # Phase 3: Short simulated annealing for escape
        if time.time() > deadline - 15:
            break
        t_phase3 = min(10.0, max(3.0, (deadline - time.time()) * 0.3))
        pts3, e3 = _simulated_annealing(pts2, t_phase3)
        
        if e3 < best_energy:
            best_energy = e3
            best_pts = pts3.copy()
        
        # Phase 4: Final L-BFGS polish
        if time.time() > deadline - 10:
            break
        t_phase4 = min(8.0, max(2.0, deadline - time.time() - 5.0))
        if t_phase4 > 2.0:
            pts4, e4 = _lbfgs_optimize(pts3, t_phase4)
            if e4 < best_energy:
                best_energy = e4
                best_pts = pts4.copy()
    
    # Final safety: ensure all points are on sphere and valid
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    best_pts = _project_to_sphere(best_pts)
    
    # Validate
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
        best_pts = _project_to_sphere(best_pts)
    
    return best_pts

# EVOLVE-BLOCK-END
