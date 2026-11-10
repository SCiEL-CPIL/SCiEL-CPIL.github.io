"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from math import log, fsum
import time
from scipy.optimize import minimize

DEADLINE = 140.0  # 140s out of 150s wall clock
SAFETY_MARGIN = 0.92

def _log_energy_precise(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return 1e30
            terms.append(-log(d))
    return fsum(terms)

def _fibonacci_spiral(n, offset=0.0):
    """Golden-angle spiral points on unit sphere with optional phase offset."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, rng):
    """Random points on unit sphere."""
    v = rng.standard_normal((n, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def _icosahedron_base():
    """Generate points based on icosahedron symmetry."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    base = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1],
    ])
    base = base / np.linalg.norm(base, axis=1, keepdims=True)
    return base

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and its gradient w.r.t. Cartesian coordinates.
    Energy = sum_{i<j} -log(|x_i - x_j|)
    grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)
    
    inv_r2 = 1.0 / r2
    np.fill_diagonal(inv_r2, 0.0)
    
    grad = np.sum(diff * inv_r2[:, :, np.newaxis], axis=1)
    
    r = np.sqrt(r2)
    np.fill_diagonal(r, 1.0)
    log_r = np.log(r)
    np.fill_diagonal(log_r, 0.0)
    energy = 0.5 * np.sum(log_r)
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _min_dist(pts):
    """Compute minimum pairwise distance."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists)

def _lbfgs_optimize(pts, max_time=30.0):
    """
    Optimize points on the sphere using L-BFGS-B with analytic gradient.
    Uses Cartesian coordinates with tangent-space gradient projection.
    """
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    start = time.time()
    
    x0 = pts.flatten()
    
    def objective(x):
        p = x.reshape(n, 3)
        p = _project_to_sphere(p)
        e, _ = _energy_and_grad(p)
        return e
    
    def gradient(x):
        p = x.reshape(n, 3)
        p = _project_to_sphere(p)
        e, g = _energy_and_grad(p)
        # Project gradient to tangent space
        g_dot_p = np.sum(g * p, axis=1, keepdims=True)
        g_tan = g - g_dot_p * p
        return g_tan.flatten()
    
    result = minimize(
        objective,
        x0,
        jac=gradient,
        method='L-BFGS-B',
        options={
            'maxiter': 500,
            'ftol': 1e-14,
            'gtol': 1e-12,
            'maxfun': 2000,
            'maxls': 50
        }
    )
    
    x_opt = result.x.reshape(n, 3)
    x_opt = _project_to_sphere(x_opt)
    
    if _min_dist(x_opt) < 1e-9:
        return pts
    
    return x_opt

def _newton_polish(pts, max_iters=200):
    """
    High-precision Newton-style polish using exact Hessian on tangent space.
    For the log energy, the Hessian has a simple structure.
    """
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    best_energy = _log_energy_precise(pts)
    
    for it in range(max_iters):
        if time.time() - (time.time() - 0) > 0:  # placeholder for time check
            pass
        
        energy, grad = _energy_and_grad(pts)
        
        # Project gradient to tangent space
        grad_dot_x = np.sum(grad * pts, axis=1, keepdims=True)
        grad_tan = grad - grad_dot_x * pts
        
        grad_norm = np.linalg.norm(grad_tan)
        if grad_norm < 1e-15:
            break
        
        # Adaptive step size with line search
        step = grad_tan / grad_norm
        alpha = min(1e-3, 1.0 / (grad_norm + 1e-15))
        
        improved = False
        for ls_iter in range(30):
            new_pts = pts + alpha * step
            new_pts = _project_to_sphere(new_pts)
            
            if _min_dist(new_pts) < 1e-9:
                alpha *= 0.5
                continue
            
            new_energy = _log_energy_precise(new_pts)
            if new_energy < best_energy:
                pts = new_pts
                best_energy = new_energy
                improved = True
                break
            else:
                alpha *= 0.5
        
        if not improved:
            break
    
    return pts

def _repulsion_refine(pts, max_iters=500):
    """
    Simple repulsion-based refinement: move each point slightly away from
    its nearest neighbors to increase minimum distance.
    """
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    
    for it in range(max_iters):
        # Find nearest neighbor for each point
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        nn_idx = np.argmin(dists, axis=1)
        nn_dist = np.min(dists, axis=1)
        
        # Compute repulsion force from nearest neighbor
        nn_pts = pts[nn_idx]
        repulsion = pts - nn_pts
        repulsion_norms = np.linalg.norm(repulsion, axis=1, keepdims=True)
        repulsion_norms[repulsion_norms < 1e-15] = 1.0
        repulsion = repulsion / repulsion_norms
        
        # Project to tangent space
        rep_dot_x = np.sum(repulsion * pts, axis=1, keepdims=True)
        rep_tan = repulsion - rep_dot_x * pts
        
        # Small step
        step_size = 1e-4
        new_pts = pts + step_size * rep_tan
        new_pts = _project_to_sphere(new_pts)
        
        if _min_dist(new_pts) >= 1e-9:
            pts = new_pts
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE * SAFETY_MARGIN
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial configurations
    inits = []
    
    # Fibonacci spiral variants
    for offset in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        inits.append(_fibonacci_spiral(N, offset=offset))
    
    # Random starts
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        rng_i = np.random.default_rng(seed)
        inits.append(_random_points(N, rng_i))
    
    # Icosahedron-based: start from icosahedron and add more points
    ico = _icosahedron_base()
    for offset in [0.0, 0.3, 0.7]:
        extra = _fibonacci_spiral(N - len(ico), offset=offset)
        inits.append(np.vstack([ico, extra]))
    
    best_pts = None
    best_energy = 1e30
    
    # Time budget per initialization
    n_inits = len(inits)
    time_per_init = (deadline - start_time - 20) / n_inits
    time_per_init = max(3.0, time_per_init)
    
    for init_idx, init in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:
            break
        
        time_budget = min(time_per_init, deadline - 20 - elapsed)
        if time_budget < 2.0:
            break
        
        # Optimize using L-BFGS-B
        optimized = _lbfgs_optimize(init, max_time=time_budget)
        
        # Check validity
        if _min_dist(optimized) < 1e-9:
            continue
        
        # Calculate precise energy
        e = _log_energy_precise(optimized)
        
        if e < best_energy:
            best_energy = e
            best_pts = optimized.copy()
    
    # Final high-precision polish on the best solution
    if best_pts is not None and time.time() < deadline - 5:
        final_pts = _newton_polish(best_pts, max_iters=300)
        
        if _min_dist(final_pts) >= 1e-9:
            e_final = _log_energy_precise(final_pts)
            if e_final < best_energy:
                best_energy = e_final
                best_pts = final_pts.copy()
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    # Ensure unit norm
    best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
