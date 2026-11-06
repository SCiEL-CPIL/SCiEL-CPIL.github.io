"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask]
    
    if np.any(d_upper < 1e-15):
        return -1e18
    return math.fsum(np.log(1.0 / d_upper).tolist())

def _log_energy_grad(pts):
    """Compute logarithmic energy and gradient on sphere (vectorized)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask]
    
    if np.any(d_upper < 1e-15):
        return -1e18, np.zeros_like(pts)
    
    energy = math.fsum(np.log(1.0 / d_upper).tolist())
    
    # Gradient of sum_{i<j} log(1/|x_i - x_j|) = - sum_{i<j} log(|x_i - x_j|)
    # d/dx_i E = - sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # weights_{ij} = 1/d_{ij}^2 for i != j, 0 for i=j
    weights = 1.0 / d2
    np.fill_diagonal(weights, 0.0)
    
    sum_weights = np.sum(weights, axis=1) # shape (n,)
    weighted_sum = weights @ pts # shape (n, 3)
    
    grad = -pts * sum_weights[:, np.newaxis] + weighted_sum
    
    # Project gradient to tangent space
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tangent = grad - radial * pts
    
    return energy, grad_tangent

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    """Generate random points on unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _icosahedron_sphere(n):
    """Generate points based on icosahedron symmetry, expanded to n."""
    # Start with icosahedron vertices
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = [
        (-1,  t,  0), ( 1,  t,  0), (-1, -t,  0), ( 1, -t,  0),
        ( 0, -1,  t), ( 0,  1,  t), ( 0, -1, -t), ( 0,  1, -t),
        ( t,  0, -1), ( t,  0,  1), (-t,  0, -1), (-t,  0,  1)
    ]
    verts = np.array(verts, dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    if n <= len(verts):
        return verts[:n]
        
    # If we need more points, add random points and optimize
    # For now, return icosahedron plus randoms to be filled by optimizer
    remaining = n - len(verts)
    extra = _random_sphere(remaining, seed=99)
    return np.vstack([verts, extra])

def _optimize_lbfgs(pts0, deadline, max_iter=1000):
    """
    Optimize using L-BFGS-B on the sphere.
    We parameterize the sphere using a flat coordinate system? No, 
    standard trick: optimize in R^3 but project gradient to tangent.
    However, scipy minimize doesn't handle manifold constraints natively.
    
    Alternative: Use the fact that for small steps, we can work in R^3 
    and project back. But L-BFGS assumes Euclidean space.
    
    Better approach: Use the "pullback" method or simply use L-BFGS 
    on the R^3 coordinates but with a custom line search that projects 
    to the sphere.
    
    Actually, a common robust method is to use L-BFGS on the R^3 
    coordinates, but the objective function projects to the sphere 
    before evaluation, and the gradient is projected to the tangent 
    space. This works well if the points don't move too far from 
    the sphere during the line search, which is ensured by the 
    projection in the objective.
    """
    
    def objective_and_grad(x_flat):
        pts = x_flat.reshape(N, 3)
        pts = _project_to_sphere(pts)
        e, g = _log_energy_grad(pts)
        return e, g
        
    x0 = pts0.flatten()
    
    # L-BFGS-B works on flat vectors.
    # We pass the projected gradient.
    # Note: The gradient returned by the objective is the tangent gradient.
    # This is consistent with the manifold optimization approach.
    
    start_time = time.time()
    
    try:
        res = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'maxfun': 10000,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 50
            }
        )
    except Exception:
        return pts0, _log_energy(pts0)
    
    best_pts = res.x.reshape(N, 3)
    best_pts = _project_to_sphere(best_pts)
    best_energy = _log_energy(best_pts)
    
    return best_pts, best_energy

def _optimize_nelder_mead(pts0, deadline, max_iter=2000):
    """
    Nelder-Mead simplex method.
    This is derivative-free and can escape local minima that gradient 
    methods might get stuck in, especially if the landscape is flat 
    or noisy.
    """
    # Nelder-Mead on high-dimensional spaces (300D) is slow and 
    # often fails. It's better used for low-dimensional problems.
    # However, we can use it as a "kicker" or for small perturbations.
    # Given N=100, 300D is too high for standard NM to be efficient 
    # in 120s.
    # Instead, we will use a simple random restart strategy with L-BFGS.
    # Or, we can use a "coordinate descent" or "simulated annealing" 
    # but we already have SA in the parent.
    
    # Let's stick to multiple L-BFGS runs from diverse starts.
    # This is more robust than NM in 300D.
    
    return _optimize_lbfgs(pts0, deadline, max_iter=500)

def construct():
    """Construct 100 points on unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    best_pts = None
    best_energy = np.inf
    
    # Strategy:
    # 1. Generate diverse starting points: Fibonacci, Random, Icosahedron.
    # 2. Run L-BFGS from each start.
    # 3. Keep the best result.
    
    starts = [
        _fibonacci_sphere(N),
        _random_sphere(N, seed=1),
        _random_sphere(N, seed=2),
        _random_sphere(N, seed=3),
        _icosahedron_sphere(N)
    ]
    
    # We have ~140s.
    # L-BFGS is fast. We can run many iterations.
    # Let's allocate ~20s per run.
    
    for i, pts0 in enumerate(starts):
        if time.time() > deadline - 5.0:
            break
            
        # Run L-BFGS
        # Max iter 1000 should be enough for convergence.
        pts_opt, e_opt = _optimize_lbfgs(pts0, deadline - 5.0, max_iter=1000)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt
            
        # If time permits, try a second round of optimization 
        # starting from the current best to see if we can improve further
        # (sometimes a second pass finds a slightly better local min)
        if time.time() < deadline - 10.0:
            pts_opt2, e_opt2 = _optimize_lbfgs(best_pts, deadline - 5.0, max_iter=500)
            if e_opt2 < best_energy:
                best_energy = e_opt2
                best_pts = pts_opt2

    # Final check
    if best_pts is None:
        # Fallback
        best_pts = _fibonacci_sphere(N)
        
    return best_pts
# EVOLVE-BLOCK-END
