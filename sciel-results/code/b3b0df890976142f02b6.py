"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)"""
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)
    dist = np.sqrt(d2)
    iu = np.triu_indices(n, k=1)
    log_d = np.log(dist[iu])
    return -math.fsum(log_d)

def _gradient_log_energy(pts):
    """Tangent-space gradient of logarithmic energy, optimized with matrix ops."""
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)
    
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    sum_inv_d2 = inv_d2.sum(axis=1)
    sum_xj_inv_d2 = (pts.T @ inv_d2).T
    
    # grad of E = -sum_{i<j} log(d_ij) is: -sum_{j!=i} (x_i - x_j)/d^2
    grad = -(pts * sum_inv_d2[:, np.newaxis] - sum_xj_inv_d2)
    
    grad_dot_xi = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tan = grad - grad_dot_xi * pts
    
    return grad_tan

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / np.maximum(norms, 1e-15)

def _perturb_sphere(pts, rng, sigma=0.05):
    """Perturb points and renormalize."""
    noise = rng.standard_normal(pts.shape) * sigma
    new_pts = pts + noise
    return _project_to_sphere(new_pts)

def _icosahedron_subdivide(n, rng):
    """Generate points by subdividing an icosahedron properly."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = _project_to_sphere(verts)
    
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    ]
    
    pts = list(verts)
    
    # Add edge midpoints
    edges = set()
    for f in faces:
        for i in range(3):
            a, b = f[i], f[(i+1) % 3]
            edge = (min(a,b), max(a,b))
            edges.add(edge)
    
    for a, b in edges:
        mid = (verts[a] + verts[b]) / 2.0
        pts.append(_project_to_sphere(mid.reshape(1,3))[0])
    
    # Add face centers
    for f in faces:
        center = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        pts.append(_project_to_sphere(center.reshape(1,3))[0])
    
    pts = np.array(pts)
    
    # If we have more than n, randomly select
    if len(pts) > n:
        indices = rng.choice(len(pts), n, replace=False)
        pts = pts[indices]
    # If we have fewer, add random points
    elif len(pts) < n:
        remaining = _random_sphere(n - len(pts), rng)
        pts = np.vstack([pts, remaining])
    
    return _project_to_sphere(pts)

def _golden_spiral_v2(n, offset=0.0):
    """Alternative golden spiral with offset for diversity."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    z = np.clip(z, -1.0, 1.0)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _riemannian_descent(pts, max_iter=1000, tol=1e-12, deadline=None):
    """Riemannian gradient descent with backtracking line search on the sphere."""
    n = pts.shape[0]
    pts = _project_to_sphere(pts.copy())
    f = _log_energy(pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        grad = _gradient_log_energy(pts)
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Initial step size - adaptive based on gradient magnitude
        step = 1.0 / (1.0 + 0.3 * grad_norm)
        
        accepted = False
        for ls_iter in range(50):
            # Riemannian step: move along negative gradient, then project
            new_pts = pts - step * grad
            new_pts = _project_to_sphere(new_pts)
            new_f = _log_energy(new_pts)
            
            if np.isfinite(new_f) and new_f < f:
                # Accept if sufficient decrease
                if new_f < f - 1e-6 * step * grad_norm:
                    accepted = True
                    break
                # Accept any decrease
                if new_f < f - 1e-10:
                    accepted = True
                    break
            
            # Backtrack
            step *= 0.7
        
        if accepted:
            pts = new_pts
            f = new_f
        else:
            break
    
    return pts, f

def _repulsion_refine(pts, n_iter=50, deadline=None):
    """Local refinement using repulsion-based updates to escape shallow minima."""
    pts = _project_to_sphere(pts.copy())
    f = _log_energy(pts)
    n = pts.shape[0]
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute repulsion forces (inverse square law)
        dots = pts @ pts.T
        d2 = 2.0 - 2.0 * dots
        np.fill_diagonal(d2, 1.0)
        d2 = np.maximum(d2, 1e-15)
        
        # Repulsion: each point is pushed away from others
        inv_d2 = 1.0 / d2
        np.fill_diagonal(inv_d2, 0.0)
        
        # Force on each point: sum of (x_i - x_j) / d^2 for all j
        sum_xj_inv_d2 = (pts.T @ inv_d2).T
        sum_inv_d2 = inv_d2.sum(axis=1)
        
        force = pts * sum_inv_d2[:, np.newaxis] - sum_xj_inv_d2
        
        # Project to tangent space
        force_dot_xi = np.sum(force * pts, axis=1, keepdims=True)
        force_tan = force - force_dot_xi * pts
        
        # Step size based on force magnitude
        force_norm = np.linalg.norm(force_tan)
        if force_norm < 1e-15:
            break
        
        step = 0.01 / (1.0 + force_norm * 0.1)
        
        new_pts = _project_to_sphere(pts + step * force_tan)
        new_f = _log_energy(new_pts)
        
        if new_f < f:
            pts = new_pts
            f = new_f
    
    return pts, f

def _optimize_from_start(pts0, time_budget, deadline):
    """Optimize from a starting configuration using multiple methods."""
    best_pts = _project_to_sphere(pts0.copy())
    best_energy = _log_energy(best_pts)
    
    def objective(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        return _log_energy(pts)
    
    def gradient(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        g = _gradient_log_energy(pts)
        return g.reshape(-1)
    
    x0 = best_pts.reshape(-1)
    
    # Pass 1: L-BFGS-B with moderate tolerance
    try:
        res = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 500, 'gtol': 1e-8, 'maxfun': 1000, 'ftol': 1e-10}
        )
        pts_new = _project_to_sphere(res.x.reshape(N, 3))
        e_new = _log_energy(pts_new)
        if e_new < best_energy:
            best_pts = pts_new
            best_energy = e_new
    except Exception:
        pass
    
    if time.time() > deadline - 5.0:
        return best_pts, best_energy
    
    # Pass 2: Riemannian gradient descent for robustness
    if time.time() < deadline - 8.0:
        try:
            pts_rd, e_rd = _riemannian_descent(best_pts, max_iter=500, tol=1e-12, deadline=deadline - 8.0)
            if e_rd < best_energy:
                best_pts = pts_rd
                best_energy = e_rd
        except Exception:
            pass
    
    if time.time() > deadline - 5.0:
        return best_pts, best_energy
    
    # Pass 3: Repulsion-based refinement
    if time.time() < deadline - 6.0:
        try:
            pts_rep, e_rep = _repulsion_refine(best_pts, n_iter=100, deadline=deadline - 6.0)
            if e_rep < best_energy:
                best_pts = pts_rep
                best_energy = e_rep
        except Exception:
            pass
    
    if time.time() > deadline - 5.0:
        return best_pts, best_energy
    
    # Pass 4: Final high-precision L-BFGS polish
    try:
        x1 = best_pts.reshape(-1)
        res2 = minimize(
            objective,
            x1,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'gtol': 1e-14, 'maxfun': 2000, 'ftol': 1e-15}
        )
        pts_new = _project_to_sphere(res2.x.reshape(N, 3))
        e_new = _log_energy(pts_new)
        if e_new < best_energy:
            best_pts = pts_new
            best_energy = e_new
    except Exception:
        pass
    
    return best_pts, best_energy

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_sphere(N))
    
    # Icosahedron subdivision
    initializations.append(_icosahedron_subdivide(N, rng))
    
    # Golden spiral variants
    for offset in [0.1, 0.3, 0.7]:
        initializations.append(_golden_spiral_v2(N, offset))
    
    # Random starts
    for _ in range(5):
        initializations.append(_random_sphere(N, rng))
    
    # Perturbed Fibonacci
    for sigma in [0.02, 0.05, 0.1, 0.2]:
        initializations.append(_perturb_sphere(_fibonacci_sphere(N), rng, sigma))
    
    # Perturbed icosahedron
    for sigma in [0.03, 0.08, 0.15]:
        initializations.append(_perturb_sphere(_icosahedron_subdivide(N, rng), rng, sigma))
    
    # More random
    for _ in range(4):
        initializations.append(_random_sphere(N, rng))
    
    best_pts = initializations[0]
    best_energy = _log_energy(best_pts)
    
    # Try each initialization
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        remaining = deadline - time.time()
        # Allocate time based on remaining, cap at 15s per init
        time_for_this = min(remaining * 0.6, 15.0)
        
        if time_for_this < 3.0:
            continue
        
        pts, energy = _optimize_from_start(init_pts, time_for_this, deadline)
        
        if energy < best_energy:
            best_pts = pts
            best_energy = energy
    
    # Final polish on the best result
    if time.time() < deadline - 10.0:
        try:
            pts_final, e_final = _optimize_from_start(best_pts, deadline - time.time() - 5.0, deadline)
            if e_final < best_energy:
                best_pts = pts_final
                best_energy = e_final
        except Exception:
            pass
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Validate
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_sphere(N)
    
    # Check for duplicate points
    dots = best_pts @ best_pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    min_d2 = d2.min()
    
    if min_d2 < 1e-18:  # distance < 1e-9
        # Fallback to Fibonacci if points are too close
        return _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
