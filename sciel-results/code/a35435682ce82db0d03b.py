"""
Symmetry-reduced lattice optimization for logarithmic energy minimization.

Strategy: Model the 150 points as a subset of a 3D periodic lattice (e.g. FCC or BCC)
by optimizing a small set of lattice basis vectors (9 parameters) plus a selection
pattern, then refine the resulting points with full gradient-based optimization.
This reduces the effective search space and enforces global symmetry.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

N = 150
TIME_LIMIT = 145.0  # safety margin before 150s kill

def _energy_and_grad(pts):
    """Compute log energy sum_{i<j} log(1/|x_i - x_j|) and its gradient w.r.t. pts."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    dist = np.sqrt(dist2)
    
    iu = np.triu_indices(n, k=1)
    d_upper = dist[iu]
    energy = -np.sum(np.log(d_upper))
    
    grad = np.zeros_like(pts)
    diff_upper = diff[iu]
    d2_upper = dist2[iu]
    coeff = 1.0 / d2_upper
    
    ii = iu[0]
    jj = iu[1]
    np.add.at(grad, ii, -coeff[:, np.newaxis] * diff_upper)
    np.add.at(grad, jj, coeff[:, np.newaxis] * diff_upper)
    
    return energy, grad

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _fibonacci_spiral(n):
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(pts)

def _generate_fcc_lattice(scale, n_target=150):
    """
    Generate an FCC lattice of points inside the unit sphere.
    FCC basis vectors: a1 = (s, s, 0), a2 = (s, 0, s), a3 = (0, s, s)
    where s = scale/2 for conventional cubic cell of side 'scale'.
    We generate all FCC points within radius 1 and select the closest to origin.
    """
    # FCC lattice: points at (i*s, j*s, k*s) where i+j+k is even, with s = scale/2
    # Actually, standard FCC: all (i,j,k) with i,j,k integers and i+j+k even, scaled.
    # Let's use: lattice points are at (a/2)*(i, j, k) where i,j,k are all integers 
    # and i+j+k is even? No, that's BCC.
    # FCC: (0,0,0), (0,1/2,1/2), (1/2,0,1/2), (1/2,1/2,0) + cubic lattice.
    # So points are at (i, j, k), (i+1/2, j+1/2, k), (i+1/2, j, k+1/2), (i, j+1/2, k+1/2) 
    # all multiplied by 'scale'.
    
    s = scale
    # Determine max index
    max_idx = int(np.ceil(1.0 / s)) + 2
    
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # Four FCC points per cubic cell
                candidates = [
                    np.array([i, j, k]) * s,
                    np.array([i + 0.5, j + 0.5, k]) * s,
                    np.array([i + 0.5, j, k + 0.5]) * s,
                    np.array([i, j + 0.5, k + 0.5]) * s,
                ]
                for p in candidates:
                    if np.dot(p, p) <= 1.0:
                        points.append(p)
    
    points = np.array(points)
    if len(points) == 0:
        return np.zeros((n_target, 3))
    
    # Sort by distance to origin and take the closest n_target
    norms = np.linalg.norm(points, axis=1)
    indices = np.argsort(norms)
    points = points[indices[:n_target]]
    
    if len(points) < n_target:
        # Pad with random if not enough (shouldn't happen)
        pad = np.random.randn(n_target - len(points), 3)
        pad = _project_to_sphere(pad)
        points = np.vstack([points, pad])
    
    return points

def _generate_bcc_lattice(scale, n_target=150):
    """
    Generate a BCC lattice of points inside the unit sphere.
    BCC: (i, j, k) and (i+1/2, j+1/2, k+1/2) scaled by 'scale'.
    """
    s = scale
    max_idx = int(np.ceil(1.0 / s)) + 2
    
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                candidates = [
                    np.array([i, j, k]) * s,
                    np.array([i + 0.5, j + 0.5, k + 0.5]) * s,
                ]
                for p in candidates:
                    if np.dot(p, p) <= 1.0:
                        points.append(p)
    
    points = np.array(points)
    if len(points) == 0:
        return np.zeros((n_target, 3))
    
    norms = np.linalg.norm(points, axis=1)
    indices = np.argsort(norms)
    points = points[indices[:n_target]]
    
    if len(points) < n_target:
        pad = np.random.randn(n_target - len(points), 3)
        pad = _project_to_sphere(pad)
        points = np.vstack([points, pad])
    
    return points

def _generate_simple_cubic_lattice(scale, n_target=150):
    """
    Generate a simple cubic lattice of points inside the unit sphere.
    """
    s = scale
    max_idx = int(np.ceil(1.0 / s)) + 2
    
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                p = np.array([i, j, k]) * s
                if np.dot(p, p) <= 1.0:
                    points.append(p)
    
    points = np.array(points)
    if len(points) == 0:
        return np.zeros((n_target, 3))
    
    norms = np.linalg.norm(points, axis=1)
    indices = np.argsort(norms)
    points = points[indices[:n_target]]
    
    if len(points) < n_target:
        pad = np.random.randn(n_target - len(points), 3)
        pad = _project_to_sphere(pad)
        points = np.vstack([points, pad])
    
    return points

def _lattice_energy(params, lattice_type='fcc'):
    """
    Energy function for lattice optimization.
    params: [scale] for single-scale lattice, or [a, b, c] for anisotropic.
    """
    if lattice_type == 'fcc':
        pts = _generate_fcc_lattice(params[0], N)
    elif lattice_type == 'bcc':
        pts = _generate_bcc_lattice(params[0], N)
    else:
        pts = _generate_simple_cubic_lattice(params[0], N)
    
    if not np.all(np.isfinite(pts)) or pts.shape != (N, 3):
        return 1e10
    
    # Check for near-duplicates
    norms = np.linalg.norm(pts, axis=1)
    if np.any(norms < 1e-9):
        return 1e10
    
    energy, _ = _energy_and_grad(pts)
    return energy

def _optimize_lattice_scale(lattice_type, deadline):
    """
    Optimize the lattice scale using differential evolution or simple search.
    """
    # Volume of sphere ~ 4.18879
    # For N=150, volume per point ~ 0.0279
    # For FCC, each point occupies ~ (s/2)^3 * 2/3? Roughly s ~ 0.3
    # Let's search in a reasonable range
    
    bounds = [(0.1, 1.0)]
    
    try:
        res = differential_evolution(
            lambda x: _lattice_energy(x, lattice_type),
            bounds,
            maxiter=100,
            tol=1e-6,
            seed=42,
            polish=False,
            updating='immediate'
        )
        best_scale = res.x[0]
        if lattice_type == 'fcc':
            pts = _generate_fcc_lattice(best_scale, N)
        elif lattice_type == 'bcc':
            pts = _generate_bcc_lattice(best_scale, N)
        else:
            pts = _generate_simple_cubic_lattice(best_scale, N)
        return pts
    except Exception:
        return None

def _lbfgs_optimize(pts, deadline):
    """Optimize using L-BFGS-B with sphere constraint via normalization."""
    n = pts.shape[0]
    
    def energy_flat(x):
        p = x.reshape(n, 3)
        p = _project_to_sphere(p)
        e, _ = _energy_and_grad(p)
        return e
    
    def grad_flat(x):
        p = x.reshape(n, 3)
        p = _project_to_sphere(p)
        e, g = _energy_and_grad(p)
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g_tangent.flatten()
    
    x0 = pts.flatten()
    try:
        res = minimize(energy_flat, x0, jac=grad_flat, 
                      method='L-BFGS-B', options={'maxiter': 1000, 'gtol': 1e-12})
        p_opt = _project_to_sphere(res.x.reshape(n, 3))
        e_opt = _energy_and_grad(p_opt)[0]
        return p_opt, e_opt
    except Exception:
        return pts, _energy_and_grad(pts)[0]

def _polish_gradient_descent(pts, deadline):
    """
    Additional polishing with gradient descent using adaptive step size.
    """
    n = pts.shape[0]
    pts = pts.copy()
    step = 0.1
    best_energy = _energy_and_grad(pts)[0]
    
    for it in range(200):
        if time.time() > deadline - 5.0:
            break
        
        e, g = _energy_and_grad(pts)
        if e < best_energy:
            best_energy = e
        
        # Project gradient to tangent space
        g_tangent = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        
        # Adaptive step size
        norm_g = np.linalg.norm(g_tangent)
        if norm_g < 1e-10:
            break
        
        # Try the step
        new_pts = pts - step * g_tangent
        new_pts = _project_to_sphere(new_pts)
        new_e, _ = _energy_and_grad(new_pts)
        
        if new_e < e:
            pts = new_pts
            step = min(step * 1.1, 1.0)
        else:
            step *= 0.5
    
    return pts, _energy_and_grad(pts)[0]

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_pts = _fibonacci_spiral(N)
    best_energy = _energy_and_grad(best_pts)[0]
    
    # Step 1: Try lattice-based initialization
    for lattice_type in ['fcc', 'bcc', 'cubic']:
        if time.time() > deadline - 60.0:
            break
        
        pts = _optimize_lattice_scale(lattice_type, deadline)
        if pts is not None:
            e, _ = _energy_and_grad(pts)
            if e < best_energy:
                best_energy = e
                best_pts = pts.copy()
    
    # Step 2: Refine the best lattice result with L-BFGS
    if time.time() < deadline - 30.0:
        p_lbfgs, e_lbfgs = _lbfgs_optimize(best_pts, deadline)
        if e_lbfgs < best_energy:
            best_energy = e_lbfgs
            best_pts = p_lbfgs
    
    # Step 3: Gradient descent polishing
    if time.time() < deadline - 10.0:
        p_gd, e_gd = _polish_gradient_descent(best_pts, deadline)
        if e_gd < best_energy:
            best_energy = e_gd
            best_pts = p_gd
    
    # Step 4: Random restarts with L-BFGS if time permits
    rng = np.random.default_rng(42)
    
    for restart in range(3):
        if time.time() > deadline - 20.0:
            break
        
        p = rng.standard_normal((N, 3))
        p = _project_to_sphere(p)
        
        p_lbfgs, e_lbfgs = _lbfgs_optimize(p, deadline)
        if e_lbfgs < best_energy:
            best_energy = e_lbfgs
            best_pts = p_lbfgs
    
    # Final refinement
    if time.time() < deadline - 2.0:
        p_final, e_final = _lbfgs_optimize(best_pts, deadline)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = p_final
    
    return _project_to_sphere(best_pts)
