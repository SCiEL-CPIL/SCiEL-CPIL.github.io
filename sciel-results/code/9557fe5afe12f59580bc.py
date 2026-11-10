"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from math import log, fsum
import time
from scipy.optimize import minimize

DEADLINE = 150.0
SAFETY_MARGIN = 0.85  # use 85% of time

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

def _log_energy_vec(pts):
    """Vectorized log energy for faster evaluation."""
    n = len(pts)
    # Use squared distances to avoid sqrt where possible, but we need actual distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    # Mask out diagonal and upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d2 = dist2[mask]
    if np.any(d2 < 1e-30):
        return 1e30
    d = np.sqrt(d2)
    return -np.sum(np.log(d))

def _log_energy_grad(pts):
    """
    Compute energy and gradient for E = -sum_{i<j} log(d_ij).
    grad_i = -sum_{j!=i} (x_i - x_j) / d_ij^2
    """
    n = len(pts)
    energy = 0.0
    grad = np.zeros_like(pts)
    
    # Vectorized distance calculation
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist2 = np.sum(diff * diff, axis=2)  # (n, n)
    
    # Mask diagonal to avoid division by zero
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    
    # Compute forces: F_ij = (x_i - x_j) / d_ij^2
    # We need to handle d2 > 0
    safe_dist2 = np.where(dist2 > 1e-15, dist2, 1.0)
    inv_dist2 = np.where(mask, 1.0 / safe_dist2, 0.0)
    
    # force[i, j] = diff[i, j] * inv_dist2[i, j]
    force = diff * inv_dist2[:, :, np.newaxis]
    
    # grad[i] = -sum_j force[i, j]
    # Wait, the gradient of -log(d_ij) with respect to x_i is -(x_i - x_j) / d_ij^2.
    # So grad[i] = - sum_{j != i} (x_i - x_j) / d_ij^2
    # This is - sum_j force[i, j]
    grad = -np.sum(force, axis=1)
    
    # Energy calculation
    d = np.sqrt(dist2[mask])
    if np.any(d < 1e-15):
        return 1e30, grad
    energy = -np.sum(np.log(d))
    
    return energy, grad

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

def _repulsion_refine_vec(pts, n_iters=500, lr=0.01, deadline=None):
    """Iterative repulsion with sphere projection, vectorized."""
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces: F_i = sum_{j!=i} (x_i - x_j) / d_ij^2
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        
        # Mask diagonal
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
        
        # Strong repulsion if too close
        safe_dist2 = np.where(dist2 > 1e-12, dist2, 1e-12)
        inv_dist2 = np.where(mask, 1.0 / safe_dist2, 0.0)
        
        # Add strong repulsion for very close points
        close_mask = (dist2 < 1e-12) & mask
        inv_dist2[close_mask] = 1000.0
        
        force = diff * inv_dist2[:, :, np.newaxis]
        forces = np.sum(force, axis=1)
        
        pts += lr * forces
        # Project back to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
    return pts

def _lbfgs_optimize(pts, deadline, maxiter=1000, gtol=1e-10):
    """Run L-BFGS-B optimization on the sphere."""
    def fun_and_grad(x):
        p = x.reshape(-1, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        p = p / norms
        e, g = _log_energy_grad(p)
        return e, g.ravel()
    
    x0 = pts.ravel()
    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B',
                      options={'maxiter': maxiter, 'gtol': gtol, 'maxfun': maxiter * 2})
        if res.success or res.fun < 1e29:
            p = res.x.reshape(-1, 3)
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            p = p / norms
            e = _log_energy_vec(p)
            return p, e
    except Exception:
        pass
    return pts, _log_energy_vec(pts)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE * SAFETY_MARGIN
    
    rng = np.random.default_rng(42)
    
    # Generate multiple initial configurations
    inits = []
    # Fibonacci spiral
    inits.append(_fibonacci_spiral(N))
    # Fibonacci with different offsets
    for k in range(1, 5):
        inits.append(_fibonacci_spiral(N, offset=k * 0.7))
    # Random starts
    for k in range(5):
        inits.append(_random_points(N, rng))
    
    best_pts = None
    best_energy = 1e30
    
    # Phase 1: Quick repulsion refinement on all inits
    for init in inits:
        if time.time() > deadline:
            break
        refined = _repulsion_refine_vec(init, n_iters=200, lr=0.01, deadline=deadline)
        e = _log_energy_vec(refined)
        if e < best_energy:
            best_energy = e
            best_pts = refined.copy()
    
    # Phase 2: Gradient-based optimization on best candidates
    # Sort inits by their repulsion-refined energy
    scored_inits = []
    for init in inits:
        if time.time() > deadline:
            break
        refined = _repulsion_refine_vec(init, n_iters=100, lr=0.01, deadline=deadline)
        e = _log_energy_vec(refined)
        scored_inits.append((e, refined))
    
    scored_inits.sort(key=lambda x: x[0])
    
    for e_init, refined in scored_inits[:5]:
        if time.time() > deadline:
            break
        opt_pts, opt_e = _lbfgs_optimize(refined, deadline, maxiter=500, gtol=1e-10)
        if opt_e < best_energy:
            best_energy = opt_e
            best_pts = opt_pts.copy()
    
    # Phase 3: Simulated annealing to escape local minima
    if best_pts is not None and time.time() < deadline:
        sa_pts = best_pts.copy()
        sa_energy = best_energy
        T = 0.1
        for it in range(500):
            if time.time() > deadline:
                break
            # Random perturbation
            idx = rng.integers(0, N)
            angle = rng.uniform(0, 2 * np.pi)
            # Rotate point by small angle
            axis = rng.standard_normal(3)
            axis /= np.linalg.norm(axis)
            theta = rng.uniform(-0.1, 0.1)
            
            # Rodrigues' rotation formula
            c = np.cos(theta)
            s = np.sin(theta)
            k = axis
            v = sa_pts[idx]
            v_rot = v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1 - c)
            sa_pts_new = sa_pts.copy()
            sa_pts_new[idx] = v_rot
            sa_pts_new[idx] /= np.linalg.norm(v_rot)
            
            e_new = _log_energy_vec(sa_pts_new)
            if e_new < sa_energy or rng.random() < np.exp(-(e_new - sa_energy) / T):
                sa_pts = sa_pts_new
                sa_energy = e_new
            T *= 0.999  # Slow cooling
            
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_pts = sa_pts.copy()
    
    # Phase 4: Final aggressive refinement
    if best_pts is not None and time.time() < deadline:
        refined = _repulsion_refine_vec(best_pts, n_iters=500, lr=0.005, deadline=deadline)
        e = _log_energy_vec(refined)
        if e < best_energy:
            best_energy = e
            best_pts = refined.copy()
        
        if time.time() < deadline:
            opt_pts, opt_e = _lbfgs_optimize(best_pts, deadline, maxiter=1000, gtol=1e-12)
            if opt_e < best_energy:
                best_energy = opt_e
                best_pts = opt_pts.copy()
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    return best_pts
# EVOLVE-BLOCK-END
