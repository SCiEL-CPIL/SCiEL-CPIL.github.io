"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.optimize import minimize


def _energy_and_grad(pts):
    """Compute logarithmic energy and its gradient for points on the unit sphere.
    
    Energy: sum_{i<j} log(1 / ||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    Gradient w.r.t. x_i: sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    """
    n = len(pts)
    
    # Compute all pairwise differences and squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy using math.fsum for accuracy
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist_sq[iu])
    energy = -math.fsum(math.log(d) for d in dists)
    
    # Gradient: grad_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _halton_init(n, offset=0):
    """Generate n points using the Halton sequence in 3D with an offset."""
    def halton_sequence(n, base, start=1):
        seq = []
        for i in range(start, start + n):
            result = 0.0
            f = 1.0 / base
            num = i
            while num > 0:
                num, rem = divmod(num, base)
                result += rem * f
                f /= base
            seq.append(result)
        return np.array(seq)
    
    bases = [2, 3, 5]
    h1 = halton_sequence(n, bases[0], start=1 + offset)
    h2 = halton_sequence(n, bases[1], start=1 + offset)
    h3 = halton_sequence(n, bases[2], start=1 + offset)
    
    u = h1
    v = h2
    
    theta = 2.0 * np.pi * u
    phi = np.arccos(1.0 - 2.0 * v)
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    pts = np.column_stack([x, y, z])
    return pts


def _cartesian_gradient_step(pts, max_iter=100, lr_init=0.005):
    """Projected gradient descent in Cartesian space with projection to sphere."""
    x = pts.copy()
    lr = lr_init
    
    for it in range(max_iter):
        energy, grad_e = _energy_and_grad(x)
        
        # Project gradient to tangent space
        inner = np.sum(grad_e * x, axis=1)
        grad_t = grad_e - inner[:, np.newaxis] * x
        
        grad_norm = np.linalg.norm(grad_t)
        if grad_norm < 1e-14:
            break
        
        # Backtracking line search
        best_t = None
        best_energy = energy
        for scale in [1.0, 0.5, 0.25, 0.1, 0.05]:
            t = lr * scale
            x_trial = x - t * grad_t
            x_trial = _project_to_sphere(x_trial)
            e_trial, _ = _energy_and_grad(x_trial)
            if e_trial < best_energy:
                best_energy = e_trial
                best_t = t
        
        if best_t is None:
            lr *= 0.5
            if lr < 1e-12:
                break
        else:
            x = x - best_t * grad_t
            x = _project_to_sphere(x)
            energy = best_energy
            lr = min(lr * 1.05, 0.05)
    
    return x, energy


def _sobol_init(n, offset=0):
    """Generate n points using Sobol-like sequence (better low-discrepancy)."""
    golden = (1 + np.sqrt(5)) / 2
    pts = np.zeros((n, 3))
    for i in range(n):
        idx = i + offset
        u1 = (idx * golden) % 1.0
        u2 = (idx * golden * golden) % 1.0
        u3 = (idx * golden * golden * golden) % 1.0
        
        theta = 2.0 * np.pi * u1
        phi = np.arccos(1.0 - 2.0 * u2)
        
        pts[i, 0] = np.sin(phi) * np.cos(theta)
        pts[i, 1] = np.sin(phi) * np.sin(theta)
        pts[i, 2] = np.cos(phi)
    return pts


def _random_init(n, seed=42):
    """Generate n random points on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts = _project_to_sphere(pts)
    return pts


def _lbfgs_optimize(pts, max_iter=200):
    """Optimize using L-BFGS-B in flattened coordinates with projection."""
    x0 = pts.flatten().copy()
    
    def fun_and_grad(x_flat):
        pts = x_flat.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        e, g = _energy_and_grad(pts)
        return e, g.flatten()
    
    result = minimize(
        fun_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
    )
    
    best_pts = _project_to_sphere(result.x.reshape(-1, 3))
    best_e, _ = _energy_and_grad(best_pts)
    
    return best_pts, best_e


def _riemannian_cg(pts, max_iter=150, tol=1e-10):
    """Riemannian conjugate gradient on the sphere using exponential map."""
    x = pts.copy()
    n = len(x)
    
    energy, grad_e = _energy_and_grad(x)
    inner = np.sum(grad_e * x, axis=1)
    grad_r = grad_e - inner[:, np.newaxis] * x
    
    p = -grad_r.copy()
    grad_r_prev = grad_r.copy()
    
    for it in range(max_iter):
        grad_norm = np.linalg.norm(grad_r)
        if grad_norm < tol:
            break
        
        p_norm = np.linalg.norm(p, axis=1, keepdims=True)
        p_norm[p_norm < 1e-12] = 1e-12
        p_unit = p / p_norm
        
        best_t = None
        best_energy = energy
        for t in [0.5, 0.25, 0.1, 0.05, 0.02, 1.0, 2.0, 0.01, 0.005]:
            x_trial = x * np.cos(t * p_norm) + p_unit * np.sin(t * p_norm)
            x_trial = _project_to_sphere(x_trial)
            e_trial, _ = _energy_and_grad(x_trial)
            if e_trial < best_energy:
                best_energy = e_trial
                best_t = t
        
        if best_t is None:
            break
        
        x = x * np.cos(best_t * p_norm) + p_unit * np.sin(best_t * p_norm)
        x = _project_to_sphere(x)
        energy = best_energy
        
        _, grad_e_new = _energy_and_grad(x)
        inner_new = np.sum(grad_e_new * x, axis=1)
        grad_r_new = grad_e_new - inner_new[:, np.newaxis] * x
        
        beta_num = np.sum(grad_r_new * (grad_r_new - grad_r_prev))
        beta_den = np.sum(grad_r_prev * grad_r_prev)
        beta = max(0.0, beta_num / beta_den) if beta_den > 1e-16 else 0.0
        p = -grad_r_new + beta * p
        
        inner_p = np.sum(p * x, axis=1)
        p = p - inner_p[:, np.newaxis] * x
        
        grad_r_prev = grad_r_new
    
    return x, energy


def _check_validity(pts):
    """Check if points are valid: finite, on sphere, no points too close."""
    if not np.all(np.isfinite(pts)):
        return False
    if pts.shape != (N, 3):
        return False
    norms = np.linalg.norm(pts, axis=1)
    if not np.all(np.abs(norms - 1.0) < 1e-6):
        return False
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_d = np.sqrt(np.min(dist_sq))
    if min_d < 1e-9:
        return False
    return True


def construct():
    """Multi-initialization optimization with L-BFGS and Riemannian CG."""
    best_pts = None
    best_energy = float('inf')
    
    initializations = []
    
    # Halton sequences with different offsets
    for offset in [0, 1, 2, 3, 4, 5, 6, 7]:
        pts = _halton_init(N, offset=offset)
        pts = _project_to_sphere(pts)
        initializations.append(pts)
    
    # Sobol-like sequences
    for offset in [0, 1, 2]:
        pts = _sobol_init(N, offset=offset)
        pts = _project_to_sphere(pts)
        initializations.append(pts)
    
    # Random initializations with different seeds
    for seed in [42, 123, 456, 789]:
        pts = _random_init(N, seed=seed)
        initializations.append(pts)
    
    # Optimize each initialization
    for init_idx, pts in enumerate(initializations):
        # Phase 1: L-BFGS for fast convergence with curvature info
        pts, e = _lbfgs_optimize(pts, max_iter=150)
        
        # Phase 2: Riemannian CG for curvature-aware refinement
        pts, e = _riemannian_cg(pts, max_iter=100, tol=1e-11)
        
        # Phase 3: Another L-BFGS pass for fine-tuning
        pts, e = _lbfgs_optimize(pts, max_iter=100)
        
        # Phase 4: Final Riemannian CG polish
        pts, e = _riemannian_cg(pts, max_iter=80, tol=1e-12)
        
        pts = _project_to_sphere(pts)
        
        if _check_validity(pts) and e < best_energy:
            best_energy = e
            best_pts = pts.copy()
    
    if best_pts is None:
        pts = _halton_init(N, offset=0)
        pts = _project_to_sphere(pts)
        pts, _ = _lbfgs_optimize(pts, max_iter=100)
        pts, _ = _riemannian_cg(pts, max_iter=80, tol=1e-10)
        best_pts = pts
    
    return best_pts
# EVOLVE-BLOCK-END
