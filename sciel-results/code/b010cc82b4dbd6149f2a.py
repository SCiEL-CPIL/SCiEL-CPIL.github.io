"""Optimize log-energy on the sphere via L-BFGS-B with Riemannian polish."""
import time
import numpy as np
from math import fsum
from scipy.optimize import minimize

N = 150
HARD_LIMIT = 150.0
SAFETY = 10.0  # seconds before the hard limit


def _energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/||x_i - x_j||)."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    triu_idx = np.triu_indices(n, k=1)
    d_vals = np.sqrt(np.maximum(d2[triu_idx], 1e-30))
    return -np.sum(np.log(d_vals))


def _energy_grad(pts):
    """Compute gradient of logarithmic energy. grad_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    d2 = np.sum(diff * diff, axis=2)  # (N, N)
    np.fill_diagonal(d2, 1.0)
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    contrib = diff * inv_d2[:, :, np.newaxis]  # (N, N, 3)
    grad = np.sum(contrib, axis=1)  # (N, 3)
    return grad


def _project_tangent(pts, grad):
    """Project gradient onto tangent plane at each point: g - (g·x_i) x_i."""
    dots = np.einsum('ij,ij->i', grad, pts)
    return grad - dots[:, None] * pts


def _fibonacci_sphere():
    """Generate well-distributed points on sphere using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(rng):
    """Generate random points on sphere."""
    v = rng.standard_normal((N, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _geodesic_dome():
    """Generate points based on subdivided icosahedron for better initial distribution."""
    # Start with icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Subdivide faces to get more points
    # For N=150, we need about 12 * 10 = 120 from subdivision + 12 original = ~132, 
    # so we'll use a simpler approach: just use fibonacci for now as it's proven good
    return _fibonacci_sphere()


def _lbfgs_optimize(pts, deadline):
    """Optimize using L-BFGS-B on the sphere with penalty for deviation from unit sphere."""
    pts = pts.copy()
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    
    def objective_and_grad(x):
        """Flattened objective with sphere constraint via penalty."""
        p = x.reshape(-1, 3)
        # Energy
        e = _energy(p)
        # Gradient of energy
        g = _energy_grad(p)
        # Penalty for not being on sphere: 0.5 * sum(||p_i||^2 - 1)^2
        norms = np.linalg.norm(p, axis=1)
        penalty = 0.5 * np.sum((norms**2 - 1.0)**2)
        # Gradient of penalty: 2*(||p_i||^2-1)*p_i
        pen_grad = 2.0 * (norms**2 - 1.0)[:, None] * p
        # Total
        total_e = e + 1e4 * penalty
        total_g = g + 1e4 * pen_grad
        return total_e, total_g.flatten()
    
    x0 = pts.flatten()
    try:
        result = minimize(
            objective_and_grad, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 500}
        )
        pts_opt = result.x.reshape(-1, 3)
        pts_opt /= np.linalg.norm(pts_opt, axis=1, keepdims=True)
        e_opt = _energy(pts_opt)
        return pts_opt, e_opt
    except Exception:
        return pts, _energy(pts)


def _riemannian_polish(pts, deadline):
    """Final polish using Riemannian gradient descent with exponential map."""
    pts = pts.copy()
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    best = pts.copy()
    best_e = _energy(best)
    
    lr = 1e-4
    prev_grad = None
    prev_pts = None
    n = 0
    
    while time.time() < deadline:
        g = _energy_grad(pts)
        gt = _project_tangent(pts, g)
        gnorm = np.linalg.norm(gt)
        if gnorm < 1e-14:
            break
        
        # Adaptive step size
        if prev_grad is not None:
            y = gt - prev_grad
            s = pts - prev_pts
            ys = np.dot(y.flatten(), s.flatten())
            if abs(ys) > 1e-15:
                ss = np.dot(s.flatten(), s.flatten())
                bb_lr = ss / ys
                if bb_lr > 0:
                    lr = np.clip(bb_lr, 1e-10, 1e-2)
        
        # Exponential map: move along geodesic
        # For small steps, exp_map(x, v) ≈ x + v - 0.5*(||v||^2*x + (x·v)*x)
        v = gt * lr
        v_norm_sq = np.sum(v * v, axis=1, keepdims=True)
        x_dot_v = np.einsum('ij,ij->i', pts, v)[:, None]
        pts_new = pts + v - 0.5 * (v_norm_sq * pts + x_dot_v * pts)
        pts_new /= np.linalg.norm(pts_new, axis=1, keepdims=True)
        
        e_new = _energy(pts_new)
        
        if e_new < best_e:
            best = pts_new.copy()
            best_e = e_new
            lr = min(lr * 1.1, 1e-2)
        else:
            lr *= 0.8
            if lr < 1e-12:
                break
        
        prev_pts = pts.copy()
        prev_grad = gt.copy()
        pts = pts_new
        n += 1
    
    return best, best_e


def construct():
    t0 = time.time()
    deadline = t0 + HARD_LIMIT - SAFETY  # absolute time deadline
    rng = np.random.default_rng(12345)
    best_pts = None
    best_e = np.inf

    # Multiple initializations
    inits = [_fibonacci_sphere()]
    for _ in range(4):
        inits.append(_random_sphere(rng))

    # Phase 1: L-BFGS-B optimization with each initialization
    n_inits = len(inits)
    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        # Allocate time: give more to earlier restarts
        per_restart_time = remaining / (n_inits - i)
        per_restart_time = max(per_restart_time, 5.0)
        
        pts, e = _lbfgs_optimize(init, time.time() + per_restart_time)
        if e < best_e:
            best_e = e
            best_pts = pts

    if best_pts is None:
        best_pts = _fibonacci_sphere()
        best_e = _energy(best_pts)

    # Phase 2: Riemannian polish from best configuration
    if time.time() < deadline - 3.0:
        polish_time = deadline - time.time() - 1.0
        if polish_time > 3.0:
            pts, e = _riemannian_polish(best_pts, time.time() + polish_time)
            if e < best_e:
                best_e = e
                best_pts = pts

    # Phase 3: Another L-BFGS pass from polished result
    if time.time() < deadline - 2.0:
        lbfgs_time = deadline - time.time() - 0.5
        if lbfgs_time > 2.0:
            pts, e = _lbfgs_optimize(best_pts, time.time() + lbfgs_time)
            if e < best_e:
                best_e = e
                best_pts = pts

    # Final normalization
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    # Validate: check for very close points
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_d2 = np.min(d2)
    if min_d2 < 1e-18:
        # Points too close, fall back to fibonacci
        best_pts = _fibonacci_sphere()
        best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
