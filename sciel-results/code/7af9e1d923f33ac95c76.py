"""Optimize 100 points on the unit sphere to minimize logarithmic energy."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE_SECONDS = 115.0  # safety margin before the 120 s kill
EPS = 1e-12


def _energy_and_grad(pts):
    """
    pts: (N, 3) array of unit vectors.
    Returns (energy, grad) where energy = sum_{i<j} log(1/|xi-xj|)
    and grad is the gradient w.r.t. pts in R^{N x 3}.
    """
    # Pairwise differences: (N, N, 3)
    diff = pts[:, None, :] - pts[None, :, :]
    # Squared distances
    dist2 = np.sum(diff * diff, axis=2)
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    # Safe distances
    dist_safe = np.sqrt(np.where(mask, np.maximum(dist2, EPS), 1.0))
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    log_term = -np.log(dist_safe)
    energy = np.sum(log_term[mask])
    
    # Gradient: ∂E/∂p_i = sum_{j≠i} (p_i - p_j) / |p_i - p_j|²
    dist2_safe = np.where(mask, np.maximum(dist2, EPS), 1.0)
    inv_dist2 = np.where(mask, 1.0 / dist2_safe, 0.0)
    # For each i, sum over j of (p_i - p_j) * inv_dist2[i, j]
    # diff[i,j] = p_i - p_j
    grad = np.sum(diff * inv_dist2[:, :, None], axis=1)
    return energy, grad


def _tangent_grad(pts, grad):
    """Project Euclidean gradient onto the tangent space of the unit sphere."""
    # Tangent projection: g_t = g - (g . p) p
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - radial * pts


def _energy_only(pts):
    """Energy only."""
    e, _ = _energy_and_grad(pts)
    return e


def _fibonacci_sphere(n):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _random_sphere(n, seed):
    """Generate n random points on sphere."""
    rng = np.random.RandomState(seed)
    vecs = rng.randn(n, 3)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def _icosahedron_subdivide(n_target, max_subdivisions=10):
    """Generate points by subdividing an icosahedron."""
    # Icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Faces
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    points = verts.copy()
    faces = [list(f) for f in faces]
    
    for _ in range(max_subdivisions):
        if len(points) >= n_target:
            break
        new_faces = []
        mid_cache = {}
        for f in faces:
            mids = []
            for i in range(3):
                a, b = f[i], f[(i + 1) % 3]
                key = min(a, b), max(a, b)
                if key not in mid_cache:
                    mid = (points[a] + points[b]) / 2.0
                    mid = mid / np.linalg.norm(mid)
                    mid_cache[key] = len(points)
                    points = np.vstack([points, mid])
                mids.append(mid_cache[key])
            new_faces.extend([
                [f[0], mids[0], mids[2]],
                [f[1], mids[1], mids[0]],
                [f[2], mids[2], mids[1]],
                mids
            ])
        faces = new_faces
    
    # Select n_target points
    if len(points) >= n_target:
        return points[:n_target]
    else:
        # Pad with random points
        remaining = n_target - len(points)
        rng = np.random.RandomState(42)
        extra = rng.randn(remaining, 3)
        extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
        return np.vstack([points, extra])


def _riemannian_gradient_descent(pts, max_iter=500, tol=1e-10):
    """Riemannian gradient descent on the sphere."""
    t_start = time.time()
    x = pts.copy()
    lr = 0.01
    
    for it in range(max_iter):
        if time.time() - t_start > DEADLINE_SECONDS - 10:
            break
        
        e, g = _energy_and_grad(x)
        g_t = _tangent_grad(x, g)
        
        # Backtracking line search
        alpha = lr
        for _ in range(30):
            x_trial = x + alpha * g_t
            x_trial = x_trial / np.linalg.norm(x_trial, axis=1, keepdims=True)
            e_trial = _energy_only(x_trial)
            if e_trial < e:
                break
            alpha *= 0.5
        else:
            lr *= 0.5
            if lr < 1e-12:
                break
            continue
        
        x = x_trial
        e = e_trial
        
        if np.linalg.norm(g_t) < tol:
            break
        
        # Slowly decrease learning rate
        lr = max(0.001, lr * 0.999)
    
    return x, e


def _polish_lbfgs(pts):
    """Polish with L-BFGS-B."""
    x0 = pts.ravel()
    
    def fun_and_grad(x):
        x_pts = x.reshape(N, 3)
        x_pts = x_pts / np.linalg.norm(x_pts, axis=1, keepdims=True)
        e, g = _energy_and_grad(x_pts)
        g_t = _tangent_grad(x_pts, g)
        return e, g_t.ravel()
    
    try:
        res = minimize(fun_and_grad, x0, jac=True, method='L-BFGS-B',
                       options={'maxiter': 300, 'gtol': 1e-10,
                                'ftol': 1e-14, 'maxfun': 1000})
        x_opt = res.x.reshape(N, 3)
        x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
        e_opt = _energy_only(x_opt)
        return x_opt, e_opt
    except Exception:
        return pts, _energy_only(pts)


def construct():
    t_start = time.time()
    
    # Generate diverse initializations
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_icosahedron_subdivide(N))
    
    # Add random initializations
    for seed in range(5):
        inits.append(_random_sphere(N, seed))
    
    best_pts = None
    best_energy = float('inf')
    
    for init in inits:
        if time.time() - t_start > DEADLINE_SECONDS - 5:
            break
        
        # Riemannian gradient descent
        x, e = _riemannian_gradient_descent(init, max_iter=300)
        
        # L-BFGS-B polish
        x, e = _polish_lbfgs(x)
        
        if e < best_energy:
            best_energy = e
            best_pts = x.copy()
    
    # Final polish on best
    if best_pts is not None and time.time() - t_start < DEADLINE_SECONDS - 2:
        best_pts, best_energy = _polish_lbfgs(best_pts)
    
    # Ensure unit sphere
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    return best_pts
