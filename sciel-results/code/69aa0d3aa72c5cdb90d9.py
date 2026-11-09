import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def _score(points):
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists = pdist(points, metric='euclidean')
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max < 1e-12:
        return 0.0
    return (d_min / d_max) ** 2

def _spherical_to_cartesian(theta_phi):
    """Convert (theta, phi) pairs to Cartesian coordinates on unit sphere.
    theta: polar angle [0, pi], phi: azimuthal angle [0, 2*pi]
    """
    n = len(theta_phi) // 2
    theta = theta_phi[:n]
    phi = theta_phi[n:]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])

def _neg_score_spherical(x, alpha_min=100.0, alpha_max=100.0):
    """Negative soft score using log-sum-exp for smooth min/max.
    x: 28D array of (theta_1..theta_14, phi_1..phi_14)
    """
    points = _spherical_to_cartesian(x)
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    # Soft min: -log(sum(exp(-alpha*d))) / alpha approximates min
    # Soft max: log(sum(exp(alpha*d))) / alpha approximates max
    log_sum_exp_neg = np.log(np.sum(np.exp(-alpha_min * dists)))
    soft_min = -log_sum_exp_neg / alpha_min
    log_sum_exp_pos = np.log(np.sum(np.exp(alpha_max * dists)))
    soft_max = log_sum_exp_pos / alpha_max
    if soft_max < 1e-12:
        return 0.0
    ratio = soft_min / soft_max
    return -(ratio ** 2)

def _neg_score_spherical_exact(x):
    """Exact (nonsmooth) negative score for final polish."""
    points = _spherical_to_cartesian(x)
    return -_score(points)

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    timeout = 600
    deadline = start_time + timeout - 30  # safety margin
    best_score = -1
    best_points = None
    best_spherical = None
    
    n = 14
    dim = 2 * n  # 28 dimensions (theta, phi per point)
    
    # Generate initializations in spherical coordinates
    inits = []
    
    # 1. Random uniform on sphere (Fibonacci-like + random)
    for seed in range(30):
        np.random.seed(seed)
        theta = np.random.uniform(0, np.pi, n)
        phi = np.random.uniform(0, 2 * np.pi, n)
        inits.append(np.concatenate([theta, phi]))
        
    # 2. Fibonacci sphere approximation
    def fibonacci_sphere(n_pts):
        theta = np.zeros(n_pts)
        phi = np.zeros(n_pts)
        golden = np.pi * (3 - np.sqrt(5))
        for i in range(n_pts):
            theta[i] = np.arccos(1 - 2 * (i + 0.5) / n_pts)
            phi[i] = golden * i
        return np.concatenate([theta, phi])
    
    inits.append(fibonacci_sphere(n))
    
    # 3. Perturbed Fibonacci
    for seed in range(5):
        np.random.seed(100 + seed)
        base = fibonacci_sphere(n)
        perturbation = np.random.randn(dim) * 0.1
        inits.append(base + perturbation)
    
    # 4. Icosahedron-based initializations (12 vertices + 2 more)
    def icosahedron_init():
        t = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
        ])
        verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
        # Add 2 more points: opposite of first two, or random
        np.random.seed(42)
        extra = np.random.randn(2, 3)
        extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
        all_pts = np.vstack([verts, extra])
        # Convert to spherical
        theta = np.arccos(np.clip(all_pts[:, 2], -1, 1))
        phi = np.arctan2(all_pts[:, 1], all_pts[:, 0])
        phi = np.mod(phi, 2 * np.pi)
        return np.concatenate([theta, phi])
    
    inits.append(icosahedron_init())
    
    # Run multi-start Nelder-Mead on spherical coordinates with soft objective
    for idx, init in enumerate(inits):
        if time.time() > deadline:
            break
        try:
            res = minimize(_neg_score_spherical, init, 
                          method='Nelder-Mead',
                          args=(100.0, 100.0),
                          options={'xatol': 1e-8, 'fatol': 1e-8, 
                                   'maxiter': 15000, 'adaptive': True})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_spherical = res.x.copy()
                best_points = _spherical_to_cartesian(res.x)
        except Exception:
            continue
    
    # Polish with tighter tolerances and higher alpha for sharper convergence
    if best_spherical is not None and time.time() < deadline - 10:
        try:
            res = minimize(_neg_score_spherical, best_spherical,
                          method='Nelder-Mead',
                          args=(500.0, 500.0),
                          options={'xatol': 1e-10, 'fatol': 1e-10,
                                   'maxiter': 30000, 'adaptive': True})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_spherical = res.x.copy()
                best_points = _spherical_to_cartesian(res.x)
        except Exception:
            pass
    
    # Final polish with exact (nonsmooth) objective for maximum precision
    if best_spherical is not None and time.time() < deadline - 5:
        try:
            res = minimize(_neg_score_spherical_exact, best_spherical,
                          method='Nelder-Mead',
                          options={'xatol': 1e-12, 'fatol': 1e-12,
                                   'maxiter': 50000, 'adaptive': True})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_spherical = res.x.copy()
                best_points = _spherical_to_cartesian(res.x)
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: return Fibonacci sphere
        theta = np.zeros(n)
        phi = np.zeros(n)
        golden = np.pi * (3 - np.sqrt(5))
        for i in range(n):
            theta[i] = np.arccos(1 - 2 * (i + 0.5) / n)
            phi[i] = golden * i
        best_points = _spherical_to_cartesian(np.concatenate([theta, phi]))
    
    return best_points
