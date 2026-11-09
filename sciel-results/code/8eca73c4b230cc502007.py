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
    n = len(theta_phi) // 2
    theta = theta_phi[:n]
    phi = theta_phi[n:]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])

def _neg_score_spherical(x, alpha_min=100.0, alpha_max=100.0):
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
    points = _spherical_to_cartesian(x)
    return -_score(points)

def _neg_score_cartesian(x, alpha_min=100.0, alpha_max=100.0):
    """Negative soft score in Cartesian coordinates with centering."""
    points = x.reshape(14, 3)
    # Center to remove translational invariance
    points = points - np.mean(points, axis=0)
    dists = pdist(points, metric='euclidean')
    if len(dists) == 0:
        return 0.0
    log_sum_exp_neg = np.log(np.sum(np.exp(-alpha_min * dists)))
    soft_min = -log_sum_exp_neg / alpha_min
    log_sum_exp_pos = np.log(np.sum(np.exp(alpha_max * dists)))
    soft_max = log_sum_exp_pos / alpha_max
    if soft_max < 1e-12:
        return 0.0
    ratio = soft_min / soft_max
    return -(ratio ** 2)

def _neg_score_cartesian_exact(x):
    points = x.reshape(14, 3)
    points = points - np.mean(points, axis=0)
    return -_score(points)

def _fibonacci_sphere(n_pts):
    theta = np.zeros(n_pts)
    phi = np.zeros(n_pts)
    golden = np.pi * (3 - np.sqrt(5))
    for i in range(n_pts):
        theta[i] = np.arccos(1 - 2 * (i + 0.5) / n_pts)
        phi[i] = golden * i
    return np.concatenate([theta, phi])

def _icosahedron_init():
    t = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    np.random.seed(42)
    extra = np.random.randn(2, 3)
    extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
    all_pts = np.vstack([verts, extra])
    theta = np.arccos(np.clip(all_pts[:, 2], -1, 1))
    phi = np.arctan2(all_pts[:, 1], all_pts[:, 0])
    phi = np.mod(phi, 2 * np.pi)
    return np.concatenate([theta, phi])

def _dodecahedron_init():
    """Dodecahedron has 20 vertices; take 14 of them."""
    phi_g = (1 + np.sqrt(5)) / 2
    inv_phi = 1.0 / phi_g
    # Dodecahedron vertices
    verts = []
    # 8 vertices from cube
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                verts.append([sx, sy, sz])
    # 12 vertices from three rectangles
    for s in [-1, 1]:
        verts.append([0, s*inv_phi, s*phi_g])
        verts.append([s*inv_phi, s*phi_g, 0])
        verts.append([s*phi_g, 0, s*inv_phi])
    verts = np.array(verts)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    # Take first 14
    verts = verts[:14]
    theta = np.arccos(np.clip(verts[:, 2], -1, 1))
    phi_ang = np.arctan2(verts[:, 1], verts[:, 0])
    phi_ang = np.mod(phi_ang, 2 * np.pi)
    return np.concatenate([theta, phi_ang])

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    timeout = 600
    deadline = start_time + timeout - 30  # safety margin
    best_score = -1
    best_points = None
    best_spherical = None
    best_cartesian = None
    
    n = 14
    dim_sph = 2 * n  # 28 dimensions
    dim_cart = 3 * n  # 42 dimensions
    
    # Generate initializations
    inits_sph = []
    inits_cart = []
    
    # 1. Random uniform on sphere
    for seed in range(40):
        np.random.seed(seed)
        theta = np.random.uniform(0, np.pi, n)
        phi = np.random.uniform(0, 2 * np.pi, n)
        inits_sph.append(np.concatenate([theta, phi]))
        # Also add Cartesian version
        pts = _spherical_to_cartesian(np.concatenate([theta, phi]))
        inits_cart.append(pts.flatten())
        
    # 2. Fibonacci sphere
    fib = _fibonacci_sphere(n)
    inits_sph.append(fib)
    inits_cart.append(_spherical_to_cartesian(fib).flatten())
    
    # 3. Perturbed Fibonacci
    for seed in range(10):
        np.random.seed(100 + seed)
        base = _fibonacci_sphere(n)
        perturbation = np.random.randn(dim_sph) * 0.05
        inits_sph.append(base + perturbation)
        pts = _spherical_to_cartesian(base + perturbation)
        inits_cart.append(pts.flatten())
    
    # 4. Icosahedron-based
    ico = _icosahedron_init()
    inits_sph.append(ico)
    inits_cart.append(_spherical_to_cartesian(ico).flatten())
    
    # 5. Dodecahedron-based
    dodec = _dodecahedron_init()
    inits_sph.append(dodec)
    inits_cart.append(_spherical_to_cartesian(dodec).flatten())
    
    # 6. Random Cartesian (centered)
    for seed in range(20):
        np.random.seed(200 + seed)
        pts = np.random.randn(n, 3)
        pts = pts - np.mean(pts, axis=0)
        inits_cart.append(pts.flatten())
    
    # Phase 1: Multi-start Nelder-Mead on spherical coordinates with soft objective
    for idx, init in enumerate(inits_sph):
        if time.time() > deadline - 120:
            break
        try:
            res = minimize(_neg_score_spherical, init, 
                          method='Nelder-Mead',
                          args=(100.0, 100.0),
                          options={'xatol': 1e-8, 'fatol': 1e-8, 
                                   'maxiter': 10000, 'adaptive': True})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_spherical = res.x.copy()
                best_points = _spherical_to_cartesian(res.x)
        except Exception:
            continue
    
    # Phase 2: Multi-start on Cartesian coordinates with soft objective
    for idx, init in enumerate(inits_cart):
        if time.time() > deadline - 90:
            break
        try:
            res = minimize(_neg_score_cartesian, init, 
                          method='Nelder-Mead',
                          args=(100.0, 100.0),
                          options={'xatol': 1e-8, 'fatol': 1e-8, 
                                   'maxiter': 10000, 'adaptive': True})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_cartesian = res.x.copy()
                best_points = res.x.reshape(14, 3)
                best_points = best_points - np.mean(best_points, axis=0)
                # Also update spherical for consistency
                theta = np.arccos(np.clip(best_points[:, 2], -1, 1))
                phi_ang = np.arctan2(best_points[:, 1], best_points[:, 0])
                phi_ang = np.mod(phi_ang, 2 * np.pi)
                best_spherical = np.concatenate([theta, phi_ang])
        except Exception:
            continue
    
    # Phase 3: Polish with tighter tolerances and higher alpha for sharper convergence
    if best_spherical is not None and time.time() < deadline - 60:
        try:
            res = minimize(_neg_score_spherical, best_spherical,
                          method='Nelder-Mead',
                          args=(500.0, 500.0),
                          options={'xatol': 1e-10, 'fatol': 1e-10,
                                   'maxiter': 20000, 'adaptive': True})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_spherical = res.x.copy()
                best_points = _spherical_to_cartesian(res.x)
        except Exception:
            pass
    
    # Phase 4: Final polish with exact (nonsmooth) objective for maximum precision
    if best_spherical is not None and time.time() < deadline - 30:
        try:
            res = minimize(_neg_score_spherical_exact, best_spherical,
                          method='Nelder-Mead',
                          options={'xatol': 1e-12, 'fatol': 1e-12,
                                   'maxiter': 30000, 'adaptive': True})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_spherical = res.x.copy()
                best_points = _spherical_to_cartesian(res.x)
        except Exception:
            pass
    
    # Phase 5: If we have a good Cartesian solution, try to polish it too
    if best_cartesian is not None and time.time() < deadline - 15:
        try:
            res = minimize(_neg_score_cartesian_exact, best_cartesian,
                          method='Nelder-Mead',
                          options={'xatol': 1e-12, 'fatol': 1e-12,
                                   'maxiter': 20000, 'adaptive': True})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_cartesian = res.x.copy()
                best_points = res.x.reshape(14, 3)
                best_points = best_points - np.mean(best_points, axis=0)
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
    
    # Ensure finite and correct shape
    best_points = np.array(best_points, dtype=np.float64)
    if best_points.shape != (14, 3):
        best_points = best_points.reshape(14, 3)
    if not np.all(np.isfinite(best_points)):
        # Last resort fallback
        theta = np.zeros(n)
        phi = np.zeros(n)
        golden = np.pi * (3 - np.sqrt(5))
        for i in range(n):
            theta[i] = np.arccos(1 - 2 * (i + 0.5) / n)
            phi[i] = golden * i
        best_points = _spherical_to_cartesian(np.concatenate([theta, phi]))
    
    return best_points
