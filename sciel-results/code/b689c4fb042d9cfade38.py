import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

N = 50

def _compute_radius(pts):
    """Compute the minimum radius for a given set of centers using vectorized ops."""
    n = len(pts)
    if n == 0:
        return 0.0
    # Distance to boundary
    norms = np.linalg.norm(pts, axis=1)
    boundary_dist = 1.0 - norms
    min_boundary = np.min(boundary_dist)
    
    # Pairwise distances using cdist (vectorized)
    dist_matrix = cdist(pts, pts)
    # Get upper triangle excluding diagonal
    iu = np.triu_indices(n, k=1)
    if len(iu[0]) == 0:
        min_pair = np.inf
    else:
        min_pair = np.min(dist_matrix[iu]) / 2.0
    
    return min(min_boundary, min_pair)


def _soft_min_radius(pts, beta=100.0):
    """
    Compute a soft-min approximation of the radius using log-sum-exp.
    """
    n = len(pts)
    
    # Boundary distances
    norms = np.linalg.norm(pts, axis=1)
    boundary_dist = 1.0 - norms
    
    # Pairwise distances (half) using cdist
    dist_matrix = cdist(pts, pts)
    iu = np.triu_indices(n, k=1)
    pair_dists = dist_matrix[iu] / 2.0
    
    # Combine all constraints into one array
    all_dists = np.concatenate([boundary_dist, pair_dists])
    
    # Soft min via log-sum-exp
    min_val = np.min(all_dists)
    if min_val < -100:
        return min_val
    
    exp_vals = np.exp(-beta * (all_dists - min_val))
    sum_exp = np.sum(exp_vals)
    
    if sum_exp == 0:
        return min_val
    
    soft_min = min_val - np.log(sum_exp) / beta
    
    return soft_min


def _soft_min_radius_grad(pts, beta=100.0):
    """
    Compute the gradient of the soft-min radius.
    """
    n = len(pts)
    grad = np.zeros_like(pts)
    
    # Boundary distances
    norms = np.linalg.norm(pts, axis=1)
    boundary_dist = 1.0 - norms
    
    # Pairwise distances using cdist
    dist_matrix = cdist(pts, pts)
    iu = np.triu_indices(n, k=1)
    pair_dists = dist_matrix[iu] / 2.0
    
    # Combine all constraints
    all_dists = np.concatenate([boundary_dist, pair_dists])
    
    min_val = np.min(all_dists)
    if min_val < -100:
        return grad
    
    exp_vals = np.exp(-beta * (all_dists - min_val))
    sum_exp = np.sum(exp_vals)
    
    if sum_exp == 0:
        return grad
    
    weights = exp_vals / sum_exp
    
    # Boundary terms
    for i in range(n):
        w = weights[i]
        if norms[i] > 1e-10:
            grad[i] += w * (-pts[i] / norms[i])
    
    # Pair terms
    for k in range(len(iu[0])):
        w = weights[n + k]
        i, j = iu[0][k], iu[1][k]
        diff = pts[i] - pts[j]
        norm_diff = np.linalg.norm(diff)
        if norm_diff > 1e-10:
            grad[i] += w * (diff / norm_diff)
            grad[j] += w * (-diff / norm_diff)
    
    return grad


def _hex_init(spacing):
    """Generate hexagonal lattice points."""
    k = int(1.0 / spacing) + 2
    rows = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            rows.append([x, y])
    points = np.array(rows)
    radius = np.linalg.norm(points, axis=1)
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]
    
    # Select closest to origin
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    points = points[indices[:N]]
    
    return points


def _ring_init():
    """Generate ring-based initialization."""
    points = [[0.0, 0.0]]
    
    r1 = 0.28
    n1 = 6
    for i in range(n1):
        angle = 2 * np.pi * i / n1
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    
    r2 = 0.58
    n2 = 12
    for i in range(n2):
        angle = 2 * np.pi * i / n2 + np.pi / 12
        points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    
    r3 = 0.85
    n3 = 18
    for i in range(n3):
        angle = 2 * np.pi * i / n3
        points.append([r3 * np.cos(angle), r3 * np.sin(angle)])
    
    # That's 1 + 6 + 12 + 18 = 37, need 13 more
    r4 = 0.96
    for i in range(13):
        angle = 2 * np.pi * i / 13 + np.pi / 13
        points.append([r4 * np.cos(angle), r4 * np.sin(angle)])
    
    return np.array(points)


def _random_init(seed):
    """Generate random initialization."""
    np.random.seed(seed)
    points = []
    for _ in range(N):
        r = np.sqrt(np.random.rand()) * 0.95
        theta = 2 * np.pi * np.random.rand()
        points.append([r * np.cos(theta), r * np.sin(theta)])
    return np.array(points)


def _optimize_from_init(init_pts, deadline=None):
    """
    Optimize from an initial configuration using multi-stage annealing with L-BFGS-B.
    """
    if deadline is not None and time.time() > deadline:
        return init_pts, _compute_radius(init_pts)
    
    best_pts = init_pts.copy()
    best_r = _compute_radius(init_pts)
    
    # Multi-stage beta annealing
    betas = [30, 60, 120, 250, 500]
    
    for beta in betas:
        if deadline is not None and time.time() > deadline - 5:
            break
        
        def objective(x):
            pts = x.reshape(-1, 2)
            return -_soft_min_radius(pts, beta)
        
        def gradient(x):
            pts = x.reshape(-1, 2)
            return -_soft_min_radius_grad(pts, beta)
        
        x0 = best_pts.flatten()
        
        try:
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            
            new_pts = res.x.reshape(-1, 2)
            new_r = _compute_radius(new_pts)
            if new_r > best_r:
                best_pts = new_pts
                best_r = new_r
        except Exception:
            pass
    
    return best_pts, best_r


def _nelder_mead_polish(pts, deadline=None):
    """Polish with Nelder-Mead for local refinement."""
    if deadline is not None and time.time() > deadline - 3:
        return pts
    
    def objective(x):
        p = x.reshape(-1, 2)
        return -_compute_radius(p)
    
    try:
        res = minimize(objective, pts.flatten(), method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14})
        new_pts = res.x.reshape(-1, 2)
        new_r = _compute_radius(new_pts)
        old_r = _compute_radius(pts)
        if new_r > old_r:
            return new_pts
    except Exception:
        pass
    
    return pts


def _final_polish(pts, deadline=None):
    """
    Final high-precision polish using very high beta and tight tolerances.
    """
    if deadline is not None and time.time() > deadline - 2:
        return pts
    
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    # Very high beta for precision
    for beta in [1000, 2000, 5000]:
        if deadline is not None and time.time() > deadline - 1:
            break
        
        def objective(x):
            p = x.reshape(-1, 2)
            return -_soft_min_radius(p, beta)
        
        def gradient(x):
            p = x.reshape(-1, 2)
            return -_soft_min_radius_grad(p, beta)
        
        try:
            res = minimize(objective, best_pts.flatten(), jac=gradient, method='L-BFGS-B',
                          options={'maxiter': 50, 'ftol': 1e-15, 'gtol': 1e-13})
            new_pts = res.x.reshape(-1, 2)
            new_r = _compute_radius(new_pts)
            if new_r > best_r:
                best_pts = new_pts
                best_r = new_r
        except Exception:
            pass
    
    return best_pts


def construct():
    start_time = time.time()
    timeout = 140  # Leave 10s margin for 150s limit
    deadline = start_time + timeout
    
    # Generate multiple initializations
    inits = []
    
    # Hex lattice
    s = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    hex_pts = _hex_init(s)
    inits.append(hex_pts)
    
    # Ring-based
    ring_pts = _ring_init()
    inits.append(ring_pts)
    
    # Random inits
    for seed in [42, 123, 456]:
        rand_pts = _random_init(seed)
        inits.append(rand_pts)
    
    best_pts = None
    best_r = -np.inf
    
    # Optimize from each initialization
    for init_pts in inits:
        if time.time() > deadline - 10:
            break
        
        try:
            pts, r = _optimize_from_init(init_pts, deadline)
            if r > best_r:
                best_r = r
                best_pts = pts
        except Exception:
            pass
    
    # Nelder-Mead polish
    if best_pts is not None and time.time() < deadline - 5:
        polished = _nelder_mead_polish(best_pts, deadline)
        r_polished = _compute_radius(polished)
        if r_polished > best_r:
            best_r = r_polished
            best_pts = polished
    
    # Final high-precision polish
    if best_pts is not None and time.time() < deadline - 2:
        final_pts = _final_polish(best_pts, deadline)
        r_final = _compute_radius(final_pts)
        if r_final > best_r:
            best_r = r_final
            best_pts = final_pts
    
    # Ensure we return a valid result
    if best_pts is None:
        # Fallback to hex lattice
        best_pts = hex_pts
    
    # Validate: ensure all points are inside the unit circle
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms >= 1.0):
        # Project points back inside
        for i in range(N):
            if norms[i] >= 1.0:
                best_pts[i] = best_pts[i] / norms[i] * 0.999
    
    return best_pts.reshape(N, 2)
