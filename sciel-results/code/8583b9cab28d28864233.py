"""Heilbronn convex regions: maximize min triangle area / hull area for 13 points."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools

N = 13

# EVOLVE-BLOCK-START

# Precompute triangle indices
_TRI_IDX = np.array(list(itertools.combinations(range(N), 3)))
_TRI_I = _TRI_IDX[:, 0]
_TRI_J = _TRI_IDX[:, 1]
_TRI_K = _TRI_IDX[:, 2]

def _hull_area(pts):
    """Calculate the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _min_tri_area(pts):
    """Calculate the minimum area of any triangle formed by three points."""
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _score(pts):
    """Calculate the score: min triangle area / hull area."""
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 0.0
    mta = _min_tri_area(pts)
    return mta / ha

def _soft_min_triangle_ratio(pts, temperature=0.01):
    """
    Smooth surrogate for min triangle area / hull area.
    """
    n = len(pts)
    ha = _hull_area(pts)
    if ha < 1e-8:
        return -1e6
    
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min using log-sum-exp trick
    neg_areas = -areas / temperature
    max_neg = np.max(neg_areas)
    log_sum = np.log(np.sum(np.exp(neg_areas - max_neg))) + max_neg
    soft_min = -log_sum * temperature
    
    return soft_min / ha

def _objective_flat(flat_pts, temperature=0.01):
    """
    Objective function: minimize negative soft min triangle ratio.
    """
    pts = flat_pts.reshape(N, 2)
    val = _soft_min_triangle_ratio(pts, temperature)
    return -val

def _decode_params(x):
    """
    Decode parameter vector x into 2D points.
    x[0:N] are log-radii, x[N:2*N] are angles.
    Sorts points by angle to enforce convex order.
    """
    log_radii = x[:N]
    angles = x[N:2*N]
    
    # Ensure angles are in [0, 2*pi)
    angles = np.mod(angles, 2 * np.pi)
    
    # Sort by angle
    sort_idx = np.argsort(angles)
    sorted_angles = angles[sort_idx]
    sorted_log_radii = log_radii[sort_idx]
    
    # Exponentiate to get positive radii
    radii = np.exp(sorted_log_radii)
    
    pts = np.column_stack([
        radii * np.cos(sorted_angles),
        radii * np.sin(sorted_angles)
    ])
    return pts

def _soft_min(areas, tau):
    """
    Compute smooth minimum using log-sum-exp.
    """
    max_area = np.max(areas)
    log_sum_exp = np.log(np.sum(np.exp(-(areas - max_area) / tau)))
    return max_area - tau * log_sum_exp

def _objective_angular(x, tau=0.05):
    """
    Objective function for angular parameterization: 
    Negative smooth minimum triangle area normalized by hull area.
    """
    pts = _decode_params(x)
    
    # Compute all triangle areas
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    areas = 0.5 * np.abs(cross)
    
    if np.any(areas < 1e-10):
        return 1e10
    
    # Compute hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 1e10
        
    if hull_area < 1e-10:
        return 1e10

    # Smooth min using log-sum-exp
    smooth_min = _soft_min(areas, tau)
    score = smooth_min / hull_area
    return -score

def _optimize_from_start(start_pts, deadline, max_iter=200):
    """
    Optimize from a starting point using L-BFGS-B with smooth surrogate.
    """
    temperatures = [0.1, 0.05, 0.02, 0.01, 0.005]
    best_pts = start_pts.copy()
    best_score = _score(start_pts)
    
    for temp in temperatures:
        if time.time() > deadline - 5:
            break
            
        def obj(x):
            return _objective_flat(x, temp)
        
        try:
            res = minimize(
                obj,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            opt_pts = res.x.reshape(N, 2)
            score = _score(opt_pts)
            
            if score > best_score:
                best_score = score
                best_pts = opt_pts
        except Exception:
            continue
    
    return best_pts, best_score

def _optimize_angular(x0, deadline, tau=0.05, max_iter=100):
    """
    Optimize using angular parameterization with SLSQP.
    """
    bounds = [(-5.0, 5.0)] * N + [(0.0, 2 * np.pi)] * N
    
    def obj(x):
        return _objective_angular(x, tau)
    
    try:
        res = minimize(
            obj,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-10, 'disp': False}
        )
        x_opt = res.x
        pts_opt = _decode_params(x_opt)
        score_opt = _score(pts_opt)
        return pts_opt, score_opt
    except Exception:
        return x0, _score(_decode_params(x0))

def _generate_diverse_starts():
    """Generate diverse initial configurations."""
    starts = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    starts.append(reg_pts)
    
    # 2. Random points in circle with different seeds
    for seed in [0, 1, 2, 3, 4]:
        rng = np.random.default_rng(seed)
        r = np.sqrt(rng.random(N))
        theta = rng.random(N) * 2 * np.pi
        pts = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        starts.append(pts)
    
    # 3. Points on a deformed circle (radial perturbations)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    rng = np.random.default_rng(10)
    radii = 1.0 + rng.normal(0, 0.1, N)
    pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    starts.append(pts)
    
    # 4. Grid-like with jitter
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    grid_pts = np.array([[xi, yi] for xi in x for yi in y])[:13]
    rng2 = np.random.default_rng(20)
    grid_pts = grid_pts + rng2.normal(0, 0.05, size=(13, 2))
    starts.append(grid_pts)
    
    return starts

def _construct_angular_params(seed):
    """
    Generate initial parameters (log-radii, angles) for angular optimization.
    """
    rng = np.random.default_rng(seed)
    
    # Start with a regular polygon-like structure
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.normal(0, 0.1, N)
    log_radii = np.zeros(N) + rng.normal(0, 0.1, N)
    
    x = np.concatenate([log_radii, angles])
    return x

def _differential_evolution_angular(deadline):
    """
    Use differential evolution on the angular parameter space.
    """
    # Bounds for differential evolution
    bounds = [(-5.0, 5.0)] * N + [(0.0, 2 * np.pi)] * N
    
    def de_objective(x):
        return _objective_angular(x, tau=0.1)
    
    try:
        result = differential_evolution(
            de_objective,
            bounds,
            seed=42,
            maxiter=50,
            tol=1e-8,
            polish=True,
            init='sobol',
            popsize=15,
            updating='deferred'
        )
        if result.success:
            x_opt = result.x
            pts_opt = _decode_params(x_opt)
            score_opt = _score(pts_opt)
            return pts_opt, score_opt
    except Exception:
        pass
    return None, 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_score = -np.inf
    best_pts = None
    
    # Phase 1: Multi-start with angular parameterization + SLSQP
    num_angular_starts = 10
    time_per_start = (deadline - start_time - 30) / num_angular_starts
    
    for i in range(num_angular_starts):
        if time.time() > deadline - 30:
            break
            
        x0 = _construct_angular_params(seed=1000 + i)
        
        try:
            # Use SLSQP with angular parameterization
            opt_pts, opt_score = _optimize_angular(x0, deadline, tau=0.05, max_iter=80)
            
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts
        except Exception:
            continue
    
    # Phase 2: Differential evolution on angular space
    if time.time() < deadline - 40:
        de_pts, de_score = _differential_evolution_angular(deadline)
        if de_pts is not None and de_score > best_score:
            best_score = de_score
            best_pts = de_pts
    
    # Phase 3: Refine best solution with L-BFGS-B in Cartesian space
    if best_pts is not None and time.time() < deadline - 15:
        opt_pts, opt_score = _optimize_from_start(best_pts, deadline)
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts
    
    # Phase 4: Additional Cartesian multi-start if time permits
    if time.time() < deadline - 20:
        starts = _generate_diverse_starts()
        for i, start in enumerate(starts):
            if time.time() > deadline - 10:
                break
            
            opt_pts, opt_score = _optimize_from_start(start, deadline)
            
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts
    
    # Fallback to regular polygon
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return best_pts

# EVOLVE-BLOCK-END
