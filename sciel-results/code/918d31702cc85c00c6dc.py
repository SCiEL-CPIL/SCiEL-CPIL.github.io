"""Heilbronn convex regions: maximize min triangle area / hull area for 13 points."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 13

# EVOLVE-BLOCK-START

def _hull_area(pts):
    """Calculate the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _min_tri_area(pts):
    """Calculate the minimum area of any triangle formed by three points."""
    n = len(pts)
    if n < 3:
        return 0.0
    
    idx = np.array(list(itertools.combinations(range(n), 3)))
    i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
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

def _halton_sequence(n, base):
    """Generate first n Halton sequence values in [0, 1) for given base."""
    result = np.zeros(n)
    for i in range(1, n + 1):
        f = 1.0
        val = 0.0
        j = i
        while j > 0:
            f /= base
            val += f * (j % base)
            j //= base
        result[i - 1] = val
    return result

def _generate_halton_points(n_points):
    """Generate points using 2D Halton sequence mapped to unit circle."""
    h1 = _halton_sequence(n_points, 2)
    h2 = _halton_sequence(n_points, 3)
    r = np.sqrt(h1)
    theta = 2 * np.pi * h2
    pts = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    return pts

def _soft_min_triangle_ratio(pts, temperature=0.01):
    """
    Smooth surrogate for min triangle area / hull area.
    Uses log-sum-exp to approximate the minimum.
    """
    n = len(pts)
    ha = _hull_area(pts)
    if ha < 1e-8:
        return -1e6
    
    idx = np.array(list(itertools.combinations(range(n), 3)))
    i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min using log-sum-exp trick
    # min(a) = -log(sum(exp(-a/T))) * T for small T
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

def _optimize_from_start(start_pts, deadline, max_iter=200):
    """
    Optimize from a starting point using L-BFGS-B with smooth surrogate.
    """
    # Start with a relatively high temperature, then anneal
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

def _local_search_around(pts, deadline, n_steps=50):
    """
    Simple local search with random perturbations to escape local minima.
    """
    best_pts = pts.copy()
    best_score = _score(pts)
    
    rng = np.random.default_rng(42)
    step_size = 0.1
    
    for _ in range(n_steps):
        if time.time() > deadline - 5:
            break
        
        # Try random perturbation
        perturbed = best_pts + rng.normal(0, step_size, size=(N, 2))
        score = _score(perturbed)
        
        if score > best_score:
            best_score = score
            best_pts = perturbed
            step_size *= 0.95
        else:
            step_size *= 0.98
    
    return best_pts, best_score

def _generate_diverse_starts():
    """Generate diverse initial configurations."""
    starts = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    starts.append(reg_pts)
    
    # 2. Halton sequence points
    starts.append(_generate_halton_points(N))
    
    # 3. Random points in circle with different seeds
    for seed in [0, 1, 2, 3, 4]:
        rng = np.random.default_rng(seed)
        r = np.sqrt(rng.random(N))
        theta = rng.random(N) * 2 * np.pi
        pts = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        starts.append(pts)
    
    # 4. Points on a deformed circle (radial perturbations)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    rng = np.random.default_rng(10)
    radii = 1.0 + rng.normal(0, 0.1, N)
    pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    starts.append(pts)
    
    # 5. Grid-like with jitter
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    grid_pts = np.array([[xi, yi] for xi in x for yi in y])[:13]
    rng2 = np.random.default_rng(20)
    grid_pts = grid_pts + rng2.normal(0, 0.05, size=(13, 2))
    starts.append(grid_pts)
    
    return starts

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_score = -np.inf
    best_pts = None
    
    starts = _generate_diverse_starts()
    
    for i, start in enumerate(starts):
        if time.time() > deadline - 15:
            break
        
        # Optimize from this start
        opt_pts, opt_score = _optimize_from_start(start, deadline)
        
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts
        
        # Local search around the optimized point
        if time.time() > deadline - 10:
            break
        ls_pts, ls_score = _local_search_around(opt_pts, deadline, n_steps=30)
        
        if ls_score > best_score:
            best_score = ls_score
            best_pts = ls_pts
    
    # If nothing worked, fallback to regular polygon
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return best_pts

# EVOLVE-BLOCK-END
