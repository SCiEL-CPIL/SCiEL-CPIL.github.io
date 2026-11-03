"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations
import sys

def _all_triangle_areas(pts):
    """Compute areas of all C(n,3) triangles formed by n points."""
    n = len(pts)
    areas = np.empty((n*(n-1)*(n-2))//6)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                areas[idx] = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
                idx += 1
    return areas

def _convex_hull_area(pts):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    areas = _all_triangle_areas(pts)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    return min_area / hull_area

def _soft_min_objective(x, beta):
    """
    Soft-min surrogate: -log-sum-exp(-beta * areas) / beta approximates min(areas).
    We want to maximize min(areas)/hull_area, so minimize -min(areas)/hull_area.
    Using soft-min: f = -(-lse(-beta*areas)/beta) / hull_area = lse(-beta*areas)/(beta*hull_area)
    But we also need to handle hull_area smoothly. Let's use a different approach.
    
    Actually, let's define:
    - Compute all triangle areas (vectorized)
    - Compute hull area
    - Objective to minimize = -min_triangle_area / hull_area (but min is non-smooth)
    
    For smooth optimization, use:
    objective = - (sum(exp(-beta*areas)) / (beta * hull_area)) ** (1/beta) ... this is getting complex.
    
    Simpler: minimize -min(areas) + penalty*(hull_area - 1)^2 to normalize hull to ~1
    But min is non-smooth.
    
    Let's use: objective = -lse(-beta * areas) / (beta * hull_area)
    This approximates -min(areas)/hull_area when beta is large.
    We minimize this, which maximizes min(areas)/hull_area.
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    # Compute all triangle areas vectorized
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                areas.append(0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])))
    areas = np.array(areas)
    
    # Convex hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 1e6
    
    if hull_area < 1e-10:
        return 1e6
    
    # Soft min approximation
    beta_safe = max(beta, 1.0)
    # lse(-beta * areas) / beta approximates -min(areas)
    # We want to maximize min(areas)/hull_area, so minimize -min(areas)/hull_area
    # Using: -lse(-beta*areas)/(beta*hull_area) approximates -min(areas)/hull_area
    log_terms = -beta_safe * areas
    max_log = np.max(log_terms)
    lse = max_log + np.log(np.sum(np.exp(log_terms - max_log)))
    obj = -lse / (beta_safe * hull_area)
    
    return obj

def _score_fast(x):
    """Fast score computation for tracking best."""
    pts = x.reshape(-1, 2)
    return _score(pts)

def _regular_polygon(n, radius=1.0, center=(0,0)):
    """Generate regular polygon vertices."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius*np.cos(angles), 
                            center[1] + radius*np.sin(angles)])

def _concentric_rings(n, n_rings=3):
    """Generate points on concentric rings for diverse initialization."""
    pts = []
    # Center point
    pts.append([0.0, 0.0])
    remaining = n - 1
    for ring in range(n_rings):
        if remaining <= 0:
            break
        # Distribute points on this ring
        n_on_ring = min(remaining, max(3, (n-1)//n_rings))
        if ring == n_rings - 1:
            n_on_ring = remaining
        radius = (ring + 1) / n_rings
        angles = np.linspace(0, 2*np.pi, n_on_ring, endpoint=False) + ring * np.pi/6
        ring_pts = np.column_stack([radius*np.cos(angles), radius*np.sin(angles)])
        pts.append(ring_pts)
        remaining -= n_on_ring
    
    if remaining > 0:
        # Add remaining points
        angles = np.linspace(0, 2*np.pi, remaining, endpoint=False)
        ring_pts = np.column_stack([1.0*np.cos(angles), 1.0*np.sin(angles)])
        pts.append(ring_pts)
    
    result = np.vstack(pts)
    return result[:n]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_pts = None
    best_score = -np.inf
    
    # Initial candidates
    candidates = []
    
    # 1. Regular 13-gon
    candidates.append(_regular_polygon(N, radius=1.0))
    
    # 2. Concentric rings
    candidates.append(_concentric_rings(N, n_rings=3))
    
    # 3. Concentric rings with different offset
    candidates.append(_concentric_rings(N, n_rings=4))
    
    # 4. Random points in unit disk
    rng = np.random.default_rng(N)
    for _ in range(5):
        angles = rng.uniform(0, 2*np.pi, N)
        radii = rng.uniform(0, 1, N)
        pts = np.column_stack([radii*np.cos(angles), radii*np.sin(angles)])
        candidates.append(pts)
    
    # 5. Grid-like points
    grid_pts = []
    for i in range(4):
        for j in range(4):
            grid_pts.append([i/3.0, j/3.0])
    grid_pts = np.array(grid_pts[:N])
    candidates.append(grid_pts)
    
    # Evaluate initial candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        try:
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_pts = cand.copy()
        except Exception:
            pass
    
    # Progressive beta annealing with L-BFGS-B
    betas = [5, 10, 20, 50, 100, 200]
    
    # Starting points for optimization
    start_points = []
    for cand in candidates:
        start_points.append(cand.flatten())
    
    # Add some perturbed versions
    for cand in candidates[:3]:
        for scale in [0.5, 1.5]:
            perturbed = cand * scale
            # Normalize to roughly unit area
            try:
                hull = ConvexHull(perturbed)
                if hull.volume > 0:
                    scale_factor = np.sqrt(1.0 / hull.volume)
                    perturbed = perturbed * scale_factor
                    start_points.append(perturbed.flatten())
            except Exception:
                pass
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        for sp in start_points:
            if time.time() > deadline:
                break
            
            try:
                result = minimize(
                    _soft_min_objective,
                    sp,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                
                if result.success or result.fun < 1e5:
                    x_opt = result.x
                    pts_opt = x_opt.reshape(-1, 2)
                    s = _score(pts_opt)
                    if s > best_score:
                        best_score = s
                        best_pts = pts_opt.copy()
                        start_points.append(x_opt)
                        
            except Exception:
                pass
    
    # Final local refinement with higher beta
    if best_pts is not None and time.time() < deadline:
        for beta in [500, 1000]:
            if time.time() > deadline:
                break
            try:
                result = minimize(
                    _soft_min_objective,
                    best_pts.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if result.fun < 1e5:
                    pts_opt = result.x.reshape(-1, 2)
                    s = _score(pts_opt)
                    if s > best_score:
                        best_score = s
                        best_pts = pts_opt.copy()
            except Exception:
                pass
    
    if best_pts is None:
        # Fallback: regular polygon
        best_pts = _regular_polygon(N, radius=1.0)
    
    return best_pts
# EVOLVE-BLOCK-END
