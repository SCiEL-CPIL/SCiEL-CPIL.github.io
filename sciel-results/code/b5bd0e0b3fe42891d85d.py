# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools
import math

N = 13
# Precompute triangle indices for efficiency
TRI_IDX = np.array(list(itertools.combinations(range(N), 3)))

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull
    Returns 0.0 if invalid
    """
    if not np.all(np.isfinite(points)):
        return 0.0
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
        
    p1 = points[TRI_IDX[:, 0]]
    p2 = points[TRI_IDX[:, 1]]
    p3 = points[TRI_IDX[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    A_min = np.min(areas)
    
    return A_min / A_hull

def _compute_areas(points):
    """Compute all 286 triangle areas (vectorized)."""
    p1 = points[TRI_IDX[:, 0]]
    p2 = points[TRI_IDX[:, 1]]
    p3 = points[TRI_IDX[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return np.abs(cross) * 0.5

def _soft_min_score(points, temperature=0.01):
    """
    Smooth surrogate: minimize -soft_min_area / A_hull.
    """
    if not np.all(np.isfinite(points)):
        return 1e10
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 1e10
    
    if A_hull <= 0:
        return 1e10
        
    areas = _compute_areas(points)
    
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_area = -temperature * log_sum
    
    return -soft_min_area / A_hull

def generate_initial_candidates(rng):
    """
    Generate diverse initial candidates based on geometric templates.
    """
    candidates = []
    
    # Template 1: Regular 13-gon
    angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(pts)
    
    # Template 2: 1 center + 12 outer (regular 12-gon)
    center = np.array([0.0, 0.0])
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    outer = np.column_stack([np.cos(angles), np.sin(angles)])
    pts = np.vstack([center, outer])
    candidates.append(pts)
    
    # Template 3: 1 center + 6 inner + 6 outer
    center = np.array([0.0, 0.0])
    angles_in = np.linspace(0, 2*np.pi, 6, endpoint=False)
    inner = 0.5 * np.column_stack([np.cos(angles_in), np.sin(angles_in)])
    angles_out = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    outer = 1.0 * np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    pts = np.vstack([center, inner, outer])
    candidates.append(pts)
    
    # Template 4: 2 center + 5 inner + 6 outer
    c1 = np.array([-0.1, 0.0])
    c2 = np.array([0.1, 0.0])
    angles_in = np.linspace(0, 2*np.pi, 5, endpoint=False)
    inner = 0.5 * np.column_stack([np.cos(angles_in), np.sin(angles_in)])
    angles_out = np.linspace(0, 2*np.pi, 6, endpoint=False)
    outer = 1.0 * np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    pts = np.vstack([c1, c2, inner, outer])
    candidates.append(pts)
    
    # Template 5: Grid-like (4x4 grid, remove 3 points to get 13, centered)
    xs = np.array([-1, -0.33, 0.33, 1.0])
    ys = np.array([-1, -0.33, 0.33, 1.0])
    grid = np.array([[x, y] for x in xs for y in ys])
    pts = grid[:13]
    candidates.append(pts)
    
    # Template 6: Random
    pts = rng.uniform(-1, 1, size=(13, 2))
    candidates.append(pts)
    
    # Template 7: Fibonacci spiral
    golden_angle = 2.39996
    radii = np.sqrt(np.arange(13) / 13.0)
    angles = np.arange(13) * golden_angle
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    candidates.append(pts)
    
    # Template 8: Perturbed Grid
    xs = np.array([-1, -0.33, 0.33, 1.0])
    ys = np.array([-1, -0.33, 0.33, 1.0])
    grid = np.array([[x, y] for x in xs for y in ys])
    indices = rng.choice(16, 13, replace=False)
    pts = grid[indices]
    candidates.append(pts)
    
    # Template 9: Two rings
    r1 = 0.5
    r2 = 1.0
    angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    angles2 = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
    ring1 = r1 * np.column_stack([np.cos(angles1), np.sin(angles1)])
    ring2 = r2 * np.column_stack([np.cos(angles2), np.sin(angles2)])
    pts = np.vstack([ring1, ring2])
    candidates.append(pts)
    
    return candidates

def _fast_coordinate_descent(points, time_limit, initial_step=0.05):
    """
    Fast coordinate descent: for each point, try 8 directions.
    Uses vectorized area computation. Much faster than the original.
    """
    start_time = time.time()
    current_points = points.copy()
    current_score = compute_score(current_points)
    step_size = initial_step
    directions = np.array([
        [-1, 0], [1, 0], [0, -1], [0, 1],
        [-1, -1], [1, -1], [-1, 1], [1, 1]
    ], dtype=float)
    directions /= np.sqrt(2.0)
    
    while time.time() - start_time < time_limit:
        improved = False
        for idx in range(N):
            for d in directions:
                new_points = current_points.copy()
                new_points[idx] += d * step_size
                
                if not np.all(np.isfinite(new_points)):
                    continue
                try:
                    hull = ConvexHull(new_points)
                    if hull.volume <= 0:
                        continue
                except Exception:
                    continue
                
                new_score = compute_score(new_points)
                if new_score > current_score + 1e-12:
                    current_points = new_points
                    current_score = new_score
                    improved = True
        
        if not improved:
            step_size *= 0.5
            if step_size < 1e-6:
                break
    return current_points, current_score

def heilbronn_convex13() -> np.ndarray:
    """
    Optimize 13 points to maximize the minimum triangle area ratio.
    Uses multi-start with soft-min annealing + fast coordinate descent polish.
    """
    start_time = time.time()
    deadline = start_time + 585.0  # 15s safety margin before 600s kill
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -np.inf
    
    # Generate initial candidates
    candidates = generate_initial_candidates(rng)
    
    # Add perturbed versions
    for base in candidates[:6]:
        for scale in [0.05, 0.15, 0.3, 0.5]:
            perturbed = base + rng.normal(0, scale, size=base.shape)
            candidates.append(perturbed)
    
    rng.shuffle(candidates)
    
    n_candidates = len(candidates)
    
    # Phase 1: Powell optimization with soft-min annealing
    # Allocate 70% of budget to search, 30% to polish
    search_deadline = start_time + (deadline - start_time) * 0.70
    
    # Temperature schedule: start high (smooth), end low (sharp)
    temps = [0.1, 0.05, 0.02, 0.01, 0.005]
    
    for i, init_pts in enumerate(candidates):
        current_time = time.time()
        if current_time > search_deadline:
            break
            
        # Normalize
        init_pts_centered = init_pts - np.mean(init_pts, axis=0)
        max_norm = np.max(np.linalg.norm(init_pts_centered, axis=1))
        if max_norm > 1e-6:
            init_pts_scaled = init_pts_centered / max_norm
        else:
            init_pts_scaled = init_pts_centered
        
        x0 = init_pts_scaled.flatten()
        
        # Try a few temperature levels; stop early if score is good
        for temp in temps:
            current_time = time.time()
            if current_time > search_deadline:
                break
            
            time_left = search_deadline - current_time
            if time_left < 2.0:
                break
            
            # Give less time to later (colder) temperatures
            frac = 1.0 / len(temps)
            max_iter = int(200 * frac * (1.0 + temp))  # more iters for higher temp
            max_iter = max(max_iter, 50)
            
            def obj_func(x_flat, _temp=temp):
                pts = x_flat.reshape(N, 2)
                return _soft_min_score(pts, temperature=_temp)
            
            try:
                res = minimize(
                    obj_func,
                    x0,
                    method='Powell',
                    options={'maxiter': max_iter, 'xtol': 1e-6, 'ftol': 1e-6, 'disp': False},
                    timeout=min(time_left * 0.9, 30.0)
                )
                x0 = res.x
            except Exception:
                pass
        
        # Convert back
        pts_opt = x0.reshape(N, 2)
        score = compute_score(pts_opt)
        
        if score > best_score:
            best_score = score
            best_points = pts_opt.copy()
        
        # Quick polish after each candidate to keep best_score accurate
        current_time = time.time()
        if current_time < search_deadline - 1.0 and best_score > 0:
            polish_t = min(3.0, search_deadline - current_time - 0.5)
            if polish_t > 0.5:
                pp, ps = _fast_coordinate_descent(best_points, polish_t, initial_step=0.02)
                if ps > best_score:
                    best_score = ps
                    best_points = pp.copy()
    
    # Phase 2: Thorough polish of the best solution
    if best_points is not None:
        polish_deadline = deadline - 3.0
        current_time = time.time()
        if current_time < polish_deadline:
            polish_time = polish_deadline - current_time
            
            # Multi-stage polish: start with larger steps, then shrink
            stages = [
                (0.03, 0.35),  # step, fraction of time
                (0.01, 0.25),
                (0.005, 0.20),
                (0.001, 0.10),
                (0.0003, 0.10),
            ]
            
            for step_init, frac in stages:
                current_time = time.time()
                if current_time >= polish_deadline:
                    break
                t_alloc = polish_time * frac
                t_alloc = min(t_alloc, polish_deadline - current_time - 0.5)
                if t_alloc < 0.3:
                    break
                
                pp, ps = _fast_coordinate_descent(best_points, t_alloc, initial_step=step_init)
                if ps > best_score:
                    best_score = ps
                    best_points = pp.copy()
            
            # Final ultra-fine polish
            current_time = time.time()
            if current_time < polish_deadline - 0.5:
                t_final = polish_deadline - current_time - 0.5
                if t_final > 0.5:
                    pp, ps = _fast_coordinate_descent(best_points, t_final, initial_step=0.0001)
                    if ps > best_score:
                        best_score = ps
                        best_points = pp.copy()
    
    # Fallback
    if best_points is None:
        best_points = candidates[0]
        
    # Ensure valid output
    if not np.all(np.isfinite(best_points)):
        best_points = np.column_stack([np.cos(np.linspace(0, 2*np.pi, 13, endpoint=False)), np.sin(np.linspace(0, 2*np.pi, 13, endpoint=False))])
        
    return best_points

# EVOLVE-BLOCK-END
