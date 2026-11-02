"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

S3H = np.sqrt(3) / 2
TRI_AREA = S3H / 2.0
DEADLINE = 145  # seconds, safety margin from 150s limit

# Vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

def _min_area_normalized(points):
    """Compute minimum triangle area normalized by the unit equilateral triangle area.
    Uses vectorized operations for speed.
    """
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    # dx[i, j] = x[j] - x[i]
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    
    # Calculate cross product terms for all i, j, k
    # term1[i, j, k] = dx[i, j] * dy[i, k]
    # term2[i, j, k] = dy[i, j] * dx[i, k]
    term1 = dx[:, :, None] * dy[:, None, :]
    term2 = dy[:, :, None] * dx[:, None, :]
    cross = term1 - term2
    areas = 0.5 * np.abs(cross)
    
    # Extract i < j < k
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    valid_areas = areas[mask]
    
    if len(valid_areas) == 0:
        return 0.0
    
    min_area = np.min(valid_areas)
    return min_area / TRI_AREA

def _is_inside_triangle(p):
    """Check if point p is inside the equilateral triangle."""
    x, y = p
    eps = 1e-9
    if y < -eps or y > S3H + eps:
        return False
    if x < -eps or x > 1 + eps:
        return False
    if y > S3H * (1 - x) + eps:
        return False
    if y > S3H * x + eps:
        return False
    return True

def _penalized_min_area(x):
    """Objective: minimize negative min area plus penalty for out-of-bounds."""
    points = x.reshape(-1, 2)
    
    # Penalty for points outside triangle
    penalty = 0.0
    for i in range(N):
        px, py = points[i]
        d_bottom = -py
        d_top = py - S3H
        d_left = -px
        d_right = px - 1
        d_slant_right = py - S3H * (1 - px)
        d_slant_left = py - S3H * px
        
        dist = max(0.0, d_bottom, d_top, d_left, d_right, d_slant_right, d_slant_left)
        penalty += dist ** 2
    
    min_a = _min_area_normalized(points)
    # Weight penalty heavily to ensure validity, but not so much it dominates the landscape entirely
    return -min_a + 10.0 * penalty

def _random_point_in_triangle(rng):
    """Generate a random point inside the triangle using rejection sampling."""
    while True:
        u, v = rng.random(), rng.random()
        if u + v < 1:
            x = u + 0.5 * v
            y = S3H * v
            # Ensure strictly inside to avoid boundary issues in optimization
            if 0.01 < x < 0.99 and 0.01 < y < S3H * 0.99:
                if y < S3H * (1 - x) * 0.99 and y < S3H * x * 0.99:
                    return np.array([x, y])

def _farthest_point_sampling(rng, n_points):
    """Generate n_points inside the triangle using Farthest Point Sampling."""
    points = []
    # Start with a random point near the center
    start_x = 0.5 + (rng.random() - 0.5) * 0.1
    start_y = S3H / 3.0 + (rng.random() - 0.5) * 0.1
    if not _is_inside_triangle([start_x, start_y]):
        start_x, start_y = 0.5, S3H / 3.0
    points.append(np.array([start_x, start_y]))
    
    for _ in range(n_points - 1):
        best_point = None
        best_min_dist = -1
        
        # Generate many candidates
        for _ in range(500):
            c = _random_point_in_triangle(rng)
            # Calculate min distance to existing points
            min_dist = np.inf
            for p in points:
                d = np.linalg.norm(c - p)
                if d < min_dist:
                    min_dist = d
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_point = c
                
        if best_point is not None:
            points.append(best_point)
        else:
            points.append(_random_point_in_triangle(rng))
            
    return np.array(points)

def _local_polish(points, start_time, deadline_margin=5):
    """Polish a set of points using Powell followed by Nelder-Mead."""
    if time.time() - start_time > DEADLINE - deadline_margin:
        return points, _min_area_normalized(points)
    
    best_points = points.copy()
    best_score = _min_area_normalized(points)
    x0 = best_points.flatten()
    
    # Step 1: Powell's Method
    # Powell is robust for non-smooth functions and handles constraints via penalty well
    try:
        res_powell = minimize(
            _penalized_min_area,
            x0,
            method='Powell',
            options={'maxiter': 300, 'maxfev': 3000, 'ftol': 1e-12, 'xtol': 1e-10}
        )
        if time.time() - start_time > DEADLINE - deadline_margin:
            return best_points, best_score
            
        pts = res_powell.x.reshape(-1, 2)
        valid = all(_is_inside_triangle(p) for p in pts)
        if valid:
            score = _min_area_normalized(pts)
            if score > best_score:
                best_points = pts
                best_score = score
                x0 = pts.flatten()
    except Exception:
        pass
    
    # Step 2: Nelder-Mead for fine polishing
    if time.time() - start_time < DEADLINE - deadline_margin:
        try:
            res_nm = minimize(
                _penalized_min_area,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            pts = res_nm.x.reshape(-1, 2)
            valid = all(_is_inside_triangle(p) for p in pts)
            if valid:
                score = _min_area_normalized(pts)
                if score > best_score:
                    best_points = pts
                    best_score = score
        except Exception:
            pass
            
    return best_points, best_score

def construct():
    start_time = time.time()
    
    best_points = None
    best_score = -float('inf')
    
    # Phase 1: Differential Evolution
    # The vectorized objective is fast, so we can afford a decent population size.
    # We limit the time for this phase to ensure we have time for local polishing.
    de_time_limit = 60.0
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))  # x
        bounds.append((0, S3H))  # y
    
    def de_callback(xk, convergence=None):
        return time.time() - start_time > start_time + de_time_limit

    try:
        result = differential_evolution(
            _penalized_min_area,
            bounds,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            callback=de_callback,
            polish=False, # We will do our own polishing
            init='latinhypercube'
        )
        
        if time.time() - start_time < DEADLINE - 10:
            pts = result.x.reshape(-1, 2)
            valid = all(_is_inside_triangle(p) for p in pts)
            if valid:
                score = _min_area_normalized(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
    except Exception:
        pass
    
    # Phase 2: Local Polish on DE result and Random/FPS restarts
    # We use the remaining time to refine the best candidate and explore a few others.
    rng = np.random.RandomState(42)
    
    # First, polish the best candidate from DE
    if best_points is not None:
        best_points, best_score = _local_polish(best_points, start_time, deadline_margin=10)
    
    # Then, try a few FPS-based starts and polish them
    # FPS provides a good spread which might find a better local minimum than DE's single best
    n_restarts = 10
    for r in range(n_restarts):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Generate initial config using FPS
        init_pts = _farthest_point_sampling(rng, N)
        
        # Polish this candidate
        polished_pts, score = _local_polish(init_pts, start_time, deadline_margin=10)
        
        if score > best_score:
            best_score = score
            best_points = polished_pts
    
    # Fallback if no valid points found (should not happen with penalty)
    if best_points is None:
        best_points = _farthest_point_sampling(rng, N)
        best_score = _min_area_normalized(best_points)
        
    return best_points
# EVOLVE-BLOCK-END
