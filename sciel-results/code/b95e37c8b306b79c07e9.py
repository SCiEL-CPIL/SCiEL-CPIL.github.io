"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import numpy as np
from scipy.spatial import ConvexHull
from itertools import combinations
import time

N = 13


def min_triangle_area(pts):
    """Compute minimum triangle area among all triples of points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
                if area < min_area:
                    min_area = area
    return min_area


def hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0


def score(pts):
    """Score = min triangle area / convex hull area."""
    ha = hull_area(pts)
    if ha < 1e-12:
        return 0.0
    return min_triangle_area(pts) / ha


def initial_grid_points():
    """Create a structured initial configuration: points on concentric regular polygons."""
    pts = []
    # Center point
    pts.append([0.0, 0.0])
    # Inner ring: 5 points on a circle of radius r1
    r1 = 0.35
    for i in range(5):
        angle = 2 * np.pi * i / 5
        pts.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    # Outer ring: 7 points on a circle of radius r2
    r2 = 0.85
    for i in range(7):
        angle = 2 * np.pi * i / 7 + np.pi / 7  # offset to avoid alignment
        pts.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    return np.array(pts)


def initial_random_points(seed):
    """Create random points in unit square."""
    rng = np.random.RandomState(seed)
    return rng.rand(N, 2)


def simulated_annealing(pts, time_limit, rng_seed=42):
    """Run simulated annealing to maximize the score."""
    rng = np.random.RandomState(rng_seed)
    start_time = time.time()
    
    current = pts.copy()
    current_score = score(current)
    best = current.copy()
    best_score = current_score
    
    # SA parameters
    T_start = 0.01
    T_end = 1e-6
    n_steps = 50000
    
    # We'll use a geometric cooling schedule
    if n_steps > 1:
        alpha = (T_end / T_start) ** (1.0 / n_steps)
    else:
        alpha = 1.0
    
    T = T_start
    
    step_size_start = 0.05
    step_size_end = 0.001
    
    for step in range(n_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Adaptive step size
        progress = step / n_steps
        step_size = step_size_start * (step_size_end / step_size_start) ** progress
        
        # Generate a perturbation
        new_pts = current.copy()
        # Pick a random point to move
        idx = rng.randint(0, N)
        new_pts[idx] += rng.normal(0, step_size, 2)
        
        # Optionally, also do a small global perturbation occasionally
        if rng.random() < 0.05:
            new_pts += rng.normal(0, step_size * 0.3, (N, 2))
        
        new_score = score(new_pts)
        
        # Accept or reject
        if new_score > current_score:
            current = new_pts
            current_score = new_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        else:
            # Metropolis criterion
            delta = current_score - new_score
            if T > 1e-10:
                accept_prob = np.exp(-delta / T)
                if rng.random() < accept_prob:
                    current = new_pts
                    current_score = new_score
        
        T *= alpha
    
    return best, best_score


def coordinate_descent_polish(pts, time_limit, max_iters=2000):
    """Polish with coordinate descent on the hard objective."""
    start_time = time.time()
    current = pts.copy()
    current_score = score(current)
    best = current.copy()
    best_score = current_score
    
    # Coordinate descent: for each point, try small moves in x and y
    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for ss in step_sizes:
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            improved = False
            for idx in range(N):
                for coord in range(2):
                    for direction in [-1, 1]:
                        trial = current.copy()
                        trial[idx, coord] += direction * ss
                        trial_score = score(trial)
                        if trial_score > current_score:
                            current = trial
                            current_score = trial_score
                            improved = True
                            if current_score > best_score:
                                best = current.copy()
                                best_score = current_score
            
            if not improved:
                break
    
    return best, best_score


def construct():
    """Main entry point: find 13 points maximizing min triangle area / hull area."""
    total_time_limit = 140.0  # Safety margin from 150s limit
    start_time = time.time()
    
    best_points = None
    best_score_val = 0.0
    
    # Strategy 1: Start from structured grid and run SA
    t_remaining = total_time_limit - (time.time() - start_time)
    if t_remaining > 5:
        pts_grid = initial_grid_points()
        # Normalize to have reasonable scale
        # The score is scale-invariant (ratio), so we don't need to normalize
        sa_pts, sa_score = simulated_annealing(pts_grid, t_remaining - 10, rng_seed=1)
        if sa_score > best_score_val:
            best_points = sa_pts
            best_score_val = sa_score
    
    # Strategy 2: Start from random and run SA
    t_remaining = total_time_limit - (time.time() - start_time)
    if t_remaining > 5:
        pts_rand = initial_random_points(42)
        sa_pts2, sa_score2 = simulated_annealing(pts_rand, t_remaining - 10, rng_seed=2)
        if sa_score2 > best_score_val:
            best_points = sa_pts2
            best_score_val = sa_score2
    
    # Strategy 3: Coordinate descent polish on the best found
    t_remaining = total_time_limit - (time.time() - start_time)
    if t_remaining > 2 and best_points is not None:
        cd_pts, cd_score = coordinate_descent_polish(best_points, t_remaining - 1)
        if cd_score > best_score_val:
            best_points = cd_pts
            best_score_val = cd_score
    
    # Final safety: if we have no valid points, return a default
    if best_points is None:
        best_points = initial_grid_points()
    
    return best_points
