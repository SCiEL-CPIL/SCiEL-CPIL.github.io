"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time
from itertools import combinations

def _all_triangle_areas(points):
    """Compute all triangle areas vectorized."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                areas.append(area)
    return np.array(areas)

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    try:
        areas = _all_triangle_areas(points)
        if len(areas) == 0:
            return 0.0
        min_area = np.min(areas)
        hull_area = _convex_hull_area(points)
        if hull_area <= 0:
            return 0.0
        return min_area / hull_area
    except Exception:
        return 0.0

def _regular_polygon(n, radius=1.0):
    angles = 2 * np.pi * np.arange(n) / n
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _two_rings(n, r1=0.5, r2=1.0, offset=0):
    """Create two concentric rings of points."""
    n1 = n // 2
    n2 = n - n1
    angles1 = 2 * np.pi * np.arange(n1) / n1 + offset
    angles2 = 2 * np.pi * np.arange(n2) / n2 + offset + np.pi / n2
    pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
    return np.vstack([pts1, pts2])

def _hexagonal_packing(n):
    """Create a hexagonal-like packing arrangement."""
    points = []
    # Center point
    points.append([0.0, 0.0])
    # First ring
    for i in range(6):
        angle = i * np.pi / 3
        points.append([np.cos(angle), np.sin(angle)])
    # Second ring
    for i in range(6):
        angle = i * np.pi / 3
        points.append([2 * np.cos(angle), 2 * np.sin(angle)])
    # Add more to reach n
    while len(points) < n:
        angle = len(points) * 0.5
        r = 2.5
        points.append([r * np.cos(angle), r * np.sin(angle)])
    return np.array(points[:n])

def _soft_score(points, temperature=0.1):
    """Soft score using softmax over triangle areas for gradient-like signal."""
    try:
        areas = _all_triangle_areas(points)
        if len(areas) == 0:
            return 0.0
        hull_area = _convex_hull_area(points)
        if hull_area <= 0:
            return 0.0
        # Use log-sum-exp trick for numerical stability
        min_area = np.min(areas)
        # Soft minimum
        soft_min = -temperature * np.log(np.mean(np.exp(-areas / temperature)))
        return soft_min / hull_area
    except Exception:
        return 0.0

def _optimize_simulated_annealing(initial_points, deadline, rng, max_iter=5000):
    """Simulated annealing to maximize the minimum triangle area / hull area."""
    current = initial_points.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    T_start = 0.5
    T_end = 0.001
    T = T_start
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Cooling schedule
        T = T_start * (T_end / T_start) ** (it / max_iter)
        
        # Generate perturbation
        step_size = 0.1 * (1.0 + T)
        perturbation = rng.randn(N, 2) * step_size
        candidate = current + perturbation
        
        # Keep points in reasonable bounds
        candidate = np.clip(candidate, -3, 3)
        
        cand_score = _score(candidate)
        
        # Accept if better, or with probability exp(-delta/T)
        if cand_score > current_score:
            current = candidate
            current_score = cand_score
            if cand_score > best_score:
                best = candidate.copy()
                best_score = cand_score
        else:
            delta = current_score - cand_score
            if delta < 1e-10:
                accept_prob = 1.0
            else:
                accept_prob = np.exp(-delta / max(T, 1e-10))
            if rng.random() < accept_prob:
                current = candidate
                current_score = cand_score
    
    return best, best_score

def _optimize_local_fine(points, deadline, rng, max_iter=2000):
    """Local optimization with small steps."""
    current = points.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    for it in range(max_iter):
        if time.time() > deadline - 3:
            break
        
        step_size = 0.01 * (1.0 - it / max_iter) + 0.001
        perturbation = rng.randn(N, 2) * step_size
        candidate = current + perturbation
        candidate = np.clip(candidate, -3, 3)
        
        cand_score = _score(candidate)
        
        if cand_score > current_score:
            current = candidate
            current_score = cand_score
            if cand_score > best_score:
                best = candidate.copy()
                best_score = cand_score
        else:
            delta = current_score - cand_score
            if delta < 1e-10:
                accept_prob = 1.0
            else:
                accept_prob = np.exp(-delta / 0.01)
            if rng.random() < accept_prob:
                current = candidate
                current_score = cand_score
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    rng = np.random.RandomState(42)
    
    # Generate multiple diverse initial configurations
    initial_configs = []
    
    # Regular 14-gon
    initial_configs.append(_regular_polygon(N, radius=1.0))
    
    # Two rings with different radii
    for r1 in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for offset in [0, np.pi/14, np.pi/7]:
            initial_configs.append(_two_rings(N, r1=r1, r2=1.0, offset=offset))
    
    # Hexagonal-like packing
    initial_configs.append(_hexagonal_packing(N))
    
    # Random points in unit circle
    for seed in [1, 2, 3]:
        rng_init = np.random.RandomState(seed)
        angles = rng_init.uniform(0, 2*np.pi, N)
        radii = rng_init.uniform(0.3, 1.0, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initial_configs.append(pts)
    
    # Random points in unit square
    for seed in [4, 5]:
        rng_init = np.random.RandomState(seed)
        pts = rng_init.uniform(-1, 1, (N, 2))
        initial_configs.append(pts)
    
    best_points = None
    best_score = -np.inf
    
    # Evaluate all initial configurations
    for config in initial_configs:
        sc = _score(config)
        if sc > best_score:
            best_score = sc
            best_points = config.copy()
    
    # Run simulated annealing from the best initial config
    if best_points is not None:
        sa_best, sa_score = _optimize_simulated_annealing(
            best_points, deadline, rng, max_iter=3000
        )
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_best
    
    # Run local fine-tuning
    if time.time() < deadline - 10:
        local_best, local_score = _optimize_local_fine(
            best_points, deadline, rng, max_iter=2000
        )
        if local_score > best_score:
            best_score = local_score
            best_points = local_best
    
    # Try SA from a few other good initial configs if time remains
    if time.time() < deadline - 15:
        # Sort initial configs by score and try SA from top 3
        scored_configs = []
        for config in initial_configs:
            sc = _score(config)
            scored_configs.append((sc, config))
        scored_configs.sort(key=lambda x: -x[0])
        
        for sc, config in scored_configs[:3]:
            if time.time() > deadline - 5:
                break
            sa_best2, sa_score2 = _optimize_simulated_annealing(
                config, deadline, rng, max_iter=1000
            )
            if sa_score2 > best_score:
                best_score = sa_score2
                best_points = sa_best2
    
    return best_points
# EVOLVE-BLOCK-END
