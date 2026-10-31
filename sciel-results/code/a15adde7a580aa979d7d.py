"""Heilbronn convex regions: maximize min triangle area / hull area for 13 points."""
import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

N = 13

def _triangle_areas(points):
    """Compute all triangle areas for a set of points (n choose 3)."""
    n = points.shape[0]
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                areas.append(area)
    return np.array(areas)

def _min_triangle_area(points):
    """Compute minimum triangle area."""
    n = points.shape[0]
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute score: min triangle area / hull area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _normalize(points):
    """Normalize points to unit area convex hull."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return points
    # Shift to centroid
    centroid = points.mean(axis=0)
    points = points - centroid
    # Scale so hull area = 1
    scale = 1.0 / np.sqrt(hull_area)
    return points * scale

def _initial_grid():
    """Create a grid-like initial configuration for 13 points."""
    # Use a 4x4 grid with one point removed, or a structured pattern
    # Try a square grid approach
    points = []
    # 4x4 grid from -1 to 1
    coords = [-1.0, -0.333, 0.333, 1.0]
    for x in coords:
        for y in coords:
            points.append([x, y])
    # That's 16 points, take 13
    # Remove 3 corner points to get a more symmetric shape
    # Keep: (0,0), (0,1), (1,0), (1,1) and all edges
    # Actually let's use a different approach: start with a known good pattern
    
    # Use a regular polygon-like structure with interior points
    # 13 points: 8 on a circle, 4 interior, 1 center
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    r_outer = 1.0
    for a in angles:
        points.append([r_outer * np.cos(a), r_outer * np.sin(a)])
    # Interior points
    r_mid = 0.5
    angles_mid = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
    for a in angles_mid:
        points.append([r_mid * np.cos(a), r_mid * np.sin(a)])
    # Center
    points.append([0.0, 0.0])
    
    return np.array(points[:13])

def _perturb(points, step_size, rng):
    """Perturb points by small random amounts."""
    new_points = points.copy()
    for i in range(len(new_points)):
        new_points[i] += step_size * rng.gauss(0, 1, size=2)
    return new_points

def _local_search(points, max_iters=500, step_size=0.01, rng=None):
    """Local search using coordinate perturbation."""
    if rng is None:
        rng = np.random.default_rng()
    
    best_points = points.copy()
    best_score = _score(best_points)
    
    current_points = points.copy()
    current_score = best_score
    
    for it in range(max_iters):
        # Try perturbations in different directions
        improved = False
        for attempt in range(20):
            new_points = _perturb(current_points, step_size, rng)
            new_score = _score(new_points)
            
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_points = current_points.copy()
                    best_score = current_score
                break
        
        if not improved:
            # Reduce step size
            step_size *= 0.9
            if step_size < 1e-6:
                break
        
        # Occasionally jump to escape local minima
        if it % 50 == 0 and it > 0:
            jump_points = _perturb(current_points, step_size * 5, rng)
            jump_score = _score(jump_points)
            if jump_score > current_score * 0.9:  # Accept if not too bad
                current_points = jump_points
                current_score = jump_score
    
    return best_points, best_score

def construct():
    """Construct 13 points to maximize min triangle area / hull area."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    # Start with multiple initial configurations
    best_points = None
    best_score = 0.0
    
    # Initial configurations
    initial_configs = []
    
    # 1. Grid-based
    grid_points = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [0, 2], [1, 2], [2, 2], [3, 2],
        [1.5, 3]
    ], dtype=float)
    initial_configs.append(grid_points)
    
    # 2. Circle-based
    circle_points = _initial_grid()
    initial_configs.append(circle_points)
    
    # 3. Random
    for _ in range(3):
        random_points = rng.random((13, 2))
        initial_configs.append(random_points)
    
    # 4. Structured: hexagonal lattice
    hex_points = []
    for i in range(3):
        for j in range(3):
            x = i + 0.5 * (j % 2)
            y = j * 0.866
            hex_points.append([x, y])
    # Add 4 more points
    hex_points.append([0.5, 0.433])
    hex_points.append([1.5, 0.433])
    hex_points.append([0.5, 1.299])
    hex_points.append([1.5, 1.299])
    initial_configs.append(np.array(hex_points[:13]))
    
    # Run local search from each initial configuration
    for init_points in initial_configs:
        if time.time() > deadline:
            break
        
        # Normalize
        norm_points = _normalize(init_points)
        score = _score(norm_points)
        
        if score > best_score:
            best_points = norm_points.copy()
            best_score = score
        
        # Local search
        searched_points, searched_score = _local_search(
            norm_points, 
            max_iters=300, 
            step_size=0.05, 
            rng=rng
        )
        
        if searched_score > best_score:
            best_points = searched_points.copy()
            best_score = searched_score
    
    # Final polish with smaller steps
    if best_points is not None and time.time() < deadline:
        polished_points, polished_score = _local_search(
            best_points,
            max_iters=200,
            step_size=0.01,
            rng=rng
        )
        if polished_score > best_score:
            best_points = polished_points.copy()
            best_score = polished_score
    
    if best_points is None:
        # Fallback: return random points
        best_points = rng.random((13, 2))
    
    # Normalize final result
    best_points = _normalize(best_points)
    
    return best_points
