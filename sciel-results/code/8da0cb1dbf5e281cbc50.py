import time
import numpy as np
from scipy.spatial import ConvexHull
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

N = 13
_NUM_TRIPLES = len(list(combinations(range(N), 3)))

def _get_triangle_areas(points):
    """
    Vectorized calculation of areas for all triples.
    points: (N, 2) array
    returns: array of shape (num_triangles,)
    """
    n = len(points)
    indices = list(combinations(range(n), 3))
    if not indices:
        return np.array([])
    
    i_idx = np.array([p[0] for p in indices])
    j_idx = np.array([p[1] for p in indices])
    k_idx = np.array([p[2] for p in indices])
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return areas

def _hull_area(points):
    """Calculate convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate the raw score: min_triangle_area / hull_area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    areas = _get_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    min_area = np.min(areas)
    return min_area / hull_area

def _make_regular_polygon(n, radius=1.0, phase=0.0):
    """Generate a regular n-gon."""
    angles = 2 * np.pi * np.arange(n) / n + phase
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _coordinate_descent(points, deadline, max_rounds=50):
    """
    Coordinate descent: for each point, try small perturbations in 
    multiple directions and keep the one that maximizes the score.
    """
    pts = points.copy()
    best_score = _score(pts)
    no_improve_count = 0
    
    # Precompute which triangles each point participates in
    # For point i, it's in triangles (i,j,k) for all j<k, j!=i, k!=i
    # We'll just recompute areas each time for simplicity
    
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for round_num in range(max_rounds):
        if time.time() > deadline - 15:
            break
        
        improved = False
        for step_size in step_sizes:
            if time.time() > deadline - 15:
                break
            
            for i in range(N):
                if time.time() > deadline - 15:
                    break
                
                # Try moving point i in several directions
                # Radial and tangential directions relative to centroid
                centroid = np.mean(pts, axis=0)
                to_point = pts[i] - centroid
                dist = np.linalg.norm(to_point)
                
                if dist < 1e-10:
                    # At center, try random directions
                    directions = np.array([[1,0],[0,1],[1,1],[1,-1],[-1,0],[0,-1],[-1,-1],[-1,1]])
                else:
                    radial = to_point / dist
                    tangential = np.array([-radial[1], radial[0]])
                    directions = np.array([radial, -radial, tangential, -tangential,
                                          radial + tangential, radial - tangential,
                                          -radial + tangential, -radial - tangential])
                
                for d in directions:
                    new_pts = pts.copy()
                    new_pts[i] = pts[i] + step_size * d
                    
                    # Quick check: ensure convex hull area is reasonable
                    new_score = _score(new_pts)
                    if new_score > best_score + 1e-12:
                        pts = new_pts
                        best_score = new_score
                        improved = True
                        break  # Move on to next point
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 3:
                break
        else:
            no_improve_count = 0
    
    return pts, best_score

def _hill_climb_with_restarts(points, deadline, n_restarts=5):
    """
    Hill climbing with random restarts around the current best.
    """
    best_pts = points.copy()
    best_score = _score(best_pts)
    
    rng = np.random.RandomState(int(time.time() * 1000) % 10000)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        # Perturb all points slightly
        noise_scale = 0.05 * (1 - restart / n_restarts) + 0.01
        perturbed = best_pts + noise_scale * rng.normal(0, 1, (N, 2))
        
        # Coordinate descent from perturbed start
        cand_pts, cand_score = _coordinate_descent(perturbed, deadline, max_rounds=30)
        
        if cand_score > best_score:
            best_score = cand_score
            best_pts = cand_pts
    
    return best_pts, best_score

def _optimize_from_start(start_pts, deadline, rng):
    """
    Optimization from a starting point:
    1. Coordinate descent on hard score
    2. Hill climbing with restarts
    Returns best points and score found.
    """
    best_pts = start_pts.copy()
    best_score = _score(start_pts)
    
    # Stage 1: Coordinate descent
    if time.time() < deadline - 25:
        try:
            cand_pts, cand_score = _coordinate_descent(best_pts, deadline, max_rounds=50)
            if cand_score > best_score:
                best_score = cand_score
                best_pts = cand_pts
        except Exception:
            pass
    
    # Stage 2: Hill climbing with restarts
    if time.time() < deadline - 15:
        try:
            cand_pts, cand_score = _hill_climb_with_restarts(best_pts, deadline, n_restarts=5)
            if cand_score > best_score:
                best_score = cand_score
                best_pts = cand_pts
        except Exception:
            pass
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s kill
    
    best_score = 0.0
    best_points = None
    rng = np.random.RandomState(42)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Regular polygon
    initial_configs.append(_make_regular_polygon(N))
    
    # Regular polygon with phase shifts
    for phase in [np.pi/N, np.pi/(2*N), 2*np.pi/N]:
        initial_configs.append(_make_regular_polygon(N, phase=phase))
    
    # Slightly perturbed regular polygons
    for amp in [0.05, 0.1, 0.15, 0.2]:
        for freq in [1, 2, 3]:
            angles = 2 * np.pi * np.arange(N) / N
            radii = 1.0 + amp * np.sin(freq * angles)
            initial_configs.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))
    
    # Random configurations
    for _ in range(8):
        initial_configs.append(rng.uniform(-1, 1, (N, 2)))
    
    # Try each initial configuration with optimization
    for config in initial_configs:
        if time.time() > deadline - 25:
            break
        
        try:
            cand_pts, score = _optimize_from_start(config, deadline, rng)
            if score > best_score:
                best_score = score
                best_points = cand_pts
        except Exception:
            continue
    
    # If we have a good candidate, do more refinement
    if best_points is not None and time.time() < deadline - 30:
        for i in range(10):
            if time.time() > deadline - 10:
                break
            
            # Perturb the best solution and re-optimize
            noise_scale = 0.08 * (1 - i / 10.0) + 0.01
            perturbed = best_points + noise_scale * rng.normal(0, 1, (N, 2))
            
            try:
                cand_pts, score = _optimize_from_start(perturbed, deadline, rng)
                if score > best_score:
                    best_score = score
                    best_points = cand_pts
            except Exception:
                continue
    
    # Fallback to regular polygon if everything failed
    if best_points is None:
        best_points = _make_regular_polygon(N)
        
    return best_points
