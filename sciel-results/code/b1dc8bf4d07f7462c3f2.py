# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

# Precompute triangle indices for 13 points
N_POINTS = 13
TRIANGLE_IDX = np.array(list(itertools.combinations(range(N_POINTS), 3)))

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull (vectorized)
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Check for valid hull
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
        
    # Calculate all triangle areas using precomputed indices
    p1 = points[TRIANGLE_IDX[:, 0]]
    p2 = points[TRIANGLE_IDX[:, 1]]
    p3 = points[TRIANGLE_IDX[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    A_min = np.min(areas)
    
    return A_min / A_hull

def soft_min_score(points, temperature=0.01):
    """
    Smooth surrogate using log-sum-exp for soft-min of triangle areas.
    Returns a differentiable-ish proxy that is smoother than hard min.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
    
    p1 = points[TRIANGLE_IDX[:, 0]]
    p2 = points[TRIANGLE_IDX[:, 1]]
    p3 = points[TRIANGLE_IDX[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    # Soft min: -T * log(sum(exp(-areas/T)))
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_area = -temperature * log_sum
    
    return soft_min_area / A_hull

def gradient_descent_optimize(points, time_limit, lr=0.01, patience=50):
    """
    Simple gradient descent using finite differences on the soft-min surrogate.
    Faster than Nelder-Mead for this problem.
    """
    points = points.copy()
    n = len(points)
    
    def get_score(pts):
        if not np.all(np.isfinite(pts)):
            return 1e10
        return -soft_min_score(pts, temperature=0.01)
    
    current_score = get_score(points)
    best_score = current_score
    best_points = points.copy()
    no_improve = 0
    
    while time.time() < time_limit and no_improve < patience:
        # Compute gradient via central differences
        grad = np.zeros_like(points)
        eps = 1e-5
        for i in range(n):
            for j in range(2):
                pts_plus = points.copy()
                pts_minus = points.copy()
                pts_plus[i, j] += eps
                pts_minus[i, j] -= eps
                grad[i, j] = (get_score(pts_plus) - get_score(pts_minus)) / (2 * eps)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
        
        # Update
        new_points = points - lr * grad
        
        new_score = get_score(new_points)
        if new_score < current_score:
            points = new_points
            current_score = new_score
            no_improve = 0
            if current_score < best_score:
                best_score = current_score
                best_points = points.copy()
            # Decay learning rate slowly
            lr *= 0.999
        else:
            no_improve += 1
            lr *= 0.95
    
    # Final hard score
    hard_score = compute_score(best_points)
    return best_points, hard_score

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area ratio.
    
    Uses a combination of structured initial configurations and local search
    with random perturbations, inspired by high-scoring solutions.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin
    
    best_points = None
    best_score = -1.0
    
    # Generate initial candidates
    candidates = []
    
    # Candidate 1: Regular 12-gon + center
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    pts1 = np.column_stack([np.cos(angles), np.sin(angles)])
    pts1 = np.vstack([np.array([0.0, 0.0]), pts1])
    candidates.append(pts1)
    
    # Candidate 2: Two concentric hexagons + center
    # Hexagon 1: radius 0.5, angles 0, 60, ...
    angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    pts_h1 = np.column_stack([0.5 * np.cos(angles1), 0.5 * np.sin(angles1)])
    # Hexagon 2: radius 1.0, angles 30, 90, ...
    angles2 = angles1 + np.pi/6
    pts_h2 = np.column_stack([1.0 * np.cos(angles2), 1.0 * np.sin(angles2)])
    pts2 = np.vstack([np.array([0.0, 0.0]), pts_h1, pts_h2])
    candidates.append(pts2)
    
    # Candidate 3: Two concentric polygons (7 and 6)
    def make_concentric_polygons(n1=7, n2=6, r1=1.0, r2=0.5):
        angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
        angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n2
        p1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        p2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        return np.vstack([p1, p2])
    
    candidates.append(make_concentric_polygons(7, 6, 1.0, 0.5))
    candidates.append(make_concentric_polygons(6, 7, 1.0, 0.6))
    
    # Candidate 4: Fibonacci-like distribution
    def make_fibonacci_2d():
        n = 13
        golden_ratio = (1 + np.sqrt(5)) / 2
        angles = 2 * np.pi * np.arange(n) / golden_ratio
        radii = np.sqrt(np.arange(n) / n)
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return points
    
    candidates.append(make_fibonacci_2d())
    
    # Candidate 5: Grid-like arrangement
    def make_grid_arrangement():
        # 4x3 grid with one extra point
        xs = np.array([0, 1, 2, 3])
        ys = np.array([0, 1, 2, 3])
        # Take 13 points from a 4x4 grid
        points = []
        for i in range(4):
            for j in range(4):
                if len(points) < 13:
                    points.append([i, j])
        points = np.array(points)
        # Center and scale
        points = (points - points.mean(axis=0)) / points.std(axis=0)
        return points
    
    candidates.append(make_grid_arrangement())
    
    # Add random perturbations to structured configs
    rng = np.random.default_rng(12345)
    for base in candidates[:3]:
        perturb = rng.normal(0, 0.05, size=base.shape)
        candidates.append(base + perturb)
    
    # Add some fully random starts
    for _ in range(5):
        pts_rand = rng.random((13, 2)) * 2 - 1
        candidates.append(pts_rand)
    
    # Candidate: Ring with jittered inner points
    def make_ring_inner():
        outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
        outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
        inner_angles = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
        inner = np.column_stack([0.55 * np.cos(inner_angles), 0.55 * np.sin(inner_angles)])
        return np.vstack([outer, inner])
    
    candidates.append(make_ring_inner())
    candidates.append(make_ring_inner() + rng.normal(0, 0.03, size=(13,2)))
    
    # Candidate: Alternating radii pattern
    def make_alternating():
        angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
        radii = np.ones(13)
        radii[::2] = 1.0
        radii[1::2] = 0.6
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return pts
    
    candidates.append(make_alternating())
    candidates.append(make_alternating() + rng.normal(0, 0.03, size=(13,2)))
    
    # Candidate: Known good Heilbronn-like configuration
    # Based on literature: mix of boundary and interior points
    def make_heilbronn_like():
        # 7 points on outer ring (heptagon)
        outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
        outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
        
        # 5 points in inner pentagon, rotated
        inner_angles = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
        inner_r = 0.45
        inner = np.column_stack([inner_r * np.cos(inner_angles), inner_r * np.sin(inner_angles)])
        
        # 1 center point
        center = np.array([[0.0, 0.0]])
        
        return np.vstack([outer, inner, center])
    
    candidates.append(make_heilbronn_like())
    candidates.append(make_heilbronn_like() + rng.normal(0, 0.02, size=(13,2)))
    
    # Candidate: 6 outer + 6 inner + 1 center
    def make_6_6_1():
        outer_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
        
        inner_angles = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
        inner_r = 0.5
        inner = np.column_stack([inner_r * np.cos(inner_angles), inner_r * np.sin(inner_angles)])
        
        center = np.array([[0.0, 0.0]])
        
        return np.vstack([outer, inner, center])
    
    candidates.append(make_6_6_1())
    candidates.append(make_6_6_1() + rng.normal(0, 0.02, size=(13,2)))
    
    # Local search function
    def local_search(points, time_limit):
        points = points.copy()
        current_score = compute_score(points)
        best_score_local = current_score
        best_points_local = points.copy()
        
        rng = np.random.default_rng(seed=42)
        
        # Phase 1: Random perturbation local search (50% of time)
        step_size = 0.1
        phase1_deadline = time.time() + (time_limit - time.time()) * 0.5
        
        while time.time() < phase1_deadline:
            improved = False
            for idx in range(13):
                dx = rng.uniform(-step_size, step_size)
                dy = rng.uniform(-step_size, step_size)
                
                new_points = points.copy()
                new_points[idx, 0] += dx
                new_points[idx, 1] += dy
                
                new_score = compute_score(new_points)
                
                if new_score > current_score:
                    points = new_points
                    current_score = new_score
                    improved = True
                    
                    if current_score > best_score_local:
                        best_score_local = current_score
                        best_points_local = points.copy()
            
            if not improved:
                step_size *= 0.9
                if step_size < 1e-6:
                    break
        
        # Phase 2: Gradient descent polish (remaining time)
        if time.time() < time_limit:
            gd_deadline = time_limit - 5.0
            gd_pts, gd_score = gradient_descent_optimize(best_points_local, gd_deadline)
            if gd_score > best_score_local:
                best_score_local = gd_score
                best_points_local = gd_pts
        
        return best_points_local, best_score_local
    
    # Run local search on each candidate
    n_candidates = len(candidates)
    # Give more time to better candidates; use adaptive allocation
    base_time = 15.0  # seconds per candidate initially
    
    for i, init_pts in enumerate(candidates):
        if time.time() >= deadline:
            break
            
        # Allocate time: more for later candidates if we're doing well
        remaining_time = deadline - time.time()
        remaining_candidates = n_candidates - i
        
        # Use at least base_time but cap at remaining
        candidate_time = min(base_time, remaining_time / max(1, remaining_candidates))
        # If this is the last few candidates, give more time
        if remaining_candidates <= 3:
            candidate_time = min(remaining_time / remaining_candidates, 60.0)
        
        candidate_deadline = min(deadline, time.time() + candidate_time)
        
        res_pts, res_score = local_search(init_pts, candidate_deadline)
        
        if res_score > best_score:
            best_score = res_score
            best_points = res_pts
            # If we found a good score, increase base time for subsequent
            if best_score > 75.0:
                base_time = 20.0
                
    # If we have time left, do a final polish on the best point
    if best_points is not None and time.time() < deadline - 10.0:
        # Run another round of local search on the best found
        res_pts, res_score = local_search(best_points, deadline - 10.0)
        if res_score > best_score:
            best_score = res_score
            best_points = res_pts
    
    # Final fallback
    if best_points is None:
        best_points = candidates[0]
    
    # Ensure all points are finite
    if not np.all(np.isfinite(best_points)):
        best_points = candidates[0]
    
    return best_points

# EVOLVE-BLOCK-END
