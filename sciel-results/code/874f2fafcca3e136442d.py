# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from itertools import combinations

def _score_fast(points):
    """Compute A_min / A_hull using vectorized triangle area computation."""
    n = len(points)
    if n < 3:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    try:
        hull = ConvexHull(points)
        a_hull = hull.volume
    except Exception:
        return 0.0
    if a_hull <= 1e-12:
        return 0.0
    
    # Vectorized: compute all triangle areas at once
    # For each pair (i,j), compute cross product with all other points k
    # area(i,j,k) = 0.5 * |cross(P_j - P_i, P_k - P_i)|
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            # Vector over all k != i, j
            others = np.concatenate([points[:i], points[i+1:j], points[j+1:]])
            if len(others) == 0:
                continue
            # cross product of (P_j - P_i) with (P_k - P_i)
            d1 = points[j] - points[i]
            d2 = others - points[i]
            # 2D cross product: d1[0]*d2[1] - d1[1]*d2[0]
            crosses = d1[0] * d2[:, 1] - d1[1] * d2[:, 0]
            areas = np.abs(crosses) / 2.0
            min_area = min(min_area, np.min(areas))
    
    if min_area <= 1e-12:
        return 0.0
    return min_area / a_hull

def _local_search(points, rng, time_limit, max_iter=5000):
    """Local search with adaptive step size and restarts."""
    start = time.time()
    n = len(points)
    current = points.copy()
    current_score = _score_fast(current)
    if current_score <= 0:
        return current, 0.0
    
    step_size = 0.1
    no_improve_count = 0
    max_no_improve = 20
    
    for it in range(max_iter):
        if time.time() > start + time_limit:
            break
        if no_improve_count >= max_no_improve:
            step_size *= 0.5
            if step_size < 1e-6:
                break
            no_improve_count = 0
            max_no_improve = min(max_no_improve * 1.5, 100)
        
        # Pick a random point to move
        i = rng.integers(n)
        improved = False
        
        for _ in range(10):
            direction = rng.normal(size=2)
            direction /= (np.linalg.norm(direction) + 1e-12)
            for step in [step_size, step_size * 0.5, step_size * 0.25]:
                trial = current.copy()
                trial[i] = current[i] + step * direction
                s = _score_fast(trial)
                if s > current_score:
                    current = trial
                    current_score = s
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
    
    return current, current_score

def _generate_initializations(rng):
    """Generate diverse structured initializations."""
    n = 13
    inits = []
    
    # 1. Concentric polygons: 7 outer, 5 inner, 1 center
    outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    inner_angles = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
    pts = np.zeros((n, 2))
    for i in range(7):
        pts[i] = [np.cos(outer_angles[i]), np.sin(outer_angles[i])]
    for i in range(5):
        pts[7+i] = [0.5*np.cos(inner_angles[i]), 0.5*np.sin(inner_angles[i])]
    pts[12] = [0.0, 0.0]
    inits.append(pts)
    
    # 2. Fibonacci spiral
    golden_angle = np.pi * (3 - np.sqrt(5))
    pts = np.zeros((n, 2))
    for i in range(n):
        r = np.sqrt((i + 0.5) / n)
        theta = i * golden_angle
        pts[i] = [r * np.cos(theta), r * np.sin(theta)]
    inits.append(pts)
    
    # 3. Grid with perturbations
    grid = np.array([
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
        [0.0, 2.0], [1.0, 2.0], [2.0, 2.0],
        [0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5],
    ])
    inits.append(grid)
    
    # 4. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(pts)
    
    # 5. Two concentric rings (6+7)
    angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    angles2 = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
    pts = np.zeros((n, 2))
    for i in range(6):
        pts[i] = [np.cos(angles1[i]), np.sin(angles1[i])]
    for i in range(7):
        pts[6+i] = [0.6*np.cos(angles2[i]), 0.6*np.sin(angles2[i])]
    inits.append(pts)
    
    # 6. Random
    pts = rng.random((n, 2))
    inits.append(pts)
    
    # 7. Hexagonal lattice subset
    hex_pts = []
    for row in range(-2, 3):
        for col in range(-2, 3):
            x = col + 0.5 * (row % 2)
            y = row * 0.866
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts[:13])
    # Center it
    hex_pts -= hex_pts.mean(axis=0)
    inits.append(hex_pts)
    
    return inits

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the smallest triangle area
    relative to the convex hull area.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill
    
    n = 13
    rng = np.random.default_rng(seed=42)
    
    best_points = None
    best_score = 0.0
    
    # Generate diverse initializations
    inits = _generate_initializations(rng)
    
    # Score all initializations and sort by score
    scored_inits = []
    for init in inits:
        s = _score_fast(init)
        scored_inits.append((s, init))
    
    # Sort by score descending
    scored_inits.sort(key=lambda x: -x[0])
    
    # Allocate time to top candidates for local search
    # Give more time to better initializations
    total_time_budget = deadline - start_time - 30  # 30s safety
    num_candidates = min(len(scored_inits), 5)
    
    # Adaptive time allocation: top gets more time
    weights = [1.0 / (i + 1) for i in range(num_candidates)]
    total_weight = sum(weights)
    times = [total_time_budget * w / total_weight for w in weights]
    
    for idx in range(num_candidates):
        if time.time() > deadline - 10:
            break
        
        s, init = scored_inits[idx]
        if s > best_score:
            best_score = s
            best_points = init.copy()
        
        # Run local search with allocated time
        time_for_this = min(times[idx], deadline - time.time() - 10)
        if time_for_this < 5:
            continue
        
        result_pts, result_score = _local_search(init, rng, time_for_this, max_iter=10000)
        
        if result_score > best_score:
            best_score = result_score
            best_points = result_pts.copy()
        
        # If we used less time than allocated, try another round with smaller steps
        remaining = deadline - time.time() - 10
        if remaining > 10 and result_score > 0:
            result_pts2, result_score2 = _local_search(result_pts, rng, min(remaining * 0.5, 60), max_iter=5000)
            if result_score2 > best_score:
                best_score = result_score2
                best_points = result_pts2.copy()
    
    # Final validation
    if best_points is None or not np.all(np.isfinite(best_points)):
        # Fallback
        fallback = np.array([
            [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
            [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
            [0.0, 2.0], [1.0, 2.0], [2.0, 2.0],
            [0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5],
        ], dtype=float)
        return fallback
    
    final_score = _score_fast(best_points)
    if final_score <= 0:
        fallback = np.array([
            [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
            [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
            [0.0, 2.0], [1.0, 2.0], [2.0, 2.0],
            [0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5],
        ], dtype=float)
        return fallback
    
    return best_points

# EVOLVE-BLOCK-END
