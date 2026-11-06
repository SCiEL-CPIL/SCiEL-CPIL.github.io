import time
import random
import numpy as np
from collections import Counter
from itertools import combinations

DEADLINE = 85.0  # safety margin before 90s kill

def _doubled_areas(points):
    """Compute all positive doubled areas for a set of points.
    Returns a list of positive doubled areas (absolute determinant)."""
    n = len(points)
    areas = []
    # Vectorized approach for speed
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                det = abs(x1*(y2 - y3) - y1*(x2 - x3) + x2*y3 - x3*y2)
                if det > 0:
                    areas.append(det)
    return areas

def _max_multiplicity(points):
    """Compute the maximum multiplicity of any single positive doubled area."""
    areas = _doubled_areas(points)
    if not areas:
        return 0
    counts = Counter(areas)
    return max(counts.values())

def _fast_score(points):
    """Fast scoring using numpy for vectorized computation."""
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    # Compute all cross products efficiently
    # For each pair (i,j), compute vector, then cross with all other points
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            # Vector from i to j
            vx = pts[j, 0] - pts[i, 0]
            vy = pts[j, 1] - pts[i, 1]
            # Cross product with all points k > j
            for k in range(j + 1, n):
                wx = pts[k, 0] - pts[i, 0]
                wy = pts[k, 1] - pts[i, 1]
                det = abs(vx * wy - vy * wx)
                if det > 0:
                    areas.append(det)
    if not areas:
        return 0
    # Use numpy for counting
    arr = np.array(areas)
    vals, counts = np.unique(arr, return_counts=True)
    return int(counts.max())

def _generate_grid_config():
    """Generate a grid configuration as a starting point."""
    # 6x5 grid centered
    points = []
    for x in range(-3, 3):
        for y in range(-2, 3):
            points.append((x, y))
    return points

def _random_point_in_bounds():
    """Generate a random point in [-30, 30]^2."""
    return (random.randint(-30, 30), random.randint(-30, 30))

def _local_search(points, max_iters=5000, time_limit=80.0):
    """Local search: try moving individual points to improve max multiplicity."""
    start_time = time.time()
    current_points = list(points)
    current_score = _fast_score(current_points)
    best_points = list(current_points)
    best_score = current_score
    
    for it in range(max_iters):
        if time.time() > time_limit:
            break
        if time.time() > DEADLINE:
            break
        
        # Pick a random point to move
        idx = random.randint(0, len(current_points) - 1)
        old_pt = current_points[idx]
        
        # Try a few candidate moves
        improved = False
        for _ in range(20):
            # Random new position
            new_pt = _random_point_in_bounds()
            if new_pt == old_pt:
                continue
            if new_pt in current_points:
                continue
            
            current_points[idx] = new_pt
            score = _fast_score(current_points)
            
            if score > current_score:
                current_score = score
                if score > best_score:
                    best_score = score
                    best_points = list(current_points)
                improved = True
                break
            else:
                current_points[idx] = old_pt
        
        # Occasionally try a small perturbation
        if random.random() < 0.3:
            idx = random.randint(0, len(current_points) - 1)
            old_pt = current_points[idx]
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            new_pt = (old_pt[0] + dx, old_pt[1] + dy)
            if -30 <= new_pt[0] <= 30 and -30 <= new_pt[1] <= 30:
                if new_pt not in current_points:
                    current_points[idx] = new_pt
                    score = _fast_score(current_points)
                    if score >= current_score:
                        if score > best_score:
                            best_score = score
                            best_points = list(current_points)
                    else:
                        current_points[idx] = old_pt
    
    return best_points, best_score

def _hill_climb_with_restarts(num_restarts=5, time_per_restart=15.0):
    """Multiple restarts of local search from different initial configurations."""
    start_time = time.time()
    global_best_points = None
    global_best_score = 0
    
    for r in range(num_restarts):
        if time.time() > DEADLINE - 5:
            break
        
        # Generate initial configuration
        if r == 0:
            init_points = _generate_grid_config()
        else:
            # Random initial configuration with some structure
            init_points = []
            # Mix of grid-like and random
            if r % 2 == 0:
                # Grid-like
                for x in range(-4, 4):
                    for y in range(-3, 3):
                        if len(init_points) < 30:
                            init_points.append((x, y))
            else:
                # More random but avoid duplicates
                attempts = 0
                while len(init_points) < 30 and attempts < 1000:
                    p = _random_point_in_bounds()
                    if p not in init_points:
                        init_points.append(p)
                    attempts += 1
                # If we didn't get 30, fill with grid
                while len(init_points) < 30:
                    for x in range(-30, 31):
                        for y in range(-30, 31):
                            if len(init_points) >= 30:
                                break
                            if (x, y) not in init_points:
                                init_points.append((x, y))
                        if len(init_points) >= 30:
                            break
            
            if len(init_points) != 30:
                continue
        
        # Verify we have exactly 30 distinct points
        if len(set(init_points)) != 30:
            continue
        
        # Run local search
        time_limit = min(time_per_restart, DEADLINE - 5 - (time.time() - start_time))
        if time_limit < 1:
            break
        
        best_pts, best_sc = _local_search(init_points, max_iters=3000, time_limit=time_limit)
        
        if best_sc > global_best_score:
            global_best_score = best_sc
            global_best_points = best_pts
    
    return global_best_points, global_best_score

def construct():
    """Main entry point: find 30 distinct integer points in [-30,30]^2
    maximizing the largest multiplicity of a single positive doubled area."""
    start_time = time.time()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Try hill climbing with restarts
    best_points, best_score = _hill_climb_with_restarts(num_restarts=8, time_per_restart=10.0)
    
    if best_points is None or len(best_points) != 30:
        # Fallback to grid
        best_points = _generate_grid_config()
        best_score = _fast_score(best_points)
    
    # Final verification
    if time.time() < DEADLINE - 2:
        # One more quick local search pass
        final_pts, final_sc = _local_search(best_points, max_iters=1000, time_limit=DEADLINE - 2)
        if final_sc > best_score:
            best_points = final_pts
            best_score = final_sc
    
    # Ensure exactly 30 distinct points in bounds
    result = []
    seen = set()
    for p in best_points:
        if -30 <= p[0] <= 30 and -30 <= p[1] <= 30 and p not in seen:
            result.append(p)
            seen.add(p)
    
    # If we don't have 30, fill in
    if len(result) < 30:
        for x in range(-30, 31):
            for y in range(-30, 31):
                if len(result) >= 30:
                    break
                if (x, y) not in seen:
                    result.append((x, y))
                    seen.add((x, y))
            if len(result) >= 30:
                break
    
    return result[:30]
