import itertools
import time
import random
import numpy as np
from collections import Counter, defaultdict

def _doubled_areas(points):
    """
    Compute the frequency of each positive doubled area using NumPy.
    points: list of (x, y)
    Returns: dict {area: count}
    """
    n = len(points)
    if n < 3:
        return {}
    pts = np.array(points, dtype=np.int64)
    # Use vectorized computation for all triples
    # For 30 points, C(30,3) = 4060 triples
    # We'll compute in chunks to avoid memory issues
    i_idx, j_idx, k_idx = np.meshgrid(
        np.arange(n), np.arange(n), np.arange(n), indexing='ij')
    # Only keep i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    x1 = pts[i_flat, 0]
    y1 = pts[i_flat, 1]
    x2 = pts[j_flat, 0]
    y2 = pts[j_flat, 1]
    x3 = pts[k_flat, 0]
    y3 = pts[k_flat, 1]
    
    # Determinant: |x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)|
    det = np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    
    # Filter out zero areas
    det = det[det > 0]
    
    if len(det) == 0:
        return {}
    
    # Count frequencies
    areas_list = det.tolist()
    return dict(Counter(areas_list))

def _area_counter(points):
    """Alias for _doubled_areas"""
    return _doubled_areas(points)

def _max_multiplicity(points):
    """Get the maximum multiplicity of any single positive doubled area."""
    areas = _doubled_areas(points)
    if not areas:
        return 0
    return max(areas.values())

def _valid(points):
    """Check if points are 30 distinct integer coordinates in [-30,30]^2."""
    if len(points) != 30:
        return False
    if len(set(points)) != 30:
        return False
    for x, y in points:
        if not (-30 <= x <= 30 and -30 <= y <= 30):
            return False
        if not isinstance(x, int) or not isinstance(y, int):
            return False
    return True

def _score(points):
    """Score function: max multiplicity of a positive doubled area."""
    if not _valid(points):
        return 0
    return _max_multiplicity(points)

def _fast_score(points):
    """Fast scoring using NumPy."""
    return _max_multiplicity(points)

def _structured_candidates():
    """Generate structured candidate configurations."""
    candidates = []
    
    # 1. Seed: 6x5 grid
    seed_grid = [(x, y) for x in range(6) for y in range(5)]
    candidates.append(seed_grid)
    
    # 2. 7x5 grid (35 pts) remove 5 random points
    grid_7x5 = [(x, y) for x in range(7) for y in range(5)]
    for _ in range(10):
        pts = grid_7x5[:]
        random.shuffle(pts)
        pts = pts[:30]
        candidates.append(pts)
        
    # 3. 5x7 grid (35 pts) remove 5
    grid_5x7 = [(x, y) for x in range(5) for y in range(7)]
    for _ in range(10):
        pts = grid_5x7[:]
        random.shuffle(pts)
        pts = pts[:30]
        candidates.append(pts)
        
    # 4. 4x8 grid (32 pts) remove 2
    grid_4x8 = [(x, y) for x in range(4) for y in range(8)]
    for _ in range(10):
        pts = grid_4x8[:]
        random.shuffle(pts)
        pts = pts[:30]
        candidates.append(pts)
    
    # 5. 5x6 grid (30 pts) - exact fit
    grid_5x6 = [(x, y) for x in range(5) for y in range(6)]
    candidates.append(grid_5x6)
    
    # 6. 6x5 grid shifted/translated variants
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            pts = [(x + dx, y + dy) for x in range(6) for y in range(5)]
            if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
                candidates.append(pts)
    
    # 7. Dense clusters: try different cluster shapes
    # 10x3 grid
    grid_10x3 = [(x, y) for x in range(10) for y in range(3)]
    for _ in range(5):
        pts = grid_10x3[:]
        random.shuffle(pts)
        pts = pts[:30]
        candidates.append(pts)
    
    # 3x10 grid
    grid_3x10 = [(x, y) for x in range(3) for y in range(10)]
    for _ in range(5):
        pts = grid_3x10[:]
        random.shuffle(pts)
        pts = pts[:30]
        candidates.append(pts)
    
    # 8. Diagonal patterns
    for slope in [1, 2, 3]:
        pts = []
        for i in range(10):
            for j in range(3):
                x = i
                y = i * slope + j
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
        if len(pts) >= 30:
            random.shuffle(pts)
            candidates.append(pts[:30])
    
    return candidates

def _local_search(points, start_time, time_limit, target_area=None):
    """
    Local search with incremental scoring and target-area focus.
    """
    current_points = points[:]
    current_score = _max_multiplicity(current_points)
    current_areas = _doubled_areas(current_points)
    
    # If we have a target area, track its count
    if target_area and target_area in current_areas:
        target_count = current_areas[target_area]
    else:
        target_count = 0
    
    improved = True
    no_improve_count = 0
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        no_improve_count += 1
        
        # Try moving each point to nearby locations
        for i in range(30):
            if (time.time() - start_time) > time_limit:
                break
            x, y = current_points[i]
            
            # Try various neighborhood sizes
            for radius in [1, 2, 3, 5]:
                if (time.time() - start_time) > time_limit:
                    break
                    
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                            continue
                        
                        # Check if (nx, ny) is already in points
                        if (nx, ny) in current_points:
                            continue
                        
                        new_points = current_points[:]
                        new_points[i] = (nx, ny)
                        
                        if not _valid(new_points):
                            continue
                        
                        new_score = _max_multiplicity(new_points)
                        
                        if new_score > current_score:
                            current_score = new_score
                            current_points = new_points
                            current_areas = _doubled_areas(current_points)
                            improved = True
                            no_improve_count = 0
                            
                            # Update target area tracking
                            if target_area:
                                if target_area in current_areas:
                                    target_count = current_areas[target_area]
                                else:
                                    target_count = 0
                            
                            # Break out of loops to restart
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        # If no improvement, try a larger jump or different strategy
        if not improved and no_improve_count > 5:
            # Try a random perturbation
            if (time.time() - start_time) > time_limit:
                break
            
            i = random.randint(0, 29)
            x, y = current_points[i]
            # Try a random position nearby
            for _ in range(20):
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
                nx, ny = x + dx, y + dy
                if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                    continue
                if (nx, ny) in current_points:
                    continue
                
                new_points = current_points[:]
                new_points[i] = (nx, ny)
                
                if not _valid(new_points):
                    continue
                
                new_score = _max_multiplicity(new_points)
                
                if new_score > current_score:
                    current_score = new_score
                    current_points = new_points
                    current_areas = _doubled_areas(current_points)
                    improved = True
                    no_improve_count = 0
                    break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > 10:
                    break
    
    return current_points, current_score

def construct():
    # Deadline guard
    start_time = time.time()
    time_limit = 70  # Leave 20s margin for 90s limit
    
    # Generate structured candidates
    candidates = _structured_candidates()
    
    # Evaluate candidates
    best_score = -1
    best_points = None
    best_areas = {}
    
    for pts in candidates:
        if not _valid(pts):
            continue
        score = _max_multiplicity(pts)
        if score > best_score:
            best_score = score
            best_points = pts[:]
            best_areas = _doubled_areas(pts)
    
    if best_points is None:
        # Fallback: simple grid
        best_points = [(x, y) for x in range(6) for y in range(5)]
        best_score = _max_multiplicity(best_points)
        best_areas = _doubled_areas(best_points)
    
    # Identify the most frequent area for target-driven search
    if best_areas:
        target_area = max(best_areas, key=best_areas.get)
    else:
        target_area = None
    
    # Run local search with target area focus
    current_points, current_score = _local_search(
        best_points, start_time, time_limit, target_area)
    
    # If we have time, try a second pass with a different target
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 10:
        # Try optimizing for a different area
        areas = _doubled_areas(current_points)
        if areas:
            # Find the second most frequent area
            sorted_areas = sorted(areas.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_areas) > 1:
                alt_target = sorted_areas[1][0]
                # Run another local search with this target
                pts2, score2 = _local_search(
                    current_points, start_time, time_limit, alt_target)
                if score2 > current_score:
                    current_points = pts2
                    current_score = score2
    
    # Final polish: try small moves around the best solution
    if (time.time() - start_time) < time_limit - 5:
        for i in range(30):
            if (time.time() - start_time) > time_limit - 5:
                break
            x, y = current_points[i]
            for dx in [-1, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                        continue
                    if (nx, ny) in current_points:
                        continue
                    
                    new_points = current_points[:]
                    new_points[i] = (nx, ny)
                    
                    if not _valid(new_points):
                        continue
                    
                    new_score = _max_multiplicity(new_points)
                    if new_score > current_score:
                        current_score = new_score
                        current_points = new_points
    
    return current_points
