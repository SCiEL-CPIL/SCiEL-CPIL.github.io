# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
from itertools import combinations

def _normalize_line(A, B, C):
    A, B, C = int(A), int(B), int(C)
    g = math.gcd(abs(A), abs(B))
    g = math.gcd(g, abs(C))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidence(points, lines):
    """Count incidences between points (N,2) and lines (M,3)."""
    if len(points) == 0 or len(lines) == 0:
        return 0
    px = points[:, 0]
    py = points[:, 1]
    A = lines[:, 0]
    B = lines[:, 1]
    C = lines[:, 2]
    vals = A[:, None] * px[None, :] + B[:, None] * py[None, :] + C[:, None]
    return int(np.sum(vals == 0))

def _build_line_pool(points, max_pool=500):
    """
    Build a pool of candidate lines from all pairs of points.
    Returns a list of (A, B, C) normalized lines, sorted by initial incidence count.
    """
    n = len(points)
    if n < 2:
        return [(0, 1, -1000)]
    
    xs = points[:, 0]
    ys = points[:, 1]
    
    seen = set()
    line_counts = {}
    
    for i in range(n):
        x1, y1 = int(xs[i]), int(ys[i])
        for j in range(i + 1, n):
            x2, y2 = int(xs[j]), int(ys[j])
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            key = _normalize_line(A, B, C)
            if key in seen:
                continue
            seen.add(key)
            vals = key[0] * xs + key[1] * ys + key[2]
            cnt = int(np.sum(vals == 0))
            line_counts[key] = cnt
    
    # Add some generic lines (horizontal/vertical) to ensure coverage
    for c in range(-20, 20):
        for key in [_normalize_line(0, 1, -c), _normalize_line(1, 0, -c)]:
            if key not in seen:
                seen.add(key)
                vals = key[0] * xs + key[1] * ys + key[2]
                cnt = int(np.sum(vals == 0))
                line_counts[key] = cnt
    
    # Sort by count, take top max_pool
    sorted_lines = sorted(line_counts.items(), key=lambda x: x[1], reverse=True)
    pool = [item[0] for item in sorted_lines[:max_pool]]
    
    # Ensure at least k lines
    while len(pool) < 27:
        pool.append((0, 1, -1000 - len(pool)))
    
    return pool

def _select_best_lines(points, line_pool, k=27):
    """
    From a fixed pool of candidate lines, select the k with highest incidence.
    This is much faster than recomputing from scratch.
    """
    if len(line_pool) == 0:
        return np.zeros((k, 3), dtype=np.int64)
    
    xs = points[:, 0]
    ys = points[:, 1]
    
    # Vectorized: compute incidence for all pool lines at once
    pool_arr = np.array(line_pool, dtype=np.int64)
    A = pool_arr[:, 0, None]
    B = pool_arr[:, 1, None]
    C = pool_arr[:, 2, None]
    vals = A * xs[None, :] + B * ys[None, :] + C
    counts = np.sum(vals == 0, axis=1)
    
    # Get top k
    if len(counts) >= k:
        top_idx = np.argsort(counts)[::-1][:k]
    else:
        top_idx = np.argsort(counts)[::-1]
    
    selected = pool_arr[top_idx]
    
    # Pad if necessary
    if len(selected) < k:
        pad = np.zeros((k - len(selected), 3), dtype=np.int64)
        pad[:, 1] = 1
        pad[:, 2] = -1000 - len(selected)
        selected = np.vstack([selected, pad])
    
    return selected[:k].astype(np.int64)

def _evaluate_configuration(points, line_pool=None):
    """Evaluate a point configuration and return (points, lines, score)."""
    if line_pool is None:
        line_pool = _build_line_pool(points, 500)
        lines = _select_best_lines(points, line_pool, 27)
    else:
        lines = _select_best_lines(points, line_pool, 27)
    score = _count_incidence(points, lines)
    return points, lines, score

def _generate_grid_points(k, m, dx=1, dy=1, ox=0, oy=0):
    """Generate a k x m grid of points."""
    xs = np.arange(k) * dx + ox
    ys = np.arange(m) * dy + oy
    px, py = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def _generate_punctured_grid(k, m, points_to_remove, dx=1, dy=1):
    """Generate a grid with specific points removed."""
    full_grid = _generate_grid_points(k, m, dx, dy)
    mask = np.ones(len(full_grid), dtype=bool)
    for (px, py) in points_to_remove:
        # Find and mask
        match = (full_grid[:, 0] == px) & (full_grid[:, 1] == py)
        if np.sum(match) > 0:
            mask[np.where(match)[0][0]] = False
    return full_grid[mask]

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 620  # 10s safety margin
    
    best_points = None
    best_lines = None
    best_score = -1
    
    # Phase 1: Search for the best static grid configuration
    # 54 = 1*54, 2*27, 3*18, 6*9
    # Also consider near-grids like 5*11 (55 pts), 7*8 (56 pts)
    
    candidate_configs = []
    
    # Exact factorizations
    for k in [1, 2, 3, 6, 9, 18, 27, 54]:
        if 54 % k == 0:
            m = 54 // k
            # Try different spacings
            for dx in [1, 2]:
                for dy in [1, 2]:
                    pts = _generate_grid_points(k, m, dx, dy)
                    candidate_configs.append(pts)
    
    # Near-grids: 5*11 = 55 (remove 1)
    for dx in [1]:
        for dy in [1]:
            pts_full = _generate_grid_points(5, 11, dx, dy)
            # Remove corner, center, etc.
            # Just remove (0,0) as a representative, search will refine if needed
            # Actually, removing different points might change line structure slightly,
            # but (0,0) is a good heuristic start.
            pts = pts_full[1:] # Remove first point
            candidate_configs.append(pts)
            
    # Near-grids: 7*8 = 56 (remove 2)
    for dx in [1]:
        for dy in [1]:
            pts_full = _generate_grid_points(7, 8, dx, dy)
            # Remove 2 corners
            pts = pts_full[2:] # Remove first 2
            candidate_configs.append(pts)
            
    # Also try 7*7=49 + 5 random? No, structure is better.
    # Try 3*18, 2*27 etc already covered.
    
    # Evaluate all candidates
    for pts in candidate_configs:
        if len(pts) != 54:
            continue
        # Ensure unique points (grid guarantees this, but check just in case)
        if len(set(map(tuple, pts))) != 54:
            continue
            
        p, l, s = _evaluate_configuration(pts)
        if s > best_score:
            best_score = s
            best_points = p.copy()
            best_lines = l.copy()
            
    # Phase 2: Local Search / Hill Climbing
    # Start from the best static configuration
    if best_points is None:
        # Fallback
        best_points = _generate_grid_points(6, 9)
        best_lines = _get_best_lines(best_points, 27)
        best_score = _count_incidence(best_points, best_lines)
        
    # Build a line pool from the best configuration for efficient local search
    line_pool = _build_line_pool(best_points, 800)
    
    points = best_points.copy()
    lines = best_lines.copy()
    current_score = best_score
    
    # Refinement: Move points slightly to increase incidences
    # This is computationally expensive, so we limit iterations
    max_iters = 2000
    iter_count = 0
    no_improve_count = 0
    
    while time.time() < deadline - 20 and iter_count < max_iters:
        iter_count += 1
        
        # Pick a random point
        i = random.randint(0, 53)
        
        # Try a small move
        found_improvement = False
        for _ in range(5):
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            if dx == 0 and dy == 0:
                continue
            
            new_x = int(points[i, 0]) + dx
            new_y = int(points[i, 1]) + dy
            
            # Check uniqueness
            if any(int(points[j, 0]) == new_x and int(points[j, 1]) == new_y for j in range(54) if j != i):
                continue
            
            new_points = points.copy()
            new_points[i, 0] = new_x
            new_points[i, 1] = new_y
            
            # Use fixed line pool for fast evaluation
            new_lines = _select_best_lines(new_points, line_pool, 27)
            new_score = _count_incidence(new_points, new_lines)
            
            if new_score > current_score:
                points = new_points
                lines = new_lines
                current_score = new_score
                found_improvement = True
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    best_lines = lines.copy()
                    no_improve_count = 0
                break
            elif new_score == current_score and random.random() < 0.2:
                # Accept neutral moves with low prob to escape local minima
                points = new_points
                lines = new_lines
                # current_score unchanged
        
        if not found_improvement:
            no_improve_count += 1
            if no_improve_count > 100:
                # Try a larger jump
                i = random.randint(0, 53)
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
                new_x = int(points[i, 0]) + dx
                new_y = int(points[i, 1]) + dy
                if not any(int(points[j, 0]) == new_x and int(points[j, 1]) == new_y for j in range(54) if j != i):
                    new_points = points.copy()
                    new_points[i, 0] = new_x
                    new_points[i, 1] = new_y
                    new_lines = _select_best_lines(new_points, line_pool, 27)
                    new_score = _count_incidence(new_points, new_lines)
                    if new_score >= current_score:
                        points = new_points
                        lines = new_lines
                        current_score = new_score
                no_improve_count = 0
                
        # Periodically rebuild the line pool to discover new lines
        if iter_count % 200 == 0:
            line_pool = _build_line_pool(points, 800)
            # Also try a full re-optimization from the current state
            lines = _select_best_lines(points, line_pool, 27)
            score = _count_incidence(points, lines)
            if score > current_score:
                current_score = score
                lines = lines
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
                best_lines = lines.copy()
    
    # Return the best result found
    # Use the global best, not necessarily the end-of-loop state if it degraded
    final_points = best_points
    final_lines = best_lines
    
    # Final validation and formatting
    final_points = np.array(final_points, dtype=np.int64)
    final_lines = np.array(final_lines, dtype=np.int64)
    
    # Ensure shapes
    if final_points.shape != (54, 2):
        # This should not happen, but fallback
        final_points = _generate_grid_points(6, 9)
        final_lines = _get_best_lines(final_points, 27)
        
    if final_lines.shape != (27, 3):
        final_lines = final_lines[:27]
        if len(final_lines) < 27:
            pad = np.zeros((27 - len(final_lines), 3), dtype=np.int64)
            pad[:, 1] = 1
            pad[:, 2] = -1000
            final_lines = np.vstack([final_lines, pad])
            
    # Double check uniqueness of points
    unique_pts = np.unique(final_points, axis=0)
    if len(unique_pts) < 54:
        # Add missing points
        added = 0
        offset = 100
        while len(unique_pts) < 54:
            pt = np.array([offset + added, offset * 2 + added], dtype=np.int64)
            if pt not in unique_pts:
                unique_pts = np.vstack([unique_pts, pt[None, :]])
            added += 1
        final_points = unique_pts[:54]
        # Recalculate lines for final points
        final_lines = _get_best_lines(final_points, 27)
    
    # Final polish: try to improve with a fresh line pool
    if time.time() < deadline - 30:
        final_pool = _build_line_pool(final_points, 800)
        final_lines = _select_best_lines(final_points, final_pool, 27)
        final_score = _count_incidence(final_points, final_lines)
        if final_score > best_score:
            best_score = final_score
    
    return final_points.astype(np.int64), final_lines.astype(np.int64)

# EVOLVE-BLOCK-END
