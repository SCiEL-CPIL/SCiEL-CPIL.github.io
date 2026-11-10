import numpy as np
import time
import math
import random
from itertools import combinations

def _normalize_line(A, B, C):
    """Normalize a line equation A*x + B*y + C = 0 to a unique representative."""
    A, B, C = int(A), int(B), int(C)
    if A == 0 and B == 0:
        return (0, 0, 0)
    
    g = math.gcd(abs(A), abs(B))
    g = math.gcd(g, abs(C))
    if g == 0:
        return (0, 0, 0)
    
    A, B, C = A // g, B // g, C // g
    
    # Ensure the first non-zero coefficient is positive
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B == 0 and C < 0:
        A, B, C = -A, -B, -C
        
    return (A, B, C)

def _count_incidence(points, lines):
    """
    Count incidences between points (N,2) and lines (M,3).
    Returns an integer count.
    """
    if len(points) == 0 or len(lines) == 0:
        return 0
    px = points[:, 0]
    py = points[:, 1]
    A = lines[:, 0]
    B = lines[:, 1]
    C = lines[:, 2]
    # Vectorized computation: (M, 1) * (1, N) + (M, 1) * (1, N) + (M, 1)
    vals = A[:, None] * px[None, :] + B[:, None] * py[None, :] + C[:, None]
    return int(np.sum(vals == 0))

def _generate_initial_grid(k, m, shear=0, scale=1, ox=0, oy=0):
    """
    Generate a k x m grid with optional shear and scaling.
    Returns (N, 2) array.
    """
    xs = np.arange(k) * scale
    ys = np.arange(m) * scale
    px, py = np.meshgrid(xs, ys, indexing='ij')
    
    # Apply shear: x' = x + shear * y
    px = px + shear * py
    
    # Translate
    px = px + ox
    py = py + oy
    
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def _build_candidate_pool(points, max_candidates=5000):
    """
    Build a pool of candidate lines from:
    1. Lines through pairs of points.
    2. Horizontal, Vertical, and Diagonal lines with small coefficients.
    3. Lines with small integer coefficients passing through each point.
    """
    n = len(points)
    candidates = set()
    
    # 1. Lines through pairs
    for i in range(n):
        x1, y1 = int(points[i, 0]), int(points[i, 1])
        for j in range(i + 1, n):
            x2, y2 = int(points[j, 0]), int(points[j, 1])
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            key = _normalize_line(A, B, C)
            candidates.add(key)
            
    # 2. Structured lines
    xs = set(int(p[0]) for p in points)
    ys = set(int(p[1]) for p in points)
    
    for x in xs:
        candidates.add(_normalize_line(1, 0, -x)) # Vertical x = c
    for y in ys:
        candidates.add(_normalize_line(0, 1, -y)) # Horizontal y = c
        
    # Diagonals y = x + c and y = -x + c
    diffs = set(int(p[1]) - int(p[0]) for p in points)
    sums = set(int(p[1]) + int(p[0]) for p in points)
    for d in diffs:
        candidates.add(_normalize_line(1, -1, -d)) # y - x = d => x - y - d = 0
    for s in sums:
        candidates.add(_normalize_line(1, 1, -s))   # y + x = s => x + y - s = 0
        
    # 3. Lines with small coefficients through each point
    for A in range(-4, 5):
        for B in range(-4, 5):
            if A == 0 and B == 0:
                continue
            # Normalize (A, B) to avoid duplicates
            g_ab = math.gcd(abs(A), abs(B))
            if g_ab > 1:
                A, B = A // g_ab, B // g_ab
            if A < 0 or (A == 0 and B < 0):
                A, B = -A, -B
            for p in points:
                x, y = int(p[0]), int(p[1])
                C = -(A * x + B * y)
                candidates.add(_normalize_line(A, B, C))
                
    return list(candidates)

def _select_best_lines(points, candidate_lines, k=27):
    """
    Select the k lines from candidate_lines that have the highest incidence with points.
    candidate_lines: list of (A, B, C) tuples.
    """
    if not candidate_lines:
        return np.zeros((k, 3), dtype=np.int64)
        
    lines_arr = np.array(candidate_lines, dtype=np.int64)
    counts = np.zeros(len(lines_arr), dtype=np.int64)
    
    px = points[:, 0]
    py = points[:, 1]
    
    # Process in batches if lines_arr is very large to save memory, 
    # but for ~1000-2000 lines and 54 points, it's small.
    A = lines_arr[:, 0]
    B = lines_arr[:, 1]
    C = lines_arr[:, 2]
    
    vals = A[:, None] * px[None, :] + B[:, None] * py[None, :] + C[:, None]
    counts = np.sum(vals == 0, axis=1)
    
    # Get top k indices
    # np.argsort returns indices that would sort the array.
    # We want descending order of counts.
    top_k_indices = np.argsort(counts)[::-1][:k]
    
    selected_lines = lines_arr[top_k_indices]
    
    # Pad if necessary
    if len(selected_lines) < k:
        pad = np.zeros((k - len(selected_lines), 3), dtype=np.int64)
        # Use dummy lines that won't be valid or will be replaced
        pad[:, 1] = 1
        pad[:, 2] = -10000
        selected_lines = np.vstack([selected_lines, pad])
        
    return selected_lines

def _local_search(points, lines, time_limit, seed):
    """
    Perform local search on points and lines.
    Returns (points, lines, score).
    """
    random.seed(seed)
    points = points.copy()
    lines = lines.copy()
    current_score = _count_incidence(points, lines)
    
    start_time = time.time()
    no_improve = 0
    max_no_improve = 200
    pool = _build_candidate_pool(points)
    
    # Precompute incidence counts for pool
    def compute_pool_counts(pts, pool_list):
        if not pool_list:
            return np.zeros(0, dtype=np.int64)
        lines_arr = np.array(pool_list, dtype=np.int64)
        px = pts[:, 0]
        py = pts[:, 1]
        A = lines_arr[:, 0]
        B = lines_arr[:, 1]
        C = lines_arr[:, 2]
        vals = A[:, None] * px[None, :] + B[:, None] * py[None, :] + C[:, None]
        return np.sum(vals == 0, axis=1)
    
    pool_counts = compute_pool_counts(points, pool)
    
    while time.time() - start_time < time_limit:
        if no_improve > max_no_improve:
            break
            
        move_type = random.random()
        improved = False
        
        if move_type < 0.65:
            # Move a point - try to maximize incidences with existing lines
            i = random.randint(0, len(points) - 1)
            old_x, old_y = int(points[i, 0]), int(points[i, 1])
            
            # Generate candidate positions: nearby points + positions that lie on existing lines
            candidate_positions = []
            
            # Nearby positions
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx == 0 and dy == 0:
                        continue
                    candidate_positions.append((old_x + dx, old_y + dy))
            
            # Positions on existing lines (sample a few)
            for li in range(27):
                A, B, C = int(lines[li, 0]), int(lines[li, 1]), int(lines[li, 2])
                if A == 0 and B == 0:
                    continue
                # Find integer solutions near old position
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        nx, ny = old_x + dx, old_y + dy
                        if A * nx + B * ny + C == 0:
                            candidate_positions.append((nx, ny))
            
            # Try each candidate position
            best_new_score = current_score
            best_new_pos = None
            best_new_lines = None
            
            for new_x, new_y in candidate_positions:
                # Check uniqueness
                is_unique = True
                for j in range(len(points)):
                    if j == i: continue
                    if points[j, 0] == new_x and points[j, 1] == new_y:
                        is_unique = False
                        break
                if not is_unique:
                    continue
                
                # Apply move temporarily
                points[i, 0] = new_x
                points[i, 1] = new_y
                
                # Rebuild pool and select best lines
                pool = _build_candidate_pool(points)
                new_lines = _select_best_lines(points, pool, 27)
                new_score = _count_incidence(points, new_lines)
                
                if new_score > best_new_score:
                    best_new_score = new_score
                    best_new_pos = (new_x, new_y)
                    best_new_lines = new_lines
                    pool_counts = compute_pool_counts(points, pool)
                
                # Revert
                points[i, 0] = old_x
                points[i, 1] = old_y
            
            if best_new_pos is not None and best_new_score > current_score:
                points[i, 0] = best_new_pos[0]
                points[i, 1] = best_new_pos[1]
                lines = best_new_lines
                current_score = best_new_score
                improved = True
                no_improve = 0
            elif best_new_pos is not None and best_new_score == current_score and random.random() < 0.1:
                points[i, 0] = best_new_pos[0]
                points[i, 1] = best_new_pos[1]
                lines = best_new_lines
                no_improve += 1
                pool_counts = compute_pool_counts(points, pool)
            else:
                no_improve += 1
                
        else:
            # Swap a line - find the line with lowest marginal contribution
            # and replace with best available from pool
            current_line_set = set(map(tuple, lines.tolist()))
            
            # Compute counts for pool
            if not pool:
                no_improve += 1
                continue
                
            pool_counts = compute_pool_counts(points, pool)
            
            # Find best line not in current set
            mask = np.ones(len(pool), dtype=bool)
            for idx, line in enumerate(pool):
                if line in current_line_set:
                    mask[idx] = False
                    
            if not np.any(mask):
                no_improve += 1
                continue
                
            valid_indices = np.where(mask)[0]
            valid_counts = pool_counts[valid_indices]
            best_idx_in_valid = valid_indices[np.argmax(valid_counts)]
            best_line = pool[best_idx_in_valid]
            
            # Find the line to replace: the one with lowest incidence
            # among lines that have low unique contribution
            px = points[:, 0]
            py = points[:, 1]
            line_incidents = np.zeros(27, dtype=np.int64)
            for li in range(27):
                A, B, C = int(lines[li, 0]), int(lines[li, 1]), int(lines[li, 2])
                line_incidents[li] = np.sum((A * px + B * py + C) == 0)
            
            # Replace the line with lowest incidence
            replace_idx = np.argmin(line_incidents)
            old_line = lines[replace_idx]
            
            if best_line == tuple(old_line):
                no_improve += 1
                continue
                
            # Apply swap
            lines[replace_idx] = best_line
            new_score = _count_incidence(points, lines)
            
            if new_score > current_score:
                current_score = new_score
                improved = True
                no_improve = 0
            else:
                # Revert
                lines[replace_idx] = old_line
                no_improve += 1
                
        if not improved:
            no_improve += 1
        else:
            no_improve = 0
            
    return points, lines, current_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    best_points = None
    best_lines = None
    best_score = -1
    
    # List of initial configurations to try
    initial_configs = []
    
    # Standard grids
    for k in [1, 2, 3, 6, 9, 18, 27, 54]:
        if 54 % k == 0:
            m = 54 // k
            initial_configs.append(_generate_initial_grid(k, m))
            
    # Sheared grids
    for k in [3, 6, 9]:
        if 54 % k == 0:
            m = 54 // k
            for shear in [-2, -1, 1, 2]:
                initial_configs.append(_generate_initial_grid(k, m, shear=shear))
                
    # Random points in a box
    for _ in range(5):
        pts = np.random.randint(-10, 10, size=(54, 2))
        # Ensure uniqueness
        unique_pts = np.unique(pts, axis=0)
        if len(unique_pts) < 54:
            # Pad
            while len(unique_pts) < 54:
                new_pt = np.array([random.randint(-20, 20), random.randint(-20, 20)])
                if not np.any((unique_pts == new_pt).all(axis=1)):
                    unique_pts = np.vstack([unique_pts, new_pt[None, :]])
            pts = unique_pts[:54]
        initial_configs.append(pts.astype(np.int64))
        
    # Evaluate each initial configuration
    for i, pts in enumerate(initial_configs):
        if time.time() - start_time > deadline - 90:
            break
            
        # Build pool
        pool = _build_candidate_pool(pts)
        
        # Select best lines
        lines = _select_best_lines(pts, pool, 27)
        
        # Calculate score
        score = _count_incidence(pts, lines)
        
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            best_lines = lines.copy()
            
    # Local search from the best configuration found
    if best_points is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 60:
            # Run local search with multiple seeds to explore neighborhood
            seeds = [42, 123, 999, 7, 12345, 555]
            for seed in seeds:
                if time.time() - start_time > deadline - 20:
                    break
                alloc_time = min(remaining_time - 20, 150)
                if alloc_time < 10:
                    break
                p, l, s = _local_search(best_points, best_lines, alloc_time, seed)
                if s > best_score:
                    best_score = s
                    best_points = p.copy()
                    best_lines = l.copy()
                
                # Re-evaluate remaining time
                remaining_time = deadline - time.time()
                if remaining_time < 40:
                    break

    # Fallback
    if best_points is None:
        best_points = _generate_initial_grid(6, 9)
        pool = _build_candidate_pool(best_points)
        best_lines = _select_best_lines(best_points, pool, 27)
        best_score = _count_incidence(best_points, best_lines)
        
    # Final validation and formatting
    final_points = np.array(best_points, dtype=np.int64)
    final_lines = np.array(best_lines, dtype=np.int64)
    
    # Ensure shapes
    if final_points.shape != (54, 2):
        final_points = _generate_initial_grid(6, 9)
        pool = _build_candidate_pool(final_points)
        final_lines = _select_best_lines(final_points, pool, 27)
        
    if final_lines.shape != (27, 3):
        final_lines = final_lines[:27]
        if len(final_lines) < 27:
            pad = np.zeros((27 - len(final_lines), 3), dtype=np.int64)
            pad[:, 1] = 1
            pad[:, 2] = -10000
            final_lines = np.vstack([final_lines, pad])
            
    # Check for duplicate points
    unique_pts = np.unique(final_points, axis=0)
    if len(unique_pts) < 54:
        # This shouldn't happen if we were careful, but fix it just in case
        added = 0
        while len(unique_pts) < 54:
            pt = np.array([100 + added, 100 + added], dtype=np.int64)
            if not np.any((unique_pts == pt).all(axis=1)):
                unique_pts = np.vstack([unique_pts, pt[None, :]])
            added += 1
        final_points = unique_pts[:54]
        pool = _build_candidate_pool(final_points)
        final_lines = _select_best_lines(final_points, pool, 27)
        
    # Check for invalid lines (A=B=0)
    for i in range(27):
        if final_lines[i, 0] == 0 and final_lines[i, 1] == 0:
            final_lines[i, 1] = 1
            final_lines[i, 2] = -10000 - i

    return final_points.astype(np.int64), final_lines.astype(np.int64)
