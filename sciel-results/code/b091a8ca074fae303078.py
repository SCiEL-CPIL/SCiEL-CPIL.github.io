# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd
from itertools import combinations

def _normalize_line(A, B, C):
    """Normalize line coefficients (A, B, C) to a canonical form."""
    if A == 0 and B == 0:
        return (0, 0, 0)
    
    # Find GCD of A, B, C
    g = gcd(abs(A), abs(B))
    g = gcd(g, abs(C))
    
    if g > 1:
        A //= g
        B //= g
        C //= g
        
    # Ensure first non-zero entry is positive
    if A < 0:
        A = -A
        B = -B
        C = -C
    elif A == 0 and B < 0:
        B = -B
        C = -C
        
    return (A, B, C)

def _line_from_points(p1, p2):
    """Get normalized line coefficients from two points."""
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = -(x2 - x1)
    C = (x2 - x1) * y1 - (y2 - y1) * x2
    return _normalize_line(A, B, C)

def _count_incidences_for_lines(points, lines):
    """
    Count incidences for a specific set of lines.
    points: (N, 2) array
    lines: (K, 3) array
    Returns: total incidences
    """
    total = 0
    for line in lines:
        A, B, C = line
        # Vectorized check: A*x + B*y + C == 0
        vals = A * points[:, 0] + B * points[:, 1] + C
        total += np.sum(vals == 0)
    return total

def _get_top_lines(points, k=27):
    """
    Find the k lines with the most incidences among all lines defined by pairs of points.
    Returns: list of k normalized line tuples (A, B, C)
    """
    n = len(points)
    pair_counts = {}
    
    # Iterate over all pairs
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            key = _line_from_points((x1, y1), (x2, y2))
            if key != (0, 0, 0):
                pair_counts[key] = pair_counts.get(key, 0) + 1
                
    # Convert pair counts to point counts
    line_point_counts = {}
    for key, pc in pair_counts.items():
        disc = 1 + 8 * pc
        sqrt_disc = int(np.sqrt(disc))
        if sqrt_disc * sqrt_disc == disc:
            k_pts = (1 + sqrt_disc) // 2
            if k_pts * (k_pts - 1) // 2 == pc:
                line_point_counts[key] = k_pts
            else:
                line_point_counts[key] = 2
        else:
            line_point_counts[key] = 2

    # Sort by count descending
    sorted_lines = sorted(line_point_counts.items(), key=lambda item: item[1], reverse=True)
    
    top_keys = [key for key, _ in sorted_lines[:k]]
    
    # If we don't have enough, pad with arbitrary lines through points
    # This shouldn't happen for N=54, but just in case
    if len(top_keys) < k:
        existing = set(top_keys)
        # Generate lines through single points not in existing
        # Just pick lines x=c, y=c, etc. that hit at least one point
        # Or just random lines? Random lines hit 0 points.
        # Lines through 1 point hit 1 point.
        # Let's try to find lines through 1 point.
        # Actually, if N=54, there are C(54,2) = 1431 pairs.
        # The top 27 lines will almost certainly be found.
        # If not, we pad with lines that might have 1 incidence.
        
        # Simple padding: lines x = points[i][0], y = points[i][1], x+y = points[i][0]+points[i][1]
        for p in points:
            x, y = p
            candidates = [
                _normalize_line(1, 0, -x),
                _normalize_line(0, 1, -y),
                _normalize_line(1, 1, -(x+y)),
                _normalize_line(1, -1, x-y)
            ]
            for c in candidates:
                if c not in existing and c != (0,0,0):
                    top_keys.append(c)
                    existing.add(c)
                    if len(top_keys) == k:
                        break
            if len(top_keys) == k:
                break
        
        # If still not enough, add random valid lines
        while len(top_keys) < k:
            A = random.randint(-10, 10)
            B = random.randint(-10, 10)
            C = random.randint(-100, 100)
            key = _normalize_line(A, B, C)
            if key not in existing and key != (0,0,0):
                top_keys.append(key)
                existing.add(key)

    return top_keys

def _constructive_search(max_time=580):
    """
    Strategy:
    1. Start with a small set of points (e.g., 10-20) arranged in a grid or cluster.
    2. Iteratively add points. For each candidate position, calculate the increase in total incidences
       with the current top-27 lines (recalculated after each addition).
    3. Once we have 54 points, perform local search: move one point to a better position.
    4. Also alternate with "line optimization": given points, find the best 27 lines.
       (This is already done in _get_top_lines, but we can try to tweak points to align with specific lines).
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    # Initial seed: A small grid, e.g., 4x4 = 16 points
    # Or maybe 3x3 = 9 points to start smaller and grow?
    # Let's start with a 5x4 grid = 20 points.
    xs = np.arange(1, 6)
    ys = np.arange(1, 5)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    best_score = -1
    best_points = points.copy()
    best_lines = np.array(_get_top_lines(points, 27), dtype=np.int64)
    
    # Calculate initial score
    current_score = _count_incidences_for_lines(points, best_lines)
    best_score = current_score
    
    # Phase 1: Grow points from 20 to 54
    while len(points) < 54:
        if time.time() > deadline - 60:
            break
            
        current_top_lines = _get_top_lines(points, 27)
        current_lines_arr = np.array(current_top_lines, dtype=np.int64)
        current_inc = _count_incidences_for_lines(points, current_lines_arr)
        
        # Determine candidate positions for the new point
        # Candidates:
        # 1. Extensions of the current grid (x = min/max +/- 1, y = min/max +/- 1)
        # 2. Intersections of existing lines (if integer)
        # 3. Random points in the bounding box
        
        min_x = np.min(points[:, 0])
        max_x = np.max(points[:, 0])
        min_y = np.min(points[:, 1])
        max_y = np.max(points[:, 1])
        
        candidates = set()
        
        # Grid extensions
        for x in [min_x - 1, min_x, min_x + 1, max_x - 1, max_x, max_x + 1]:
            for y in [min_y - 1, min_y, min_y + 1, max_y - 1, max_y, max_y + 1]:
                candidates.add((int(x), int(y)))
                
        # Intersections of top lines?
        # This is complex to implement robustly with integer constraints.
        # Let's stick to grid extensions and random.
        
        # Random points in a slightly expanded box
        for _ in range(50):
            x = random.randint(min_x - 3, max_x + 3)
            y = random.randint(min_y - 3, max_y + 3)
            candidates.add((x, y))
            
        # Filter out existing points
        existing_pts = set(map(tuple, points))
        candidates = [c for c in candidates if c not in existing_pts]
        
        if not candidates:
            # Expand box
            candidates = [(min_x - 2, min_y - 2), (max_x + 2, max_y + 2)]
            candidates = [c for c in candidates if c not in existing_pts]
            
        # Evaluate each candidate
        best_candidate = None
        best_candidate_score = current_inc
        best_new_lines = current_lines_arr
        
        # Note: Adding a point changes the set of top lines.
        # We must recalculate top lines for each candidate?
        # That's expensive: 54 candidates * O(N^2) line counting.
        # N=20 -> 190 pairs. Very fast.
        # N=50 -> 1225 pairs. Still fast.
        
        for c in candidates:
            new_points = np.vstack([points, np.array(c)])
            new_top_lines = _get_top_lines(new_points, 27)
            new_lines_arr = np.array(new_top_lines, dtype=np.int64)
            new_inc = _count_incidences_for_lines(new_points, new_lines_arr)
            
            if new_inc > best_candidate_score:
                best_candidate_score = new_inc
                best_candidate = c
                best_new_lines = new_lines_arr
                
        if best_candidate is not None:
            points = np.vstack([points, np.array(best_candidate)])
            current_score = best_candidate_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
                best_lines = best_new_lines.copy()
        else:
            # If no improvement found, pick a random candidate to avoid stagnation
            c = random.choice(candidates)
            points = np.vstack([points, np.array(c)])
            new_top_lines = _get_top_lines(points, 27)
            best_new_lines = np.array(new_top_lines, dtype=np.int64)
            current_score = _count_incidences_for_lines(points, best_new_lines)
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
                best_lines = best_new_lines.copy()

    # Phase 2: Local Search on the 54 points
    # Move one point at a time to a better position
    points = best_points.copy()
    lines = best_lines.copy()
    score = best_score
    
    no_improvement_count = 0
    max_no_improvement = 200
    
    while time.time() < deadline - 10:
        if no_improvement_count > max_no_improvement:
            break
            
        idx = random.randint(0, len(points) - 1)
        old_pt = points[idx].copy()
        
        # Generate candidates for the new position
        min_x = np.min(points[:, 0])
        max_x = np.max(points[:, 0])
        min_y = np.min(points[:, 1])
        max_y = np.max(points[:, 1])
        
        candidates = set()
        # Local neighborhood
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                candidates.add((int(old_pt[0] + dx), int(old_pt[1] + dy)))
        
        # Some global
        for _ in range(10):
            x = random.randint(min_x - 5, max_x + 5)
            y = random.randint(min_y - 5, max_y + 5)
            candidates.add((x, y))
            
        # Filter existing
        existing_pts = set(map(tuple, points))
        existing_pts.discard(tuple(old_pt))
        candidates = [c for c in candidates if c not in existing_pts]
        
        if not candidates:
            no_improvement_count += 1
            continue
            
        best_candidate = None
        best_candidate_score = score
        best_new_lines = lines
        
        for c in candidates:
            new_points = points.copy()
            new_points[idx] = c
            new_top_lines = _get_top_lines(new_points, 27)
            new_lines_arr = np.array(new_top_lines, dtype=np.int64)
            new_inc = _count_incidences_for_lines(new_points, new_lines_arr)
            
            if new_inc > best_candidate_score:
                best_candidate_score = new_inc
                best_candidate = c
                best_new_lines = new_lines_arr
                
        if best_candidate is not None:
            points[idx] = best_candidate
            lines = best_new_lines
            score = best_candidate_score
            no_improvement_count = 0
            if score > best_score:
                best_score = score
                best_points = points.copy()
                best_lines = lines.copy()
        else:
            no_improvement_count += 1
            
    return best_points, best_lines, best_score

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization.
    Uses a constructive heuristic followed by local search.
    """
    start_time = time.time()
    timeout = 590  # Safety margin
    
    best_points, best_lines, best_score = _constructive_search(max_time=timeout - 10)
    
    # Final validation and formatting
    points_out = best_points.astype(np.int64)
    lines_out = best_lines.astype(np.int64)
    
    # Ensure shapes
    if points_out.shape != (54, 2):
        # Fallback to grid if something went wrong
        xs = np.arange(1, 10)
        ys = np.arange(1, 7)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points_out = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines_out = np.array(_get_top_lines(points_out, 27), dtype=np.int64)
        
    if lines_out.shape != (27, 3):
        lines_out = np.array(_get_top_lines(points_out, 27), dtype=np.int64)
        
    # Check for duplicates in points
    pts_set = set([tuple(p) for p in points_out])
    if len(pts_set) < 54:
        # Fallback
        xs = np.arange(1, 10)
        ys = np.arange(1, 7)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points_out = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines_out = np.array(_get_top_lines(points_out, 27), dtype=np.int64)
        
    # Check for invalid lines
    for l in lines_out:
        if l[0] == 0 and l[1] == 0:
            # Fallback
            xs = np.arange(1, 10)
            ys = np.arange(1, 7)
            px, py = np.meshgrid(xs, ys, indexing="ij")
            points_out = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
            lines_out = np.array(_get_top_lines(points_out, 27), dtype=np.int64)
            break
            
    # Check for duplicate lines
    lines_set = set([tuple(l) for l in lines_out])
    if len(lines_set) < 27:
        # Fallback
        xs = np.arange(1, 10)
        ys = np.arange(1, 7)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points_out = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines_out = np.array(_get_top_lines(points_out, 27), dtype=np.int64)

    return points_out, lines_out

# EVOLVE-BLOCK-END
