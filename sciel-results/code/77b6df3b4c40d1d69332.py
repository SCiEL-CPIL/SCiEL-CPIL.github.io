# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd
from itertools import combinations, product

def _canonical_line(A, B, C):
    """Return canonical form of line (A,B,C): divide by gcd, first nonzero positive."""
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidences(points, lines):
    """Count point-line incidences: pairs (p, l) with A*x + B*y + C == 0."""
    count = 0
    for i in range(lines.shape[0]):
        A, B, C = lines[i]
        vals = A * points[:, 0] + B * points[:, 1] + C
        count += np.count_nonzero(vals == 0)
    return count

def _all_lines_through_pairs(points):
    """Generate all lines determined by pairs of points, with their incidence counts."""
    n = len(points)
    line_to_points = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            if A == 0 and B == 0:
                continue
            key = _canonical_line(A, B, C)
            if key not in line_to_points:
                line_to_points[key] = set()
            line_to_points[key].add(i)
            line_to_points[key].add(j)
    return line_to_points

def _top_lines(points, m):
    """Find the top m lines by incidence count."""
    line_to_points = _all_lines_through_pairs(points)
    sorted_lines = sorted(line_to_points.items(), key=lambda x: -len(x[1]))
    top = []
    seen = set()
    for key, pts in sorted_lines:
        if len(top) >= m:
            break
        if key in seen:
            continue
        seen.add(key)
        top.append((key, len(pts)))
    return top

def _score_configuration(points, m=27):
    """Score = total incidences of top m lines."""
    top = _top_lines(points, m)
    return sum(cnt for _, cnt in top)

def _generate_grid_candidates():
    """Generate structured grid candidates."""
    candidates = []
    
    # Standard rectangular grids
    dims = [(1, 54), (2, 27), (3, 18), (6, 9), (9, 6), (18, 3), (27, 2), (54, 1)]
    for w, h in dims:
        xs = np.arange(1, w + 1)
        ys = np.arange(1, h + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        candidates.append(points)
        
        # Sheared grids: y' = y + s*x
        for s in [-2, -1, 1, 2]:
            points_shear = points.copy()
            points_shear[:, 1] = points_shear[:, 1] + s * points_shear[:, 0]
            candidates.append(points_shear)
            
            # Another shear direction: x' = x + s*y (less common but valid)
            points_shear2 = points.copy()
            points_shear2[:, 0] = points_shear2[:, 0] + s * points_shear2[:, 1]
            candidates.append(points_shear2)

    # Non-rectangular grids: 7x8 minus 2 points
    for remove_idx in [0, 1, 2, 3, 5, 6, 7, 10, 11, 14, 15, 20, 21, 24, 25, 30, 31, 34, 35, 40, 41, 44, 45, 50, 51, 54, 55, 56, 57]:
        points_list = []
        for x in range(1, 8):
            for y in range(1, 9):
                points_list.append((x, y))
        if remove_idx < len(points_list):
            points_list.pop(remove_idx)
        # Remove one more to get 54 (7*8=56, need 54)
        # We need to remove 2 points. Let's remove a corner and an opposite corner or similar.
        # This loop is inefficient, let's just generate a few specific subsets.
        
    # Specific subsets of 7x8
    base_7x8 = [(x, y) for x in range(1, 8) for y in range(1, 9)]
    # Remove (1,1) and (7,8)
    p1 = [p for p in base_7x8 if p not in [(1,1), (7,8)]]
    candidates.append(np.array(p1, dtype=np.int64))
    # Remove (1,8) and (7,1)
    p2 = [p for p in base_7x8 if p not in [(1,8), (7,1)]]
    candidates.append(np.array(p2, dtype=np.int64))
    # Remove (1,1) and (1,2)
    p3 = [p for p in base_7x8 if p not in [(1,1), (1,2)]]
    candidates.append(np.array(p3, dtype=np.int64))
    
    # 5x10 plus 4 points
    base_5x10 = [(x, y) for x in range(1, 6) for y in range(1, 11)]
    extras = [(6, 1), (6, 2), (7, 1), (7, 2)]
    p4 = base_5x10 + extras
    candidates.append(np.array(p4, dtype=np.int64))
    
    # 4x13 plus 2 points (4*13=52)
    base_4x13 = [(x, y) for x in range(1, 5) for y in range(1, 14)]
    extras2 = [(5, 1), (5, 2)]
    p5 = base_4x13 + extras2
    candidates.append(np.array(p5, dtype=np.int64))
    
    return candidates

def _local_search_polish(points, m, rng, time_limit):
    """Local search: try replacing one point at a time to improve score."""
    points = np.array(points, dtype=np.int64)
    n = len(points)
    best_score = _score_configuration(points, m)
    best_points = points.copy()
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 20
    
    min_coord = int(points.min()) - 10
    max_coord = int(points.max()) + 10
    
    while no_improve_count < max_no_improve:
        if time.time() - start_time > time_limit:
            break
        
        idx = rng.randint(0, n - 1)
        old_point = tuple(points[idx])
        existing = set(map(tuple, points))
        
        best_new_score = best_score
        best_new_point = None
        
        # Try 10 random replacements
        for _ in range(10):
            new_x = rng.randint(min_coord, max_coord)
            new_y = rng.randint(min_coord, max_coord)
            new_point = (new_x, new_y)
            if new_point in existing:
                continue
            
            points[idx] = new_point
            score = _score_configuration(points, m)
            if score > best_new_score:
                best_new_score = score
                best_new_point = new_point
            points[idx] = old_point
        
        if best_new_point is not None:
            points[idx] = best_new_point
            best_score = best_new_score
            best_points = points.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
    return best_points, best_score

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization. Construct 54 points and 27 lines
    to maximize point-line incidences.
    """
    start_time = time.time()
    time_limit = 580  # Leave 20s safety margin
    
    rng = random.Random(12345)
    
    candidates = _generate_grid_candidates()
    
    best_points = None
    best_score = -1
    best_candidate_idx = -1
    
    # Evaluate all structured candidates
    for i, pts in enumerate(candidates):
        if time.time() - start_time > time_limit - 60:
            break
        score = _score_configuration(pts, 27)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            best_candidate_idx = i

    # Polish the best candidate
    if best_points is not None:
        polished_points, polished_score = _local_search_polish(
            best_points, 27, rng, time_limit - (time.time() - start_time) - 30
        )
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_points

    # If time remains, try polishing the second best candidate
    if time.time() - start_time < time_limit - 30:
        scores = []
        for i, pts in enumerate(candidates):
            if i == best_candidate_idx:
                continue
            scores.append((_score_configuration(pts, 27), i, pts))
        scores.sort(key=lambda x: -x[0])
        if scores:
            second_score, second_idx, second_pts = scores[0]
            if second_score > best_score * 0.95: # Only if close
                polished2, score2 = _local_search_polish(
                    second_pts, 27, rng, time_limit - (time.time() - start_time) - 10
                )
                if score2 > best_score:
                    best_score = score2
                    best_points = polished2

    # Generate top 27 lines for best configuration
    top = _top_lines(best_points, 27)
    
    # If we have fewer than 27 lines, add more
    if len(top) < 27:
        line_to_points = _all_lines_through_pairs(best_points)
        all_lines = sorted(line_to_points.items(), key=lambda x: -len(x[1]))
        existing_keys = {t[0] for t in top}
        for key, pts in all_lines:
            if len(top) >= 27:
                break
            if key not in existing_keys:
                top.append((key, len(pts)))
                existing_keys.add(key)
    
    # Take exactly 27 lines
    top = top[:27]
    lines = np.array([t[0] for t in top], dtype=np.int64)
    
    # Ensure we have exactly 27 lines (pad if needed)
    if lines.shape[0] < 27:
        extra = []
        for i in range(27 - lines.shape[0]):
            extra.append((1, 0, -(100 + i)))
        lines = np.vstack([lines, np.array(extra, dtype=np.int64)])
    
    # Ensure points are exactly 54
    if best_points.shape[0] != 54:
        if best_points.shape[0] > 54:
            best_points = best_points[:54]
        else:
            while best_points.shape[0] < 54:
                best_points = np.vstack([best_points, [[100 + best_points.shape[0], 0]]])

    return best_points, lines

# EVOLVE-BLOCK-END
