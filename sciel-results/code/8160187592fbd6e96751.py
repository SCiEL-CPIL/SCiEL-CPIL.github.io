# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd
from itertools import combinations

def _canonical_line(A, B, C):
    """Return canonical form of line (A,B,C): divide by gcd, first nonzero positive."""
    A, B, C = int(A), int(B), int(C)
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
        for j in range(i + 1, n):
            x1, y1 = points[i]
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
    for key, pts in sorted_lines:
        if len(top) >= m:
            break
        top.append((key, len(pts)))
    return top

def _score_configuration(points, m=27):
    """Score = total incidences of top m lines. Uses vectorized line generation."""
    n = len(points)
    if n < 2:
        return 0
    # Generate all pair lines vectorized
    i_idx, j_idx = np.triu_indices(n, k=1)
    x1 = points[i_idx, 0].astype(np.int64)
    y1 = points[i_idx, 1].astype(np.int64)
    x2 = points[j_idx, 0].astype(np.int64)
    y2 = points[j_idx, 1].astype(np.int64)
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    # Filter out degenerate lines
    mask = (A != 0) | (B != 0)
    A = A[mask]
    B = B[mask]
    C = C[mask]
    if len(A) == 0:
        return 0
    # Canonicalize: divide by gcd, make first nonzero positive
    abs_A = np.abs(A)
    abs_B = np.abs(B)
    abs_C = np.abs(C)
    g = np.gcd(np.gcd(abs_A, abs_B), abs_C)
    g[g == 0] = 1
    A = A // g
    B = B // g
    C = C // g
    # Make first nonzero entry positive
    neg = (A < 0) | ((A == 0) & (B < 0)) | ((A == 0) & (B == 0) & (C < 0))
    A[neg] = -A[neg]
    B[neg] = -B[neg]
    C[neg] = -C[neg]
    # Deduplicate: sort and unique
    line_tuples = np.stack([A, B, C], axis=1)
    line_tuples = np.sort(line_tuples, axis=0)  # sort columns then rows
    unique_lines = np.unique(line_tuples, axis=0)
    # Count incidences for each unique line
    # For each line, count how many points satisfy A*x + B*y + C == 0
    # Use vectorized approach: for each line, compute values for all points
    num_lines = len(unique_lines)
    if num_lines == 0:
        return 0
    # Process in chunks to avoid memory issues
    chunk_size = max(1, min(num_lines, 500))
    line_counts = np.zeros(num_lines, dtype=np.int64)
    for start in range(0, num_lines, chunk_size):
        end = min(start + chunk_size, num_lines)
        chunk = unique_lines[start:end]
        # (chunk_size, 1, 3) * (1, n, 3) -> (chunk_size, n)
        vals = chunk[:, 0, None] * points[None, :, 0] + chunk[:, 1, None] * points[None, :, 1] + chunk[:, 2, None]
        line_counts[start:end] = np.sum(vals == 0, axis=1)
    # Get top m lines
    if m >= num_lines:
        return int(np.sum(line_counts))
    top_indices = np.argpartition(line_counts, -m)[-m:]
    return int(np.sum(line_counts[top_indices]))

def _random_point_in_range(min_coord, max_coord, existing_points, rng):
    """Generate a random point not in existing_points within range."""
    while True:
        x = rng.randint(min_coord, max_coord)
        y = rng.randint(min_coord, max_coord)
        p = (x, y)
        if p not in existing_points:
            return p

def _local_search(points, m, max_iters, rng, time_limit):
    """Local search: try replacing one point at a time to improve score."""
    points = np.array(points, dtype=np.int64)
    n = len(points)
    best_score = _score_configuration(points, m)
    best_points = points.copy()
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 60
    
    # Determine coordinate range
    min_coord = int(points.min()) - 5
    max_coord = int(points.max()) + 5
    min_coord = max(min_coord, -30)
    max_coord = min(max_coord, 30)
    
    while no_improve_count < max_no_improve:
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random point to replace
        idx = rng.randint(0, n - 1)
        old_point = tuple(points[idx])
        existing = set(map(tuple, points))
        
        # Try several candidate replacements
        best_new_score = best_score
        best_new_point = None
        
        # Number of trials adapts to remaining time
        remaining = time_limit - (time.time() - start_time)
        n_trials = max(15, min(60, int(remaining / 1.5)))
        
        # Generate smart candidates: points on high-incidence lines
        smart_candidates = []
        try:
            line_to_points = _all_lines_through_pairs(points)
            sorted_lines = sorted(line_to_points.items(), key=lambda x: -len(x[1]))
            for key, pts_set in sorted_lines[:10]:
                A, B, C = key
                # Find nearby integer points on this line not already in set
                for x in range(max(min_coord, -20), min(max_coord, 20) + 1):
                    if A * x + C == 0:
                        # B*y = -A*x - C, so y = (-A*x - C) / B
                        if B != 0:
                            num = -A * x - C
                            if num % B == 0:
                                y = num // B
                                p = (x, y)
                                if p not in existing and min_coord <= x <= max_coord and min_coord <= y <= max_coord:
                                    smart_candidates.append(p)
                    elif B != 0:
                        num = -A * x - C
                        if num % B == 0:
                            y = num // B
                            p = (x, y)
                            if p not in existing and min_coord <= x <= max_coord and min_coord <= y <= max_coord:
                                smart_candidates.append(p)
        except Exception:
            pass
        
        # Mix smart and random candidates
        for t in range(n_trials):
            if smart_candidates and rng.random() < 0.5:
                new_point = smart_candidates.pop(rng.randint(0, len(smart_candidates) - 1))
            else:
                new_point = _random_point_in_range(min_coord, max_coord, existing, rng)
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
            # Update range
            min_coord = max(int(points.min()) - 5, -30)
            max_coord = min(int(points.max()) + 5, 30)
        else:
            no_improve_count += 1
    
    return best_points, best_score

def _initial_grid_3x18():
    """3x18 grid = 54 points."""
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    return points

def _initial_grid_6x9():
    """6x9 grid = 54 points."""
    xs = np.arange(1, 7)
    ys = np.arange(1, 10)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    return points

def _initial_grid_9x6():
    """9x6 grid = 54 points."""
    xs = np.arange(1, 10)
    ys = np.arange(1, 7)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    return points

def _initial_grid_7x8_minus2():
    """7x8 grid minus 2 points = 54 points."""
    points_list = []
    for x in range(1, 8):
        for y in range(1, 9):
            points_list.append((x, y))
    rng = random.Random(42)
    remove_idx = rng.sample(range(len(points_list)), 2)
    remove_set = set(remove_idx)
    points_list = [p for i, p in enumerate(points_list) if i not in remove_set]
    return np.array(points_list, dtype=np.int64)

def _initial_grid_7x8_minus2_v2():
    """7x8 grid minus 2 different points = 54 points."""
    points_list = []
    for x in range(1, 8):
        for y in range(1, 9):
            points_list.append((x, y))
    # Remove corner points to preserve more line structure
    remove = {(1,1), (7,8)}
    points_list = [p for p in points_list if p not in remove]
    return np.array(points_list, dtype=np.int64)

def _initial_grid_7x8_minus2_v3():
    """7x8 grid minus 2 points (center-ish)."""
    points_list = []
    for x in range(1, 8):
        for y in range(1, 9):
            points_list.append((x, y))
    remove = {(4,4), (4,5)}
    points_list = [p for p in points_list if p not in remove]
    return np.array(points_list, dtype=np.int64)

def _initial_mixed_5x10_plus4():
    """5x10 grid plus 4 extra points = 54 points."""
    points_list = []
    for x in range(1, 6):
        for y in range(1, 11):
            points_list.append((x, y))
    extra = [(6, 1), (6, 2), (7, 1), (7, 2)]
    points_list.extend(extra)
    return np.array(points_list, dtype=np.int64)

def _initial_staggered():
    """Staggered grid: offset alternating rows."""
    points_list = []
    for row in range(18):
        x_base = (row // 6) + 1
        for col in range(3):
            x = x_base + col
            y = row + 1
            points_list.append((x, y))
    return np.array(points_list, dtype=np.int64)

def _initial_parallelogram():
    """Parallelogram-shaped grid."""
    points_list = []
    for i in range(6):
        for j in range(9):
            x = i + j // 3
            y = j
            points_list.append((x + 1, y + 1))
    return np.array(points_list, dtype=np.int64)

def _two_opt_search(points, m, rng, time_limit):
    """Try swapping two points to improve score."""
    points = np.array(points, dtype=np.int64)
    n = len(points)
    best_score = _score_configuration(points, m)
    best_points = points.copy()
    
    start_time = time.time()
    no_improve = 0
    max_no_improve = 30
    
    min_coord = max(int(points.min()) - 5, -30)
    max_coord = min(int(points.max()) + 5, 30)
    
    while no_improve < max_no_improve:
        if time.time() - start_time > time_limit:
            break
        
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        
        old_i = tuple(points[i])
        old_j = tuple(points[j])
        existing = set(map(tuple, points))
        
        best_new_score = best_score
        best_new_i = None
        best_new_j = None
        
        remaining = time_limit - (time.time() - start_time)
        n_trials = max(8, min(30, int(remaining / 2)))
        
        for _ in range(n_trials):
            new_i = _random_point_in_range(min_coord, max_coord, existing - {old_j}, rng)
            new_j = _random_point_in_range(min_coord, max_coord, existing - {old_i}, rng)
            if new_i == new_j:
                continue
            points[i] = new_i
            points[j] = new_j
            score = _score_configuration(points, m)
            if score > best_new_score:
                best_new_score = score
                best_new_i = new_i
                best_new_j = new_j
            points[i] = old_i
            points[j] = old_j
        
        if best_new_i is not None:
            points[i] = best_new_i
            points[j] = best_new_j
            best_score = best_new_score
            best_points = points.copy()
            no_improve = 0
            min_coord = max(int(points.min()) - 5, -30)
            max_coord = min(int(points.max()) + 5, 30)
        else:
            no_improve += 1
    
    return best_points, best_score

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization. Construct 54 points and 27 lines
    to maximize point-line incidences.
    """
    start_time = time.time()
    time_limit = 580  # Leave 50s safety margin for the 630s kill
    
    rng = random.Random(12345)
    
    # Initial configurations
    configs = [
        _initial_grid_3x18(),
        _initial_grid_6x9(),
        _initial_grid_9x6(),
        _initial_grid_7x8_minus2(),
        _initial_grid_7x8_minus2_v2(),
        _initial_grid_7x8_minus2_v3(),
        _initial_mixed_5x10_plus4(),
        _initial_staggered(),
        _initial_parallelogram(),
    ]
    
    # Evaluate initial configs
    best_points = None
    best_score = -1
    for pts in configs:
        if time.time() - start_time > time_limit:
            break
        score = _score_configuration(pts, 27)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Sort configs by initial score, best first
    config_scores = []
    for pts in configs:
        config_scores.append((_score_configuration(pts, 27), pts))
    config_scores.sort(key=lambda x: -x[0])
    
    # Local search from each config (best first)
    for init_score, pts in config_scores:
        if time.time() - start_time > time_limit - 30:
            break
        remaining = time_limit - (time.time() - start_time)
        if remaining < 15:
            break
        
        # 1-opt local search
        searched_points, searched_score = _local_search(
            pts, 27, max_iters=1000, rng=rng,
            time_limit=remaining * 0.6
        )
        if searched_score > best_score:
            best_score = searched_score
            best_points = searched_points
        
        # 2-opt search if time remains
        if time.time() - start_time < time_limit - 20:
            remaining2 = time_limit - (time.time() - start_time)
            if remaining2 > 10:
                p2, s2 = _two_opt_search(
                    best_points, 27, rng=rng,
                    time_limit=remaining2 * 0.5
                )
                if s2 > best_score:
                    best_score = s2
                    best_points = p2
    
    # Final local search polish on the best configuration
    if time.time() - start_time < time_limit - 10:
        remaining = time_limit - (time.time() - start_time)
        p_final, s_final = _local_search(
            best_points, 27, max_iters=500, rng=rng,
            time_limit=remaining * 0.7
        )
        if s_final > best_score:
            best_score = s_final
            best_points = p_final
    
    # Generate top 27 lines for best configuration
    top = _top_lines(best_points, 27)
    
    # If we have fewer than 27 lines, add more
    if len(top) < 27:
        line_to_points = _all_lines_through_pairs(best_points)
        all_lines = sorted(line_to_points.items(), key=lambda x: -len(x[1]))
        existing_keys = set(t[0] for t in top)
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
    
    # Final time check
    if time.time() - start_time > time_limit:
        points = _initial_grid_7x8_minus2_v2()
        top_fb = _top_lines(points, 27)
        lines_fb = np.array([t[0] for t in top_fb], dtype=np.int64)
        if lines_fb.shape[0] < 27:
            extra = [(1, 0, -(100+i)) for i in range(27 - lines_fb.shape[0])]
            lines_fb = np.vstack([lines_fb, np.array(extra, dtype=np.int64)])
        return points, lines_fb
    
    return best_points, lines

# EVOLVE-BLOCK-END
