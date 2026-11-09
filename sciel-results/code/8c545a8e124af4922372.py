import numpy as np
import time
from math import gcd
from itertools import combinations

def _normalize_line(A, B, C):
    if A == 0 and B == 0:
        return None
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return None
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A; B = -B; C = -C
    return (A, B, C)

def _compute_top27(points):
    """
    Computes the top 27 lines by incidence count for the given points.
    Returns (score, lines_array).
    """
    n = len(points)
    if n < 2:
        return 0, np.zeros((27, 3), dtype=np.int64)
    
    # Dictionary to store line counts
    # Key: normalized (A, B, C)
    # Value: count of points on the line
    line_counts = {}
    
    # Pre-allocate lists for speed?
    # Loop over pairs to generate lines
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            key = _normalize_line(A, B, C)
            if key is not None:
                if key in line_counts:
                    line_counts[key] += 1
                else:
                    line_counts[key] = 1
                    
    # Now we need the ACTUAL number of points on each line, not just pairs.
    # A line with k points has C(k,2) pairs.
    # We can't just use pair count. We must count points.
    # However, if we just want the top 27, we can estimate.
    # But for exact score, we need exact count.
    
    # Optimization: Only check lines that have at least 2 pairs (so >= 3 points)?
    # No, a line with 2 points has count 2.
    # The top 27 will likely have high counts.
    # Let's just count for all unique lines found.
    
    # To speed this up, we can filter lines with low pair counts first?
    # If a line has 1 pair, it has 2 points.
    # If a line has 3 pairs, it has 3 points.
    # We want the top 27.
    
    # Let's identify candidate lines.
    # If we sort by pair count, the top ones are likely the top by point count.
    # But a line with 100 pairs (14 points) is better than 99 pairs (14 points).
    # Actually, pair count is a monotonic function of point count for a fixed set of points?
    # No, different lines have different point sets.
    # But for a single line, $k$ points $\iff \binom{k}{2}$ pairs.
    # So sorting by pair count is equivalent to sorting by point count.
    
    # So we can sort the keys by their pair count (value in line_counts).
    sorted_keys = sorted(line_counts.keys(), key=lambda k: -line_counts[k])
    
    # We only need to check the top ~50 or so to find the top 27 by point count?
    # Actually, the top 27 by pair count ARE the top 27 by point count.
    # Why? Because $\binom{k}{2}$ is strictly increasing for $k \ge 2$.
    # So the line with the most pairs has the most points.
    
    # So we just need to take the top 27 keys from sorted_keys.
    # But we need to return the (A, B, C) and the score (sum of point counts).
    # We don't have the point counts yet, only pair counts.
    # We must compute point counts for the top candidates.
    
    # How many candidates to check?
    # If the 27th line has pair count $P$, any line with pair count $< P$ cannot be in the top 27.
    # So we take the first 27 keys.
    # But what if there are ties?
    # It's safer to take a slightly larger set, e.g., top 50, compute their point counts, then pick the top 27.
    
    candidate_keys = sorted_keys[:50]
    
    final_lines = []
    final_scores = []
    
    for key in candidate_keys:
        A, B, C = key
        count = 0
        for pt in points:
            if A * pt[0] + B * pt[1] + C == 0:
                count += 1
        final_lines.append(key)
        final_scores.append(count)
        
    # Sort these 50 by actual point count
    indexed = list(zip(final_scores, final_lines))
    indexed.sort(key=lambda x: (-x[0], x[1]))
    
    # Take top 27
    selected = indexed[:27]
    
    if len(selected) < 27:
        # Pad with dummy lines if not enough (should not happen for 54 points)
        while len(selected) < 27:
            selected.append((0, (1, 0, 1000000 + len(selected))))
            
    score = sum(s for s, _ in selected)
    lines_arr = np.array([l for _, l in selected], dtype=np.int64)
    
    return score, lines_arr

def _generate_grid(a, b):
    xs = np.arange(1, a + 1)
    ys = np.arange(1, b + 1)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def _check_valid_points(points):
    if len(points) != 54:
        return False
    if points.shape != (54, 2):
        return False
    if not np.all(np.isfinite(points)):
        return False
    if not np.allclose(points, np.round(points)):
        return False
    pt_set = set(map(tuple, points))
    if len(pt_set) != 54:
        return False
    return True

def _perturb_point(points, idx, dx, dy):
    new_points = points.copy()
    new_points[idx] = points[idx] + np.array([dx, dy])
    return new_points

def _swap_points(points, i, j):
    new_points = points.copy()
    new_points[i], new_points[j] = new_points[j], new_points[i]
    return new_points

def _local_search(points, deadline, best_score, best_points, best_lines):
    n = len(points)
    # Moves: small shifts
    shifts = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1),
              (-2,0),(2,0),(0,-2),(0,2),(-2,-2),(2,2),(-2,2),(2,-2)]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(n):
            for dx, dy in shifts:
                if time.time() > deadline:
                    break
                new_points = _perturb_point(points, idx, dx, dy)
                if not _check_valid_points(new_points):
                    continue
                score, lines = _compute_top27(new_points)
                if score > best_score:
                    best_score = score
                    best_points = new_points
                    best_lines = lines
                    improved = True
                    points = new_points
                    break
            if time.time() > deadline:
                break
            # Also try swaps with a random other point?
            # Swaps are expensive to check all, so maybe just a few random swaps per iteration?
            # Let's stick to 1-opt for now for speed, SA will handle the rest.
    return best_score, best_points, best_lines

def _simulated_annealing(points, deadline, best_score, best_points, best_lines):
    n = len(points)
    current = points.copy()
    current_score, _ = _compute_top27(current)
    
    T = 10.0
    T_min = 0.1
    cooling = 0.995
    
    while time.time() < deadline and T > T_min:
        idx = np.random.randint(n)
        # Random move: shift or swap
        if np.random.random() < 0.8:
            dx = np.random.randint(-3, 4)
            dy = np.random.randint(-3, 4)
            if dx == 0 and dy == 0:
                continue
            new_points = _perturb_point(current, idx, dx, dy)
        else:
            j = np.random.randint(n)
            if i == j: continue
            new_points = _swap_points(current, idx, j)
            
        if not _check_valid_points(new_points):
            T *= cooling
            continue
        
        new_score, new_lines = _compute_top27(new_points)
        delta = new_score - current_score
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current = new_points
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
                best_lines = new_lines
        
        T *= cooling
    
    return best_score, best_points, best_lines

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 610  # Safety margin
    
    np.random.seed(42)
    
    best_score = -1
    best_points = None
    best_lines = None
    
    def update_best(pts, score, lines):
        nonlocal best_score, best_points, best_lines
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            best_lines = lines
    
    # Phase 1: Grids
    grid_configs = [
        (6, 9), (9, 6), (5, 11), (11, 5), (4, 14), (14, 4), (3, 18), (18, 3), (2, 27), (27, 2)
    ]
    
    for a, b in grid_configs:
        if time.time() > deadline - 60:
            break
        pts = _generate_grid(a, b)
        if len(pts) != 54:
            continue
        if not _check_valid_points(pts):
            continue
        score, lines = _compute_top27(pts)
        update_best(pts, score, lines)
        
        # Local search from this grid
        if time.time() < deadline - 30:
            best_score, best_points, best_lines = _local_search(
                pts, deadline - 30, best_score, best_points, best_lines
            )

    # Phase 2: Simulated Annealing from best found
    if best_points is not None and time.time() < deadline - 30:
        best_score, best_points, best_lines = _simulated_annealing(
            best_points, deadline - 10, best_score, best_points, best_lines
        )
        
    # Fallback
    if best_points is None:
        pts = _generate_grid(6, 9)
        score, lines = _compute_top27(pts)
        best_points = pts
        best_lines = lines
        best_score = score
        
    # Ensure integer types
    best_points = np.asarray(best_points, dtype=np.int64)
    best_lines = np.asarray(best_lines, dtype=np.int64)
    
    return best_points, best_lines
