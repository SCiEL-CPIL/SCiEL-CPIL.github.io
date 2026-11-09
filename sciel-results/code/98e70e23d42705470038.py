import numpy as np
import time
import random
import math
from math import gcd

def _normalize_line(A, B, C):
    """Normalize line coefficients so we can check uniqueness."""
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _is_valid(points, lines):
    """Check validity of the (points, lines) configuration."""
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    pt_set = set()
    for i in range(54):
        x, y = points[i]
        if not np.isfinite(x) or not np.isfinite(y):
            return False
        if abs(x - round(x)) > 1e-6 or abs(y - round(y)) > 1e-6:
            return False
        xi, yi = int(round(x)), int(round(y))
        if (xi, yi) in pt_set:
            return False
        pt_set.add((xi, yi))
    line_set = set()
    for i in range(27):
        A, B, C = lines[i]
        if not np.isfinite(A) or not np.isfinite(B) or not np.isfinite(C):
            return False
        if abs(A - round(A)) > 1e-6 or abs(B - round(B)) > 1e-6 or abs(C - round(C)) > 1e-6:
            return False
        Ai, Bi, Ci = int(round(A)), int(round(B)), int(round(C))
        if Ai == 0 and Bi == 0:
            return False
        key = _normalize_line(Ai, Bi, Ci)
        if key in line_set:
            return False
        line_set.add(key)
    return True

def _count_incidentences(points, lines):
    """Count incidences exactly."""
    count = 0
    for i in range(54):
        x, y = points[i]
        for j in range(27):
            A, B, C = lines[j]
            if A * x + B * y + C == 0:
                count += 1
    return count

def _generate_line_from_two_points(p1, p2):
    """Generate normalized line through two points."""
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    if A == 0 and B == 0:
        return None
    return _normalize_line(A, B, C)

def _score_and_best_lines(points):
    """
    For a given set of 54 points, compute the optimal 27 lines.
    Returns (score, best_lines_array) where score is the total incidences.
    """
    n = 54
    lines_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            key = _generate_line_from_two_points(tuple(points[i]), tuple(points[j]))
            if key is not None:
                lines_set.add(key)
    
    # Convert to list
    lines_list = list(lines_set)
    
    if len(lines_list) < 27:
        # Should not happen with 54 distinct points, but handle gracefully
        # This would be invalid, so we return 0 score
        return 0, np.zeros((27, 3), dtype=np.int64)
    
    # Convert to numpy arrays for vectorized computation
    lines_arr = np.array(lines_list, dtype=np.int64)
    pts_arr = np.array(points, dtype=np.int64)
    
    # Vectorized incidence count
    # For each line, count how many points lie on it
    # A*x + B*y + C == 0
    x = pts_arr[:, 0:1]  # (54, 1)
    y = pts_arr[:, 1:2]  # (54, 1)
    A = lines_arr[:, 0:1].T  # (1, L)
    B = lines_arr[:, 1:1].T  # (1, L)
    C = lines_arr[:, 2:3].T  # (1, L)
    
    # (54, L) matrix: A_j * x_i + B_j * y_i + C_j
    vals = A * x + B * y + C
    incidences = np.sum(vals == 0, axis=0)  # (L,)
    
    # Find top 27 lines
    if len(incidences) > 27:
        # Get indices of top 27
        top_idx = np.argpartition(incidences, -27)[-27:]
        # Sort by incidence count descending for consistency
        top_idx = top_idx[np.argsort(incidences[top_idx])[::-1]]
        top_lines = lines_arr[top_idx]
        score = int(np.sum(incidences[top_idx]))
    else:
        top_lines = lines_arr
        score = int(np.sum(incidences))
    
    return score, top_lines.astype(np.int64)

def _local_search(points, max_time, deadline):
    """
    Perform local search from a starting configuration.
    Returns (best_points, best_lines, best_score).
    """
    current_points = points.copy()
    current_score, current_lines = _score_and_best_lines(current_points)
    
    best_points = current_points.copy()
    best_lines = current_lines.copy()
    best_score = current_score
    
    # Track the set of points for quick duplicate checking
    current_pt_set = set(map(tuple, current_points))
    
    # Local search: try moving each point to nearby positions
    # and keep the best improvement found
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        improved = False
        
        # Try moving each point to a nearby position
        for i in range(54):
            if time.time() >= deadline:
                break
            
            old_p = tuple(current_points[i])
            old_x, old_y = old_p
            
            # Try nearby positions in a small neighborhood
            for dx in range(-4, 5):
                if time.time() >= deadline:
                    break
                for dy in range(-4, 5):
                    if dx == 0 and dy == 0:
                        continue
                    new_p = (old_x + dx, old_y + dy)
                    if new_p in current_pt_set:
                        continue
                    
                    # Try this move
                    new_points = current_points.copy()
                    new_points[i] = new_p
                    new_score, _ = _score_and_best_lines(new_points)
                    
                    if new_score > current_score:
                        current_points = new_points
                        current_pt_set.remove(old_p)
                        current_pt_set.add(new_p)
                        current_score = new_score
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_points = current_points.copy()
                            _, best_lines = _score_and_best_lines(current_points)
                        no_improve_count = 0
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturbation: move several points randomly
                for _ in range(3):
                    idx = random.randint(0, 53)
                    dx = random.randint(-8, 8)
                    dy = random.randint(-8, 8)
                    new_p = (current_points[idx][0] + dx, current_points[idx][1] + dy)
                    while new_p in current_pt_set:
                        new_p = (current_points[idx][0] + random.randint(-8, 8),
                                 current_points[idx][1] + random.randint(-8, 8))
                    current_pt_set.remove(tuple(current_points[idx]))
                    current_points[idx] = new_p
                    current_pt_set.add(new_p)
                
                current_score, current_lines = _score_and_best_lines(current_points)
                no_improve_count = 0
    
    return best_points, best_lines, best_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 595  # Safety margin (630s limit, leave buffer)
    
    # Initialize with a structured grid
    # 6x9 grid
    xs = list(range(1, 7))
    ys = list(range(1, 10))
    initial_points = np.array([(x, y) for x in xs for y in ys], dtype=np.int64)
    
    best_points = initial_points.copy()
    best_lines = np.zeros((27, 3), dtype=np.int64)
    best_score = 0
    
    # Run local search from initial configuration
    best_points, best_lines, best_score = _local_search(initial_points, deadline, deadline)
    
    # Check if we have time for more restarts
    elapsed = time.time() - start_time
    time_per_search = elapsed
    if elapsed < 500:
        # Try a few more random restarts
        attempts = 0
        while time.time() < deadline and attempts < 5:
            # Generate a random starting configuration
            # Use a random grid with some perturbation
            random_points = []
            used = set()
            for _ in range(54):
                while True:
                    x = random.randint(1, 15)
                    y = random.randint(1, 15)
                    if (x, y) not in used:
                        used.add((x, y))
                        random_points.append((x, y))
                        break
            
            random_points_arr = np.array(random_points, dtype=np.int64)
            
            # Run local search from this random start
            candidate_points, candidate_lines, candidate_score = _local_search(
                random_points_arr, deadline, deadline
            )
            
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_points
                best_lines = candidate_lines
            
            attempts += 1
    
    # Final validation
    if not _is_valid(best_points, best_lines):
        # Fallback to a known valid configuration
        xs = list(range(1, 7))
        ys = list(range(1, 10))
        fallback_points = np.array([(x, y) for x in xs for y in ys], dtype=np.int64)
        _, fallback_lines = _score_and_best_lines(fallback_points)
        return fallback_points, fallback_lines
    
    return best_points, best_lines
