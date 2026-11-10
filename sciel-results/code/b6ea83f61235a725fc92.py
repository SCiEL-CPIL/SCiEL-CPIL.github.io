import numpy as np
import time
import random
from math import gcd
from itertools import combinations

def normalize_line(A, B, C):
    if A == 0 and B == 0:
        return None
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        g = 1
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def count_incidences(points, lines):
    P = points.astype(np.int64)
    L = lines.astype(np.int64)
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    A = L[:, 0].reshape(1, -1)
    B = L[:, 1].reshape(1, -1)
    C = L[:, 2].reshape(1, -1)
    vals = A * px + B * py + C
    return int(np.sum(vals == 0))

def point_on_line(pt, line):
    x, y = pt
    A, B, C = line
    return A * x + B * y + C == 0

def line_through_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    nl = normalize_line(A, B, C)
    return nl

def evaluate_full(points, lines):
    """Compute total incidence count."""
    P = points.astype(np.int64)
    L = lines.astype(np.int64)
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    A = L[:, 0].reshape(1, -1)
    B = L[:, 1].reshape(1, -1)
    C = L[:, 2].reshape(1, -1)
    vals = A * px + B * py + C
    return int(np.sum(vals == 0))

def find_best_position_for_point_on_lines(point, lines, other_points, max_coord=50):
    """
    Find the best integer position for a point that maximizes incidences with the given lines.
    Consider positions near the current point and at line intersections.
    """
    best_score = -1
    best_pos = point.copy()
    
    # Generate candidate positions: current point, neighbors, and line intersections
    candidates = set()
    candidates.add(tuple(point))
    
    # Neighbors in a small window
    x0, y0 = int(point[0]), int(point[1])
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            nx, ny = x0 + dx, y0 + dy
            if abs(nx) <= max_coord and abs(ny) <= max_coord:
                candidates.add((nx, ny))
    
    # Intersections of pairs of lines (integer check)
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            A1, B1, C1 = lines[i]
            A2, B2, C2 = lines[j]
            det = A1 * B2 - A2 * B1
            if det == 0:
                continue
            # x = (B1*C2 - B2*C1) / det
            # y = (A2*C1 - A1*C2) / det
            num_x = B1 * C2 - B2 * C1
            num_y = A2 * C1 - A1 * C2
            if num_x % det == 0 and num_y % det == 0:
                ix, iy = num_x // det, num_y // det
                if abs(ix) <= max_coord and abs(iy) <= max_coord:
                    candidates.add((ix, iy))
    
    # Also add points that are already on many lines (from other_points)
    for op in other_points:
        if tuple(op) != tuple(point):
            candidates.add(tuple(op))
    
    best_score = -1
    best_pos = point.copy()
    for cx, cy in candidates:
        pt = np.array([cx, cy], dtype=np.int64)
        # Check if this point would be a duplicate
        if any(np.array_equal(pt, op) for op in other_points):
            continue
        # Count incidences
        score = 0
        for line in lines:
            A, B, C = line
            if A * cx + B * cy + C == 0:
                score += 1
        if score > best_score:
            best_score = score
            best_pos = pt
    
    return best_pos, best_score

def find_best_line_through_points(points, current_lines, max_coord=50):
    """
    Find the line through the most points (among all pairs of points).
    Return the normalized line and its incidence count.
    """
    best_count = 0
    best_line = None
    
    # Count how many points each candidate line passes through
    # Use a dictionary: line -> count
    line_counts = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            p1 = tuple(points[i])
            p2 = tuple(points[j])
            if p1 == p2:
                continue
            nl = line_through_points(p1, p2)
            if nl is None:
                continue
            if nl not in line_counts:
                line_counts[nl] = 0
            line_counts[nl] += 1
    
    for nl, cnt in line_counts.items():
        # cnt is number of point-pairs on this line; actual point count is different
        # We need to recount: for line nl, how many of the 54 points lie on it?
        A, B, C = nl
        actual_count = 0
        for pt in points:
            if A * pt[0] + B * pt[1] + C == 0:
                actual_count += 1
        if actual_count > best_count:
            best_count = actual_count
            best_line = nl
    
    return best_line, best_count

def initial_configuration():
    """Create an initial structured configuration."""
    # Start with a 7x7 grid centered at origin, pick 54 points
    # Lines: 9 vertical, 9 horizontal, 9 diagonal (slope 1)
    # This gives a known baseline.
    
    # 7x7 grid: coords -3 to 3
    coords = list(range(-3, 4))
    all_points = [(x, y) for x in coords for y in coords]
    # Pick 54 out of 49... wait, 7x7=49. Let's use a larger grid.
    # Use 9x9 grid: -4 to 4, 81 points, pick 54.
    coords = list(range(-4, 5))
    all_points = [(x, y) for x in coords for y in coords]
    
    # Select 54 points: all 81 minus 27 randomly? No, let's be structured.
    # Take all points where x+y is even (40 points) + 14 more?
    # Simpler: just take first 54 from a sorted list
    random.seed(42)
    random.shuffle(all_points)
    points = np.array(all_points[:54], dtype=np.int64)
    
    # Generate 27 lines from pairs of these points
    line_set = set()
    for i in range(54):
        for j in range(i + 1, 54):
            nl = line_through_points(tuple(points[i]), tuple(points[j]))
            if nl is not None:
                line_set.add(nl)
            if len(line_set) >= 200:
                break
        if len(line_set) >= 200:
            break
    
    lines_list = list(line_set)
    # Pick 27 lines that cover the most points
    scored_lines = []
    for nl in lines_list:
        A, B, C = nl
        cnt = sum(1 for pt in points if A * pt[0] + B * pt[1] + C == 0)
        scored_lines.append((cnt, nl))
    scored_lines.sort(key=lambda x: -x[0])
    lines = np.array([sl[1] for sl in scored_lines[:27]], dtype=np.int64)
    
    return points, lines

def hill_climb(points, lines, deadline):
    """
    Joint hill-climbing: alternate between moving points and swapping lines.
    """
    current_points = points.copy()
    current_lines = lines.copy()
    current_score = evaluate_full(current_points, current_lines)
    best_points = current_points.copy()
    best_lines = current_lines.copy()
    best_score = current_score
    
    max_coord = 30
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False
        
        # Phase 1: Try to improve by moving one point
        # Pick a random point to move
        idx = random.randint(0, 53)
        old_pt = current_points[idx].copy()
        
        # Generate candidate new positions
        candidates = set()
        x0, y0 = int(old_pt[0]), int(old_pt[1])
        
        # Neighbors
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                nx, ny = x0 + dx, y0 + dy
                if abs(nx) <= max_coord and abs(ny) <= max_coord:
                    candidates.add((nx, ny))
        
        # Line intersections
        for i in range(27):
            for j in range(i + 1, 27):
                A1, B1, C1 = current_lines[i]
                A2, B2, C2 = current_lines[j]
                det = A1 * B2 - A2 * B1
                if det == 0:
                    continue
                num_x = B1 * C2 - B2 * C1
                num_y = A2 * C1 - A1 * C2
                if num_x % det == 0 and num_y % det == 0:
                    ix, iy = num_x // det, num_y // det
                    if abs(ix) <= max_coord and abs(iy) <= max_coord:
                        candidates.add((ix, iy))
        
        # Points already on many lines
        for k in range(54):
            if k != idx:
                candidates.add(tuple(current_points[k]))
        
        best_candidate = None
        best_candidate_score = -1
        
        for cx, cy in candidates:
            # Check duplicate
            dup = False
            for k in range(54):
                if k != idx and current_points[k, 0] == cx and current_points[k, 1] == cy:
                    dup = True
                    break
            if dup:
                continue
            
            # Count incidences for this point with current lines
            score = 0
            for i in range(27):
                A, B, C = current_lines[i]
                if A * cx + B * cy + C == 0:
                    score += 1
            
            # Also consider: does moving this point affect other points' incidences?
            # No, incidences are per (point, line) pair, independent.
            # So the delta is just (new_score_for_this_point - old_score_for_this_point)
            
            if score > best_candidate_score:
                best_candidate_score = score
                best_candidate = (cx, cy)
        
        if best_candidate is not None:
            # Compute old score for this point
            old_score = 0
            for i in range(27):
                A, B, C = current_lines[i]
                if A * old_pt[0] + B * old_pt[1] + C == 0:
                    old_score += 1
            
            delta = best_candidate_score - old_score
            if delta > 0:
                current_points[idx] = best_candidate
                current_score += delta
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
                    best_lines = current_lines.copy()
        
        # Phase 2: Try to improve by swapping one line
        if not improved:
            line_idx = random.randint(0, 26)
            old_line = current_lines[line_idx].copy()
            
            # Find the best alternative line
            # Generate candidate lines from pairs of current points
            candidate_lines = set()
            for i in range(54):
                for j in range(i + 1, 54):
                    nl = line_through_points(tuple(current_points[i]), tuple(current_points[j]))
                    if nl is not None:
                        candidate_lines.add(nl)
            
            # Remove current lines from candidates
            current_line_set = set()
            for i in range(27):
                nl = normalize_line(int(current_lines[i, 0]), int(current_lines[i, 1]), int(current_lines[i, 2]))
                if nl:
                    current_line_set.add(nl)
            
            best_alt = None
            best_alt_score = -1
            
            for nl in candidate_lines:
                if nl in current_line_set:
                    continue
                A, B, C = nl
                cnt = 0
                for k in range(54):
                    if A * current_points[k, 0] + B * current_points[k, 1] + C == 0:
                        cnt += 1
                if cnt > best_alt_score:
                    best_alt_score = cnt
                    best_alt = nl
            
            if best_alt is not None:
                # Compute old line's incidence count
                old_cnt = 0
                A, B, C = int(old_line[0]), int(old_line[1]), int(old_line[2])
                for k in range(54):
                    if A * current_points[k, 0] + B * current_points[k, 1] + C == 0:
                        old_cnt += 1
                
                delta = best_alt_score - old_cnt
                if delta > 0:
                    current_lines[line_idx] = best_alt
                    current_score += delta
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_points = current_points.copy()
                        best_lines = current_lines.copy()
        
        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0
    
    return best_points, best_lines, best_score

def validate_and_fix(points, lines):
    """Ensure all constraints are met."""
    # Ensure 54 unique points
    unique_pts = []
    seen = set()
    for pt in points:
        key = (int(pt[0]), int(pt[1]))
        if key not in seen:
            seen.add(key)
            unique_pts.append(key)
    while len(unique_pts) < 54:
        # Add a new point far away
        unique_pts.append((100 + len(unique_pts), 100 + len(unique_pts)))
    points = np.array(unique_pts[:54], dtype=np.int64)
    
    # Ensure 27 unique valid lines
    unique_lines = []
    seen_lines = set()
    for line in lines:
        nl = normalize_line(int(line[0]), int(line[1]), int(line[2]))
        if nl is not None and nl not in seen_lines:
            seen_lines.add(nl)
            unique_lines.append(nl)
    
    # Pad if needed
    fallback_lines = [
        (1, 0, -50 - i) for i in range(27)
    ] + [
        (0, 1, -50 - i) for i in range(27)
    ] + [
        (1, 1, -50 - i) for i in range(27)
    ]
    for fl in fallback_lines:
        if len(unique_lines) >= 27:
            break
        if fl not in seen_lines:
            seen_lines.add(fl)
            unique_lines.append(fl)
    
    lines = np.array(unique_lines[:27], dtype=np.int64)
    
    return points, lines

def szemeredi_trotter_incidences():
    deadline = time.time() + 570  # 30s safety margin
    
    # Initial configuration
    points, lines = initial_configuration()
    
    # Hill climbing
    points, lines, score = hill_climb(points, lines, deadline - 30)
    
    # Final validation
    points, lines = validate_and_fix(points, lines)
    
    # Final score
    final_score = evaluate_full(points, lines)
    
    return points, lines
