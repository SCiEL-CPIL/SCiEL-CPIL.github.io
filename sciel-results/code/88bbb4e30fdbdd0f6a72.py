# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import time
from math import gcd

def _line_from_two_points(p1, p2):
    """Compute (A, B, C) for line through p1, p2, normalized."""
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    # Normalize: divide by gcd, make first nonzero positive
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        g = 1
    A //= g
    B //= g
    C //= g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _line_key(line):
    """Canonical key for a line (A, B, C) after normalization."""
    A, B, C = line
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        g = 1
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidences(points, lines):
    """Count point-line incidences exactly."""
    n, m = len(points), len(lines)
    count = 0
    for i in range(n):
        x, y = points[i]
        for j in range(m):
            A, B, C = lines[j]
            if A * x + B * y + C == 0:
                count += 1
    return count

def _find_best_lines_for_points(points, num_lines):
    """Find the best num_lines lines for a given set of points."""
    n = len(points)
    # For each pair of points, compute the line and count how many points lie on it
    line_points = {}  # line_key -> set of point indices
    
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(_line_from_two_points(points[i], points[j]))
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    
    # Sort lines by number of points they pass through
    sorted_lines = sorted(line_points.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Greedily select lines, preferring those that cover many points
    # But we need to be smart: select lines that maximize total incidences
    # A simple approach: sort by count descending, take top num_lines
    # Better: use a greedy approach that considers marginal gain
    
    selected_lines = []
    selected_keys = set()
    covered = [0] * n  # count of lines covering each point
    
    for key, pts in sorted_lines:
        if len(selected_lines) >= num_lines:
            break
        if key in selected_keys:
            continue
        # Compute marginal gain: number of points on this line not yet "well covered"
        # Actually, each incidence counts independently, so marginal gain is just len(pts)
        # But we want diversity. Let's just take the top ones.
        selected_lines.append(key)
        selected_keys.add(key)
    
    # If we don't have enough, we need to add more lines
    # This shouldn't happen for reasonable point sets
    while len(selected_lines) < num_lines:
        # Find a new line not already selected
        added = False
        for key, pts in sorted_lines:
            if key not in selected_keys:
                selected_lines.append(key)
                selected_keys.add(key)
                added = True
                break
        if not added:
            # Should not happen
            break
    
    return [list(k) for k in selected_lines]

def _optimize_lines(points, num_lines, max_iters=500):
    """Local search to optimize line selection for given points."""
    n = len(points)
    
    # Initial: greedy selection
    current_lines = _find_best_lines_for_points(points, num_lines)
    current_score = _count_incidences(points, current_lines)
    
    # Local search: try replacing one line with a better one
    # Precompute all candidate lines
    all_candidate_lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(_line_from_two_points(points[i], points[j]))
            if key not in all_candidate_lines:
                all_candidate_lines[key] = set()
            all_candidate_lines[key].add(i)
            all_candidate_lines[key].add(j)
    
    candidate_keys = list(all_candidate_lines.keys())
    candidate_counts = {k: len(v) for k, v in all_candidate_lines.items()}
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        current_set = set(tuple(l) for l in current_lines)
        
        for idx in range(num_lines):
            # Try replacing line idx with each candidate
            best_gain = 0
            best_candidate = None
            current_line_key = tuple(current_lines[idx])
            current_line_pts = all_candidate_lines.get(current_line_key, set())
            
            for cand_key in candidate_keys:
                if cand_key in current_set:
                    continue
                cand_pts = all_candidate_lines[cand_key]
                # Gain = |cand_pts| - |current_line_pts|
                # But this is approximate; true gain depends on overlaps
                # For exact: new_score - old_score
                # Approximate: use set difference
                gain = len(cand_pts - current_line_pts) - len(current_line_pts - cand_pts)
                # Actually, since each incidence is independent, the gain is just
                # len(cand_pts) - len(current_line_pts) only if no other line covers those points
                # But other lines might cover them too, so this is an upper bound
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = cand_key
            
            if best_candidate is not None and best_gain > 0:
                # Verify with exact count
                old_score = current_score
                new_lines = current_lines.copy()
                new_lines[idx] = list(best_candidate)
                new_score = _count_incidences(points, new_lines)
                if new_score > current_score:
                    current_lines = new_lines
                    current_score = new_score
                    current_set = set(tuple(l) for l in current_lines)
                    improved = True
                    break  # Restart from beginning
    
    return current_lines, current_score

def _perturb_points(points, num_perturb=1):
    """Perturb some points by small integer amounts."""
    new_points = points.copy()
    n = len(new_points)
    for _ in range(num_perturb):
        idx = np.random.randint(n)
        dx = np.random.randint(-3, 4)
        dy = np.random.randint(-3, 4)
        if dx == 0 and dy == 0:
            continue
        new_points[idx, 0] += dx
        new_points[idx, 1] += dy
    return new_points

def _check_valid(points, lines):
    """Check if the configuration is valid."""
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    # Check no NaN/inf
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(lines)):
        return False
    # Check integer
    if not np.allclose(points, np.round(points)) or not np.allclose(lines, np.round(lines)):
        return False
    # Check no repeated points
    if len(set(map(tuple, points))) != 54:
        return False
    # Check no zero lines
    for A, B, C in lines:
        if A == 0 and B == 0:
            return False
    # Check no duplicate lines
    line_keys = set()
    for A, B, C in lines:
        key = _line_key((A, B, C))
        if key in line_keys:
            return False
        line_keys.add(key)
    return True

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 600  # 600 second budget
    
    np.random.seed(42)
    
    best_score = 0
    best_points = None
    best_lines = None
    
    # Strategy 1: Start with a grid and optimize
    # Try different grid dimensions
    grid_configs = [
        (6, 9),   # 6x9 = 54
        (9, 6),
        (3, 18),
        (18, 3),
        (2, 27),
        (27, 2),
        (1, 54),
        (54, 1),
    ]
    
    for gx, gy in grid_configs:
        if time.time() > deadline - 60:
            break
        if gx * gy != 54:
            continue
        
        # Create grid points
        xs = np.arange(1, gx + 1)
        ys = np.arange(1, gy + 1)
        px, py = np.meshgrid(xs, ys, indexing='ij')
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        # Optimize lines
        lines, score = _optimize_lines(points, 27, max_iters=200)
        
        if _check_valid(points, lines) and score > best_score:
            best_score = score
            best_points = points.copy()
            best_lines = np.array(lines, dtype=np.int64)
    
    # Strategy 2: Local search on points
    if best_points is not None and time.time() < deadline - 30:
        current_points = best_points.copy()
        current_lines = best_lines.copy()
        current_score = best_score
        
        while time.time() < deadline - 10:
            # Perturb points
            new_points = _perturb_points(current_points, num_perturb=1)
            
            # Check validity
            if len(set(map(tuple, new_points))) != 54:
                continue
            if not np.all(np.isfinite(new_points)):
                continue
            
            # Optimize lines for new points
            new_lines, new_score = _optimize_lines(new_points, 27, max_iters=100)
            
            if new_score > current_score:
                current_points = new_points
                current_lines = np.array(new_lines, dtype=np.int64)
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
                    best_lines = current_lines.copy()
    
    # Strategy 3: Try some random point configurations
    if time.time() < deadline - 30:
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            # Random points in a box
            scale = np.random.choice([5, 10, 15, 20])
            points = np.random.randint(-scale, scale + 1, size=(54, 2)).astype(np.int64)
            
            # Ensure no duplicates
            while len(set(map(tuple, points))) < 54:
                idx = np.random.randint(54)
                points[idx] = np.random.randint(-scale, scale + 1, size=2)
            
            lines, score = _optimize_lines(points, 27, max_iters=50)
            
            if _check_valid(points, lines) and score > best_score:
                best_score = score
                best_points = points.copy()
                best_lines = np.array(lines, dtype=np.int64)
    
    # Final validation
    if best_points is None:
        # Fallback to original grid
        k = 3
        xs = np.arange(1, k + 1)
        ys = np.arange(1, 2 * k * k + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        rows = []
        for a in range(1, k + 1):
            for b in range(1, k * k + 1):
                rows.append((a, -1, b))
        best_lines = np.array(rows, dtype=np.int64)
    
    return best_points, best_lines

# EVOLVE-BLOCK-END
