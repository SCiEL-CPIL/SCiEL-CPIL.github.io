# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
import math
import random

def normalize_line(A, B, C):
    """Normalize a line so first nonzero coefficient is positive and gcd=1."""
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g > 0:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def count_incidences(points, lines):
    """Count total incidences exactly using int64 arithmetic."""
    total = 0
    for i in range(len(points)):
        x, y = points[i]
        for j in range(len(lines)):
            A, B, C = lines[j]
            if A * x + B * y + C == 0:
                total += 1
    return total

def build_lines_from_families(params):
    """
    Build 27 lines from family parameters.
    params: list of (family_type, n_lines, offsets)
    family_type: 'V' (vertical), 'H' (horizontal), 'D1' (slope 1), 'D2' (slope -1),
                 'S' (slope s), 'I' (slope -1/s)
    Returns list of 27 (A, B, C) tuples.
    """
    lines = []
    for ftype, n, offsets in params:
        for c in offsets:
            if ftype == 'V':
                lines.append((1, 0, -c))
            elif ftype == 'H':
                lines.append((0, 1, -c))
            elif ftype == 'D1':
                lines.append((1, 1, -c))
            elif ftype == 'D2':
                lines.append((1, -1, c))
            elif ftype == 'S':
                # slope = c, line: y = c*x + b, but we need integer coefficients
                # Use parametric form: for slope p/q in lowest terms, line is q*y - p*x = const
                # Here c is used as numerator/denominator pair encoded
                pass
            elif ftype == 'I':
                pass
    return lines

def compute_incidence_matrix(points, lines):
    """Compute for each point how many lines pass through it, and which lines."""
    n_pts = len(points)
    n_lines = len(lines)
    inc_count = np.zeros(n_pts, dtype=np.int64)
    inc_lines = [set() for _ in range(n_pts)]
    for i in range(n_pts):
        x, y = points[i]
        for j in range(n_lines):
            A, B, C = lines[j]
            if A * x + B * y + C == 0:
                inc_count[i] += 1
                inc_lines[i].add(j)
    return inc_count, inc_lines

def local_search(points, lines, max_iters=5000, time_limit=30.0):
    """
    Local search: try moving each point to nearby integer positions
    to increase total incidences.
    """
    start_time = time.time()
    current_score = count_incidences(points, lines)
    best_score = current_score
    best_points = points.copy()
    n_pts = len(points)
    n_lines = len(lines)
    
    # Precompute line coefficients as numpy arrays for speed
    line_arr = np.array(lines, dtype=np.int64)
    
    def quick_score(pts):
        """Fast incidence count."""
        total = 0
        for i in range(n_pts):
            x, y = pts[i]
            for j in range(n_lines):
                A, B, C = line_arr[j]
                if A * x + B * y + C == 0:
                    total += 1
        return total
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(n_pts):
            if time.time() - start_time > time_limit:
                break
            x0, y0 = points[i]
            # Try small perturbations
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x0 + dx, y0 + dy
                    # Check if this creates a duplicate point
                    dup = False
                    for k in range(n_pts):
                        if k != i and points[k][0] == nx and points[k][1] == ny:
                            dup = True
                            break
                    if dup:
                        continue
                    old_pt = points[i].copy()
                    points[i] = [nx, ny]
                    new_score = quick_score(points)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        if new_score > best_score:
                            best_score = new_score
                            best_points = points.copy()
                    else:
                        points[i] = old_pt
        
        if not improved:
            # Try larger jumps
            for i in range(n_pts):
                if time.time() - start_time > time_limit:
                    break
                x0, y0 = points[i]
                for _ in range(5):
                    dx = random.randint(-10, 10)
                    dy = random.randint(-10, 10)
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x0 + dx, y0 + dy
                    dup = False
                    for k in range(n_pts):
                        if k != i and points[k][0] == nx and points[k][1] == ny:
                            dup = True
                            break
                    if dup:
                        continue
                    old_pt = points[i].copy()
                    points[i] = [nx, ny]
                    new_score = quick_score(points)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        if new_score > best_score:
                            best_score = new_score
                            best_points = points.copy()
                    else:
                        points[i] = old_pt
                if improved:
                    break
        
        if not improved:
            break
    
    return best_points, best_score

def simulated_annealing(points, lines, time_limit=60.0):
    """
    Simulated annealing to escape local optima.
    """
    start_time = time.time()
    n_pts = len(points)
    n_lines = len(lines)
    line_arr = np.array(lines, dtype=np.int64)
    
    current_score = count_incidences(points, lines)
    best_score = current_score
    best_points = points.copy()
    
    temp = 10.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        temp *= cooling
        if temp < 0.01:
            break
        
        i = random.randint(0, n_pts - 1)
        x0, y0 = points[i]
        dx = random.randint(-5, 5)
        dy = random.randint(-5, 5)
        if dx == 0 and dy == 0:
            continue
        nx, ny = x0 + dx, y0 + dy
        
        # Check duplicate
        dup = False
        for k in range(n_pts):
            if k != i and points[k][0] == nx and points[k][1] == ny:
                dup = True
                break
        if dup:
            continue
        
        old_pt = points[i].copy()
        points[i] = [nx, ny]
        new_score = count_incidences(points, lines)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / max(temp, 1e-10)):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = points.copy()
        else:
            points[i] = old_pt
    
    return best_points, best_score

def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for n=54 points and m=27 lines.
    Uses a multi-phase approach:
    1. Start with 3-direction grid (162 incidences)
    2. Local search to refine point positions
    3. Simulated annealing to escape local optima
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    # Phase 1: Build initial configuration (3-direction grid)
    U = list(range(1, 10))
    V = list(range(1, 10))
    
    sum_freq = {}
    for u in U:
        for v in V:
            s = u + v
            sum_freq[s] = sum_freq.get(s, 0) + 1
    
    sorted_sums = sorted(sum_freq.keys(), key=lambda s: sum_freq[s], reverse=True)
    W = sorted(sorted_sums[:9])
    
    valid_points = []
    for u in U:
        for v in V:
            if u + v in W:
                valid_points.append([u, v])
    
    points = np.array(valid_points[:54], dtype=np.int64).tolist()
    
    lines = []
    for u in U:
        lines.append((1, 0, -u))
    for v in V:
        lines.append((0, 1, -v))
    for w in W:
        lines.append((1, 1, -w))
    
    lines = np.array(lines, dtype=np.int64)
    
    initial_score = count_incidences(points, lines)
    best_score = initial_score
    best_points = points.copy()
    
    # Phase 2: Local search
    if time.time() < deadline - 100:
        ls_points, ls_score = local_search(points.copy(), lines, max_iters=3000, time_limit=40.0)
        if ls_score > best_score:
            best_score = ls_score
            best_points = ls_points
    
    # Phase 3: Simulated annealing from best
    if time.time() < deadline - 30:
        sa_points, sa_score = simulated_annealing(best_points.copy(), lines, time_limit=60.0)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points
    
    # Phase 4: Another local search pass from SA result
    if time.time() < deadline - 10:
        final_points, final_score = local_search(best_points.copy(), lines, max_iters=2000, time_limit=20.0)
        if final_score > best_score:
            best_score = final_score
            best_points = final_points
    
    # Convert to numpy arrays
    points_arr = np.array(best_points, dtype=np.int64)
    
    # Validate
    assert points_arr.shape == (54, 2)
    assert lines.shape == (27, 3)
    
    # Check for duplicate points
    unique_pts = set(map(tuple, points_arr.tolist()))
    if len(unique_pts) != 54:
        # Fallback to initial
        points_arr = np.array(valid_points[:54], dtype=np.int64)
    
    # Check for duplicate lines
    norm_lines = set()
    for l in lines:
        A, B, C = int(l[0]), int(l[1]), int(l[2])
        norm_lines.add(normalize_line(A, B, C))
    if len(norm_lines) != 27:
        raise ValueError("Duplicate lines")
    
    return points_arr, lines

# EVOLVE-BLOCK-END
