import numpy as np
import time
import random
from math import gcd

def _normalize_line(A, B, C):
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _line_key(A, B, C):
    return _normalize_line(A, B, C)

def _count_incidentences(points, lines):
    # points: (54,2) int64, lines: (27,3) int64
    # Vectorized: for each line, check all points
    # lines: (27,3), points: (54,2)
    # A*x + B*y + C == 0
    # Use broadcasting: (27,1,3) * (1,54,2) doesn't work directly
    # Better: (27,1,1)*pts[:,0] + (27,1,1)*pts[:,1] ... 
    # Actually: lines[:, 0:1] * points[:, 0:1].T + lines[:, 1:2] * points[:, 1:2].T + lines[:, 2:3]
    # This gives (27, 54) matrix
    val = (lines[:, 0:1] * points[:, 0:1].T + 
           lines[:, 1:2] * points[:, 1:2].T + 
           lines[:, 2:3])
    return int(np.sum(val == 0))

def _is_valid(points, lines):
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
        key = _line_key(Ai, Bi, Ci)
        if key in line_set:
            return False
        line_set.add(key)
    return True

def _best_lines_for_points(points, num_lines=27):
    """Find the top num_lines lines by incidence count for given points."""
    n = len(points)
    # Generate candidate lines from all pairs of points
    candidates = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            key = _line_key(A, B, C)
            if key[0] == 0 and key[1] == 0:
                continue
            if key not in candidates:
                candidates[key] = 0
            # Count how many points lie on this line
            # We'll count later in batch
    # Count incidences for each candidate line
    # To speed up, only count for lines that pass through at least 2 points
    # Actually all candidates pass through at least 2 points by construction
    # Let's count for each unique line
    line_list = list(candidates.keys())
    if not line_list:
        return np.zeros((num_lines, 3), dtype=np.int64)
    
    lines_arr = np.array(line_list, dtype=np.int64)
    # Vectorized count
    val = (lines_arr[:, 0:1] * points[:, 0:1].T + 
           lines_arr[:, 1:2] * points[:, 1:2].T + 
           lines_arr[:, 2:3])
    counts = np.sum(val == 0, axis=1)
    
    # Sort by count descending
    order = np.argsort(-counts)
    top = order[:num_lines]
    result = lines_arr[top]
    return result

def _make_initial_grid():
    """Create a 6x9 grid as starting point."""
    xs = list(range(1, 7))
    ys = list(range(1, 10))
    points = np.array([(x, y) for x in xs for y in ys], dtype=np.int64)
    return points

def _perturb_points(points, n_perturb=2):
    """Randomly perturb some points to nearby integer positions."""
    new_points = points.copy()
    n = len(points)
    for _ in range(n_perturb):
        idx = random.randint(0, n-1)
        dx = random.randint(-3, 3)
        dy = random.randint(-3, 3)
        new_points[idx, 0] = points[idx, 0] + dx
        new_points[idx, 1] = points[idx, 1] + dy
    return new_points

def _perturb_lines(lines, n_perturb=1, points=None):
    """Randomly perturb some lines. If points given, may generate lines from point pairs."""
    new_lines = lines.copy()
    m = len(new_lines)
    for _ in range(n_perturb):
        if points is not None and random.random() < 0.5:
            # Generate a new line from a random pair of points
            i1 = random.randint(0, len(points)-1)
            i2 = random.randint(0, len(points)-1)
            if i1 == i2:
                continue
            x1, y1 = points[i1]
            x2, y2 = points[i2]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            key = _line_key(A, B, C)
            if key[0] == 0 and key[1] == 0:
                continue
            idx = random.randint(0, m-1)
            new_lines[idx] = key
        else:
            # Perturb an existing line slightly
            idx = random.randint(0, m-1)
            A, B, C = new_lines[idx]
            dA = random.randint(-2, 2)
            dB = random.randint(-2, 2)
            dC = random.randint(-5, 5)
            new_A, new_B, new_C = A + dA, B + dB, C + dC
            if new_A == 0 and new_B == 0:
                continue
            key = _line_key(new_A, new_B, new_C)
            if key[0] == 0 and key[1] == 0:
                continue
            new_lines[idx] = key
    return new_lines

def _check_and_fix_duplicates(points, lines):
    """Check for duplicate points and lines, fix if needed."""
    # Check points
    pt_list = [(int(points[i,0]), int(points[i,1])) for i in range(len(points))]
    if len(set(pt_list)) != len(pt_list):
        return False
    # Check lines
    line_keys = set()
    for i in range(len(lines)):
        A, B, C = int(lines[i,0]), int(lines[i,1]), int(lines[i,2])
        if A == 0 and B == 0:
            return False
        key = _line_key(A, B, C)
        if key in line_keys:
            return False
        line_keys.add(key)
    return True

def szemeredi_trotter_incidences():
    deadline = time.time() + 600
    safety_margin = 30
    
    # Start with 6x9 grid
    points = _make_initial_grid()
    lines = _best_lines_for_points(points, 27)
    
    best_points = points.copy()
    best_lines = lines.copy()
    best_count = _count_incidentences(best_points, best_lines)
    
    # Simulated annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.9995
    
    while time.time() < deadline - safety_margin:
        # Generate candidate
        if random.random() < 0.4:
            # Perturb points
            cand_points = _perturb_points(points, random.randint(1, 3))
            cand_lines = _best_lines_for_points(cand_points, 27)
        else:
            # Perturb lines
            cand_points = points.copy()
            cand_lines = _perturb_lines(lines, random.randint(1, 2), points)
        
        # Validate
        if not _check_and_fix_duplicates(cand_points, cand_lines):
            continue
        
        cand_count = _count_incidentences(cand_points, cand_lines)
        
        # Accept if better, or with probability exp((cand - best)/T)
        delta = cand_count - best_count
        if delta > 0 or random.random() < np.exp(delta / max(T, 0.01)):
            points = cand_points
            lines = cand_lines
            best_count = cand_count
            if cand_count > best_count:
                pass  # already updated
            # Track best
            if cand_count >= best_count:
                best_points = cand_points.copy()
                best_lines = cand_lines.copy()
                best_count = cand_count
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Final: make sure best is valid
    if not _is_valid(best_points, best_lines):
        # Fallback
        return points, lines
    
    return best_points, best_lines
