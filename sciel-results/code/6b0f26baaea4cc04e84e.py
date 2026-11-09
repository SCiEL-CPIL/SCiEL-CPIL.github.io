import numpy as np
from math import gcd
from functools import reduce
import time


def _gcd3(a, b, c):
    return reduce(gcd, (abs(int(a)), abs(int(b)), abs(int(c))))


def _normalize(A, B, C):
    A, B, C = int(A), int(B), int(C)
    g = _gcd3(A, B, C)
    if g != 0:
        A //= g
        B //= g
        C //= g
    for v in (A, B, C):
        if v != 0:
            if v < 0:
                A, B, C = -A, -B, -C
            break
    return (A, B, C)


def _count_incidence(points, A, B, C):
    """Count points on line A*x+B*y+C=0."""
    return int(np.sum(A * points[:, 0] + B * points[:, 1] + C == 0))


def _build_grid(a, b):
    """Build a x b grid of points."""
    xs = np.arange(1, a + 1)
    ys = np.arange(1, b + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)


def _generate_candidate_lines(points, max_lines=100):
    """Generate many candidate lines and count incidences for each."""
    n = len(points)
    candidate_lines = []
    
    # Horizontal lines
    for x in range(1, 20):
        candidate_lines.append((1, 0, -x))
    
    # Vertical lines
    for y in range(1, 20):
        candidate_lines.append((0, 1, -y))
    
    # Slope 1: y = x + c → x - y + c = 0
    for c in range(-15, 16):
        candidate_lines.append((1, -1, c))
    
    # Slope -1: y = -x + c → x + y - c = 0
    for c in range(2, 30):
        candidate_lines.append((1, 1, -c))
    
    # Slope 2: y = 2x + c → 2x - y + c = 0
    for c in range(-15, 16):
        candidate_lines.append((2, -1, c))
    
    # Slope 1/2: y = x/2 + c → x - 2y + 2c = 0
    for c in range(-15, 16):
        candidate_lines.append((1, -2, 2 * c))
    
    # Slope -2: y = -2x + c → 2x + y - c = 0
    for c in range(2, 30):
        candidate_lines.append((2, 1, -c))
    
    # Slope -1/2: y = -x/2 + c → x + 2y - 2c = 0
    for c in range(1, 20):
        candidate_lines.append((1, 2, -2 * c))
    
    # Remove duplicates and count incidences
    seen = set()
    scored_lines = []
    for (A, B, C) in candidate_lines:
        norm = _normalize(A, B, C)
        if norm in seen:
            continue
        seen.add(norm)
        cnt = _count_incidence(points, norm[0], norm[1], norm[2])
        if cnt > 0:
            scored_lines.append((cnt, norm))
    
    scored_lines.sort(key=lambda x: -x[0])
    return scored_lines


def _select_best_lines(scored_lines, num_lines, points):
    """Greedy selection of lines to maximize total incidences."""
    selected = []
    remaining_incidences = np.zeros(len(points), dtype=np.int64)
    total = 0
    
    for cnt, norm in scored_lines:
        if len(selected) >= num_lines:
            break
        A, B, C = norm
        mask = (A * points[:, 0] + B * points[:, 1] + C == 0)
        # Check if this line adds new incidences (not already covered)
        # For simplicity, just add if it has any incidence
        if cnt > 0:
            selected.append(norm)
            total += cnt
            remaining_incidences[mask] += 1
    
    return selected, total


def _evaluate(points, lines):
    """Compute total incidences."""
    total = 0
    for (A, B, C) in lines:
        total += _count_incidence(points, A, B, C)
    return total


def _try_grid(a, b, num_lines=27):
    """Try an a x b grid with optimal line selection."""
    if a * b != 54:
        return -1, None, None
    
    points = _build_grid(a, b)
    scored_lines = _generate_candidate_lines(points)
    selected, total = _select_best_lines(scored_lines, num_lines, points)
    
    if len(selected) < num_lines:
        return -1, None, None
    
    # Verify no duplicate lines
    norms = set()
    for norm in selected:
        if norm in norms:
            return -1, None, None
        norms.add(norm)
    
    lines_arr = np.array(selected, dtype=np.int64)
    return total, points, lines_arr


def _try_grid_with_mixed_slopes(a, b, num_lines=27):
    """Try grid with explicit mixed slope selection."""
    if a * b != 54:
        return -1, None, None
    
    points = _build_grid(a, b)
    
    # Generate all possible lines with various slopes
    all_lines = []
    
    # Horizontal: x = k
    for k in range(1, a + 1):
        all_lines.append((1, 0, -k))
    
    # Vertical: y = k
    for k in range(1, b + 1):
        all_lines.append((0, 1, -k))
    
    # Slope 1: x - y + c = 0
    for c in range(-(a + b), a + b + 1):
        all_lines.append((1, -1, c))
    
    # Slope -1: x + y - c = 0
    for c in range(2, a + b + 1):
        all_lines.append((1, 1, -c))
    
    # Slope 2: 2x - y + c = 0
    for c in range(-(2*a + b), 2*a + b + 1):
        all_lines.append((2, -1, c))
    
    # Slope 1/2: x - 2y + c = 0
    for c in range(-(a + 2*b), a + 2*b + 1):
        all_lines.append((1, -2, c))
    
    # Slope -2: 2x + y - c = 0
    for c in range(2, 2*a + b + 1):
        all_lines.append((2, 1, -c))
    
    # Slope -1/2: x + 2y - c = 0
    for c in range(2, a + 2*b + 1):
        all_lines.append((1, 2, -c))
    
    # Slope 3: 3x - y + c = 0
    for c in range(-(3*a + b), 3*a + b + 1):
        all_lines.append((3, -1, c))
    
    # Slope 1/3: x - 3y + c = 0
    for c in range(-(a + 3*b), a + 3*b + 1):
        all_lines.append((1, -3, c))
    
    # Slope -3: 3x + y - c = 0
    for c in range(2, 3*a + b + 1):
        all_lines.append((3, 1, -c))
    
    # Slope -1/3: x + 3y - c = 0
    for c in range(2, a + 3*b + 1):
        all_lines.append((1, 3, -c))
    
    # Remove duplicates
    seen = set()
    unique_lines = []
    for line in all_lines:
        norm = _normalize(*line)
        if norm not in seen:
            seen.add(norm)
            unique_lines.append(norm)
    
    # Score each line
    scored = []
    for (A, B, C) in unique_lines:
        cnt = _count_incidence(points, A, B, C)
        if cnt > 0:
            scored.append((cnt, (A, B, C)))
    
    scored.sort(key=lambda x: -x[0])
    
    # Greedy select top num_lines
    selected = [norm for cnt, norm in scored[:num_lines]]
    
    # Verify no duplicates
    norms = set(selected)
    if len(norms) != len(selected):
        return -1, None, None
    
    total = _evaluate(points, selected)
    lines_arr = np.array(selected, dtype=np.int64)
    return total, points, lines_arr


def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 600
    
    best_total = -1
    best_points = None
    best_lines = None
    
    # Try various grid configurations
    grid_configs = [(6, 9), (9, 6), (3, 18), (18, 3), (2, 27), (27, 2), (1, 54), (54, 1)]
    
    for a, b in grid_configs:
        if time.time() > deadline - 30:
            break
        total, pts, lns = _try_grid_with_mixed_slopes(a, b, 27)
        if total > best_total:
            best_total = total
            best_points = pts
            best_lines = lns
    
    if best_points is None:
        # Fallback: 6x9 grid with basic lines
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = []
        for i in range(1, 7):
            lines.append((1, 0, -i))
        for j in range(1, 10):
            lines.append((0, 1, -j))
        c_vals = [0, 1, 2, 3, -1, 4, -2, 5, -3, 6, -4, 7]
        for c in c_vals:
            lines.append((1, -1, c))
        best_lines = np.array(lines, dtype=np.int64)
        best_total = _evaluate(best_points, best_lines)
    
    return best_points, best_lines
