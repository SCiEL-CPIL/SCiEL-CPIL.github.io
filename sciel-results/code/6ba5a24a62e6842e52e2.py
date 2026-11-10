import numpy as np
import math
import time
from itertools import combinations

def _normalize_line(A, B, C):
    """Normalize line coefficients to canonical form."""
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _count_incidence(line, points):
    """Count points on a given line. line is (A, B, C), points is (N, 2) array."""
    A, B, C = line
    vals = A * points[:, 0] + B * points[:, 1] + C
    return int(np.sum(vals == 0))

def _generate_all_lines_from_points(points):
    """Generate all unique lines determined by pairs of points."""
    n = len(points)
    unique_lines = {}
    p_list = points.tolist()
    for i in range(n):
        x1, y1 = p_list[i]
        for j in range(i + 1, n):
            x2, y2 = p_list[j]
            A = y2 - y1
            B = x1 - x2
            C = -(A * x1 + B * y1)
            key = _normalize_line(A, B, C)
            if key != (0, 0, 0):
                unique_lines[key] = None
    return list(unique_lines.keys())

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = 600.0
    
    # Clique-star hybrid: 27 points on vertical line x=0, 27 on 3x9 grid
    # Vertical line points: (0, 0), (0, 1), ..., (0, 26)
    # Grid points: x in {1, 2, 3}, y in {0, 1, ..., 8}
    
    vline_points = [(0, y) for y in range(27)]
    grid_xs = [1, 2, 3]
    grid_ys = list(range(9))
    grid_points = [(x, y) for x in grid_xs for y in grid_ys]
    
    all_points = vline_points + grid_points
    points = np.array(all_points, dtype=np.int64)
    
    # Verify 54 unique points
    assert len(points) == 54
    assert len(set(map(tuple, points.tolist()))) == 54
    
    # Generate all lines from all point pairs
    all_lines = _generate_all_lines_from_points(points)
    
    # Count incidences for each line
    line_counts = []
    for line in all_lines:
        count = _count_incidence(line, points)
        line_counts.append((count, line))
    
    # Sort by incidence count descending
    line_counts.sort(key=lambda x: x[0], reverse=True)
    
    # Select top 27 lines
    selected_lines = [line_counts[i][1] for i in range(min(27, len(line_counts)))]
    
    # Local search: try swapping to improve total
    selected_set = set(selected_lines)
    total_incidence = sum(_count_incidence(line, points) for line in selected_lines)
    
    remaining = [(c, l) for c, l in line_counts if l not in selected_set]
    remaining.sort(key=lambda x: x[0], reverse=True)
    
    improved = True
    max_iterations = 500
    
    while improved and max_iterations > 0:
        improved = False
        max_iterations -= 1
        
        if time.time() > deadline:
            break
        
        # Compute per-line incidence for current selection
        line_counts_map = {}
        for line in selected_lines:
            line_counts_map[line] = _count_incidence(line, points)
        
        # Find worst line in selection
        worst_idx = min(range(len(selected_lines)), key=lambda i: line_counts_map[selected_lines[i]])
        worst_line = selected_lines[worst_idx]
        worst_count = line_counts_map[worst_line]
        
        # Try swapping with best remaining candidates
        best_gain = 0
        best_swap_line = None
        
        for count, candidate in remaining[:100]:
            if time.time() > deadline:
                break
            gain = count - worst_count
            if gain > best_gain:
                best_gain = gain
                best_swap_line = candidate
        
        if best_swap_line is not None and best_gain > 0:
            selected_lines[worst_idx] = best_swap_line
            selected_set.discard(worst_line)
            selected_set.add(best_swap_line)
            remaining = [(c, l) for c, l in remaining if l != best_swap_line]
            remaining.append((worst_count, worst_line))
            remaining.sort(key=lambda x: x[0], reverse=True)
            improved = True
    
    lines = np.array(selected_lines, dtype=np.int64)
    
    return points, lines
