# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd
from collections import defaultdict


def _normalize_line(A, B, C):
    if A == 0 and B == 0:
        return None
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A //= g
        B //= g
        C //= g
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)


def _get_all_lines_and_counts(points):
    """Enumerate all lines through >=2 points and count incidences."""
    pts = points.tolist()
    n_pts = len(pts)
    line_to_points = defaultdict(set)
    
    for i in range(n_pts):
        x1, y1 = pts[i]
        for j in range(i + 1, n_pts):
            x2, y2 = pts[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            key = _normalize_line(A, B, C)
            if key is not None:
                line_to_points[key].add(i)
    
    result = []
    for key, pts_set in line_to_points.items():
        if len(pts_set) >= 2:
            result.append((key, len(pts_set)))
    return result


def _evaluate_configuration(points, lines):
    """Count total incidences."""
    I = 0
    for x, y in points:
        for A, B, C in lines:
            if A * x + B * y + C == 0:
                I += 1
    return I


def _generate_grid(nx, ny):
    """Generate a nx*ny grid of points."""
    xs = np.arange(nx, dtype=np.int64)
    ys = np.arange(ny, dtype=np.int64)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)


def _local_search_points(points, lines, deadline, max_sweeps=50):
    """Local search: try moving each point to nearby integer positions to increase incidences."""
    pts = [tuple(p) for p in points.tolist()]
    lines_list = [tuple(l) for l in lines]
    n_pts = len(pts)
    n_lines = len(lines_list)
    
    current_I = _evaluate_configuration(pts, lines_list)
    best_I = current_I
    best_pts = pts[:]
    
    A_arr = np.array([l[0] for l in lines_list], dtype=np.int64)
    B_arr = np.array([l[1] for l in lines_list], dtype=np.int64)
    C_arr = np.array([l[2] for l in lines_list], dtype=np.int64)
    
    def count_incidence_at(x, y):
        vals = A_arr * x + B_arr * y + C_arr
        return int(np.sum(vals == 0))
    
    for sweep in range(max_sweeps):
        if time.time() > deadline - 5:
            break
        
        improved = False
        occupied = set(pts)
        
        for i in range(n_pts):
            if time.time() > deadline - 5:
                break
            
            x, y = pts[i]
            current_point_inc = count_incidence_at(x, y)
            
            best_new_inc = current_point_inc
            best_new_pos = (x, y)
            
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in occupied and (nx, ny) != (x, y):
                        continue
                    new_inc = count_incidence_at(nx, ny)
                    if new_inc > best_new_inc:
                        best_new_inc = new_inc
                        best_new_pos = (nx, ny)
            
            if best_new_pos != (x, y):
                old_pos = pts[i]
                pts[i] = best_new_pos
                occupied.discard(old_pos)
                occupied.add(best_new_pos)
                current_I += (best_new_inc - current_point_inc)
                if current_I > best_I:
                    best_I = current_I
                    best_pts = pts[:]
                improved = True
        
        if not improved:
            break
    
    return np.array(best_pts, dtype=np.int64), best_I


def szemeredi_trotter_incidences():
    """
    Point-line incidence maximization with n=54 points and m=27 lines.
    
    Strategy: Start with a grid, find best 27 lines, then do local search on points,
    then re-optimize lines, then another local search pass.
    """
    start = time.time()
    deadline = start + 610.0  # safety margin
    
    # ---- Start with 9x6 grid ----
    points = _generate_grid(9, 6)
    
    # ---- Find all candidate lines and pick top 27 ----
    candidate_lines = _get_all_lines_and_counts(points)
    candidate_lines.sort(key=lambda t: (-t[1], t[0]))
    
    if len(candidate_lines) < 27:
        while len(candidate_lines) < 27:
            candidate_lines.append(((1, 0, 0), 0))
    
    selected_lines = [key for key, _ in candidate_lines[:27]]
    lines = np.array(selected_lines, dtype=np.int64)
    
    current_I = _evaluate_configuration(points, lines)
    
    # ---- Local search on point positions ----
    if time.time() < deadline - 30:
        points, local_I = _local_search_points(points, lines, deadline)
        if local_I > current_I:
            current_I = local_I
    
    # ---- After moving points, re-optimize lines ----
    if time.time() < deadline - 20:
        candidate_lines2 = _get_all_lines_and_counts(points)
        candidate_lines2.sort(key=lambda t: (-t[1], t[0]))
        
        if len(candidate_lines2) >= 27:
            new_lines = [key for key, _ in candidate_lines2[:27]]
            new_I = _evaluate_configuration(points, new_lines)
            if new_I > current_I:
                lines = np.array(new_lines, dtype=np.int64)
                current_I = new_I
    
    # ---- Second round of local search with updated lines ----
    if time.time() < deadline - 10:
        points, local_I2 = _local_search_points(points, lines, deadline)
        if local_I2 > current_I:
            current_I = local_I2
    
    # Final verification
    assert points.shape == (54, 2)
    assert lines.shape == (27, 3)
    assert not np.any(np.isnan(points.astype(float)))
    assert not np.any(np.isinf(points.astype(float)))
    assert not np.any(np.isnan(lines.astype(float)))
    assert not np.any(np.isinf(lines.astype(float)))
    assert not np.all(lines[:, :2] == 0, axis=1).any()
    
    # Check for duplicate points
    point_tuples = set(map(tuple, points.tolist()))
    assert len(point_tuples) == 54, f"Duplicate points found: {54 - len(point_tuples)}"
    
    # Check for duplicate lines
    line_tuples = set(map(tuple, lines.tolist()))
    assert len(line_tuples) == 27, f"Duplicate lines found: {27 - len(line_tuples)}"
    
    return points, lines


# EVOLVE-BLOCK-END
