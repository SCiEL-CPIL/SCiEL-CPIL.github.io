# EVOLVE-BLOCK-START
import numpy as np
from math import gcd
import time


def _line_key(A, B, C):
    """Normalize a line (A,B,C) for deduplication."""
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _count_incidence(points, lines):
    """Count point-line incidences exactly using vectorized operations."""
    # points: (n, 2), lines: (m, 3)
    # For each line (A,B,C) and point (x,y), check A*x + B*y + C == 0
    # Vectorized: (m, 1, 2) * (1, n, 2) sum over last axis + (m, 1) * C
    A = lines[:, 0:1]  # (m, 1)
    B = lines[:, 1:2]  # (m, 1)
    C = lines[:, 2:3]  # (m, 1)
    x = points[:, 0:1].T  # (1, n)
    y = points[:, 1:2].T  # (1, n)
    vals = A * x + B * y + C  # (m, n)
    return int(np.sum(vals == 0))


def _point_incidence(point, lines):
    """Count how many lines pass through a given point."""
    x, y = point
    A = lines[:, 0]
    B = lines[:, 1]
    C = lines[:, 2]
    vals = A * x + B * y + C
    return int(np.sum(vals == 0))


def _line_incidence(line, points):
    """Count how many points lie on a given line."""
    A, B, C = line
    x = points[:, 0]
    y = points[:, 1]
    vals = A * x + B * y + C
    return int(np.sum(vals == 0))


def _generate_candidate_lines(points):
    """Generate candidate lines from pairs of points and common patterns."""
    candidates = set()
    n = points.shape[0]
    
    # Lines through pairs of points
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Line: dy*(x - x1) - dx*(y - y1) = 0
            # dy*x - dx*y + (dx*y1 - dy*x1) = 0
            A = int(dy)
            B = int(-dx)
            C = int(dx * y1 - dy * x1)
            key = _line_key(A, B, C)
            if key != (0, 0, 0):
                candidates.add(key)
    
    # Horizontal lines through each y-coordinate
    for y in set(points[:, 1].tolist()):
        candidates.add(_line_key(0, 1, -int(y)))
    
    # Vertical lines through each x-coordinate
    for x in set(points[:, 0].tolist()):
        candidates.add(_line_key(1, 0, -int(x)))
    
    return candidates


def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines.
    Uses Szemeredi-Trotter grid construction as baseline, then local search refinement.
    """
    deadline = time.time() + 580  # safety margin
    
    # Start with the classical Szemeredi-Trotter construction
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    rows = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            rows.append((a, -1, b))
    lines = np.array(rows, dtype=np.int64)
    
    best_incidence = _count_incidence(points, lines)
    
    # Local search: try moving individual points to better positions
    # and adjusting lines to pass through more points
    max_iter = 300
    for iteration in range(max_iter):
        if time.time() > deadline - 60:
            break
        
        improved = False
        
        # Try moving each point to a nearby integer position
        for i in range(points.shape[0]):
            old_x, old_y = points[i]
            old_inc = _point_incidence((int(old_x), int(old_y)), lines)
            best_new = (int(old_x), int(old_y))
            best_new_inc = best_incidence
            
            # Search in a larger neighborhood
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = int(old_x) + dx, int(old_y) + dy
                    # Check for duplicate points
                    if np.any((points[:, 0] == new_x) & (points[:, 1] == new_y)):
                        continue
                    # Count incidences with new point
                    new_point_inc = _point_incidence((new_x, new_y), lines)
                    total_inc = best_incidence - old_inc + new_point_inc
                    if total_inc > best_new_inc:
                        best_new_inc = total_inc
                        best_new = (new_x, new_y)
            
            if best_new != (int(old_x), int(old_y)):
                points[i] = best_new
                best_incidence = best_new_inc
                improved = True
        
        # Try adjusting each line to pass through more points
        for j in range(lines.shape[0]):
            A, B, C = int(lines[j, 0]), int(lines[j, 1]), int(lines[j, 2])
            current_line_inc = _line_incidence((A, B, C), points)
            
            best_new_line = (A, B, C)
            best_new_line_inc = current_line_inc
            best_total_inc = best_incidence
            
            # Try nearby lines
            for dA in range(-3, 4):
                for dB in range(-3, 4):
                    for dC in range(-8, 9):
                        if dA == 0 and dB == 0 and dC == 0:
                            continue
                        new_A, new_B, new_C = A + dA, B + dB, C + dC
                        if new_A == 0 and new_B == 0:
                            continue
                        # Check for duplicate lines
                        norm = _line_key(new_A, new_B, new_C)
                        is_dup = False
                        for k2 in range(lines.shape[0]):
                            if k2 == j:
                                continue
                            if _line_key(int(lines[k2, 0]), int(lines[k2, 1]), int(lines[k2, 2])) == norm:
                                is_dup = True
                                break
                        if is_dup:
                            continue
                        # Count incidences
                        new_line_inc = _line_incidence((new_A, new_B, new_C), points)
                        total_inc = best_incidence - current_line_inc + new_line_inc
                        if total_inc > best_total_inc:
                            best_total_inc = total_inc
                            best_new_line = (new_A, new_B, new_C)
                            best_new_line_inc = new_line_inc
            
            if best_new_line != (A, B, C):
                lines[j] = best_new_line
                best_incidence = best_total_inc
                improved = True
        
        # Try replacing a line with a better candidate line from point pairs
        if time.time() < deadline - 120:
            candidate_lines = _generate_candidate_lines(points)
            # Convert to list of (A, B, C) tuples
            cand_list = [(a, b, c) for (a, b, c) in candidate_lines]
            
            # Find the line with the fewest incidences (worst line)
            line_incs = [_line_incidence((int(lines[j, 0]), int(lines[j, 1]), int(lines[j, 2])), points) 
                        for j in range(lines.shape[0])]
            worst_line_idx = int(np.argmin(line_incs))
            worst_line_inc = line_incs[worst_line_idx]
            
            # Find the best candidate line that isn't already present
            existing_keys = set()
            for j in range(lines.shape[0]):
                existing_keys.add(_line_key(int(lines[j, 0]), int(lines[j, 1]), int(lines[j, 2])))
            
            best_cand_inc = worst_line_inc
            best_cand_line = None
            for (a, b, c) in cand_list:
                if (a, b, c) in existing_keys:
                    continue
                inc = _line_incidence((a, b, c), points)
                if inc > best_cand_inc:
                    best_cand_inc = inc
                    best_cand_line = (a, b, c)
            
            if best_cand_line is not None:
                lines[worst_line_idx] = best_cand_line
                best_incidence = best_incidence - worst_line_inc + best_cand_inc
                improved = True
        
        if not improved:
            break
    
    return points, lines


# EVOLVE-BLOCK-END
