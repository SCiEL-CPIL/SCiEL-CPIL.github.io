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
    """Count point-line incidences exactly."""
    count = 0
    for i in range(points.shape[0]):
        x, y = points[i]
        for j in range(lines.shape[0]):
            A, B, C = lines[j]
            if A * x + B * y + C == 0:
                count += 1
    return count


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
    max_iter = 200
    for iteration in range(max_iter):
        if time.time() > deadline - 60:
            break
        
        improved = False
        
        # Try moving each point to a nearby integer position
        for i in range(points.shape[0]):
            old_x, old_y = points[i]
            best_new = (old_x, old_y)
            best_new_inc = best_incidence
            
            # Search in a small neighborhood
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = old_x + dx, old_y + dy
                    # Check for duplicate points
                    if np.any((points[:, 0] == new_x) & (points[:, 1] == new_y)):
                        continue
                    # Count incidences with new point
                    inc = 0
                    for j in range(lines.shape[0]):
                        A, B, C = lines[j]
                        if A * new_x + B * new_y + C == 0:
                            inc += 1
                    # Total incidence = (best_incidence - incidences_of_old_point) + inc
                    old_inc = 0
                    for j in range(lines.shape[0]):
                        A, B, C = lines[j]
                        if A * old_x + B * old_y + C == 0:
                            old_inc += 1
                    total_inc = best_incidence - old_inc + inc
                    if total_inc > best_new_inc:
                        best_new_inc = total_inc
                        best_new = (new_x, new_y)
            
            if best_new != (old_x, old_y):
                points[i] = best_new
                best_incidence = best_new_inc
                improved = True
        
        # Try adjusting each line to pass through more points
        for j in range(lines.shape[0]):
            A, B, C = lines[j]
            # Find points on this line
            on_line = []
            for i in range(points.shape[0]):
                x, y = points[i]
                if A * x + B * y + C == 0:
                    on_line.append(i)
            
            # Try perturbing the line slightly
            best_new_line = (A, B, C)
            best_new_inc = best_incidence
            best_new_line_inc = len(on_line)
            
            # Try nearby lines
            for dA in range(-2, 3):
                for dB in range(-2, 3):
                    for dC in range(-5, 6):
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
                        inc = 0
                        for i in range(points.shape[0]):
                            x, y = points[i]
                            if new_A * x + new_B * y + new_C == 0:
                                inc += 1
                        total_inc = best_incidence - best_new_line_inc + inc
                        if total_inc > best_new_inc:
                            best_new_inc = total_inc
                            best_new_line = (new_A, new_B, new_C)
            
            if best_new_line != (A, B, C):
                lines[j] = best_new_line
                best_incidence = best_new_inc
                improved = True
        
        if not improved:
            break
    
    return points, lines


# EVOLVE-BLOCK-END
