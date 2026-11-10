# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd
from itertools import combinations

def _normalize_line(A, B, C):
    """Normalize line coefficients so first nonzero is positive and gcd divides all."""
    if A == 0 and B == 0:
        return None
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return None
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidences(points, lines):
    """Exact incidence count via int64 arithmetic."""
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    total = 0
    for i in range(lns.shape[0]):
        A, B, C = lns[i]
        vals = A * pts[:, 0] + B * pts[:, 1] + C
        total += int(np.sum(vals == 0))
    return total

def _build_grid_points(r, c):
    """r x c grid of points."""
    xs = np.arange(1, c + 1, dtype=np.int64)
    ys = np.arange(1, r + 1, dtype=np.int64)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def _generate_candidate_lines(points):
    """Generate all unique lines defined by pairs of points."""
    n = points.shape[0]
    lines_dict = {}
    for i in range(n):
        x1, y1 = int(points[i, 0]), int(points[i, 1])
        for j in range(i + 1, n):
            x2, y2 = int(points[j, 0]), int(points[j, 1])
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            norm = _normalize_line(A, B, C)
            if norm is not None:
                lines_dict[norm] = 1
    return list(lines_dict.keys())

def _select_best_lines(points, num_lines):
    """Select the num_lines lines with the highest incidence count."""
    pts = np.asarray(points, dtype=np.int64)
    cand_lines = _generate_candidate_lines(pts)
    
    # Calculate incidence for each candidate line
    line_scores = []
    for A, B, C in cand_lines:
        vals = A * pts[:, 0] + B * pts[:, 1] + C
        score = int(np.sum(vals == 0))
        if score > 0:
            line_scores.append((score, (A, B, C)))
            
    # Sort by score descending
    line_scores.sort(key=lambda x: -x[0])
    
    # Select top num_lines
    selected = [line for _, line in line_scores[:num_lines]]
    
    # If we don't have enough, fill with grid lines or arbitrary lines
    if len(selected) < num_lines:
        # Add horizontal/vertical lines not already present
        existing = set(selected)
        # Try adding lines y=k or x=k
        for k in range(1, 100):
            if len(selected) >= num_lines:
                break
            if (0, 1, -k) not in existing:
                selected.append((0, 1, -k))
                existing.add((0, 1, -k))
            if (1, 0, -k) not in existing:
                selected.append((1, 0, -k))
                existing.add((1, 0, -k))
        
        # If still not enough, add arbitrary lines
        counter = 1
        while len(selected) < num_lines:
            new_line = _normalize_line(1, 1, -counter)
            if new_line not in existing:
                selected.append(new_line)
                existing.add(new_line)
            counter += 1
            
    return np.array(selected, dtype=np.int64)

def _point_line_incidence(points, lines):
    """Returns a boolean matrix of shape (num_points, num_lines) where True indicates incidence."""
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    # M[i, j] = A_j*x_i + B_j*y_i + C_j == 0
    # Broadcasting: pts (N, 1, 2), lns (1, M, 3)
    # We need A_j*x_i + B_j*y_i + C_j
    # x_i: pts[:, 0, None] (N, 1)
    # y_i: pts[:, 1, None] (N, 1)
    # A_j: lns[None, :, 0] (1, M)
    # B_j: lns[None, :, 1] (1, M)
    # C_j: lns[None, :, 2] (1, M)
    
    x = pts[:, 0][:, None] # (N, 1)
    y = pts[:, 1][:, None] # (N, 1)
    A = lns[None, :, 0]    # (1, M)
    B = lns[None, :, 1]    # (1, M)
    C = lns[None, :, 2]    # (1, M)
    
    vals = A * x + B * y + C # (N, M)
    return vals == 0

def _local_search_points(points, lines, deadline, max_radius=3):
    """
    Local search on points. 
    lines is fixed. We try to move each point to a nearby position to increase incidences.
    """
    pts = points.astype(np.int64).copy()
    lns = lines.astype(np.int64)
    n = pts.shape[0]
    m = lns.shape[0]
    
    # Precompute line coefficients for fast evaluation
    A = lns[:, 0]
    B = lns[:, 1]
    C = lns[:, 2]
    
    # Current incidence matrix
    M = _point_line_incidence(pts, lns)
    current_total = int(np.sum(M))
    
    # Occupied points set for fast lookup
    occ = set(map(tuple, pts.tolist()))
    
    start = time.time()
    improved = True
    no_improve_count = 0
    
    while improved and (time.time() - start) < deadline - 10:
        improved = False
        for i in range(n):
            if (time.time() - start) > deadline - 10:
                break
            x, y = int(pts[i, 0]), int(pts[i, 1])
            
            # Current incidences for point i
            current_point_incs = int(np.sum(M[i, :]))
            
            best_delta = 0
            best_pos = (x, y)
            
            # Search in a square neighborhood
            for dx in range(-max_radius, max_radius + 1):
                for dy in range(-max_radius, max_radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    new_pos = (nx, ny)
                    
                    # Check if occupied by another point
                    if new_pos in occ and new_pos != (x, y):
                        continue
                    
                    # Calculate new incidences for this point
                    # vals = A*nx + B*ny + C
                    vals = A * nx + B * ny + C
                    new_point_incs = int(np.sum(vals == 0))
                    
                    delta = new_point_incs - current_point_incs
                    if delta > best_delta:
                        best_delta = delta
                        best_pos = new_pos
            
            if best_delta > 0:
                # Move the point
                occ.discard((x, y))
                occ.add(best_pos)
                pts[i, 0] = best_pos[0]
                pts[i, 1] = best_pos[1]
                
                # Update M for this point
                vals = A * best_pos[0] + B * best_pos[1] + C
                M[i, :] = (vals == 0)
                
                current_total += best_delta
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
                
    return pts, current_total

def _validate(points, lines):
    """Return (points, lines) cleaned to satisfy all validity constraints."""
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)

    # Dedup points
    seen = set()
    good = []
    for p in pts.tolist():
        t = (int(p[0]), int(p[1]))
        if t not in seen:
            seen.add(t)
            good.append(t)
    pts = np.array(good, dtype=np.int64)
    if pts.shape[0] < 54:
        # Regenerate grid
        pts = _build_grid_points(6, 9)

    # Dedup lines
    seen = set()
    good = []
    for l in lns.tolist():
        norm = _normalize_line(int(l[0]), int(l[1]), int(l[2]))
        if norm is None:
            continue
        if norm not in seen:
            seen.add(norm)
            good.append(norm)
    lns = np.array(good, dtype=np.int64)
    if lns.shape[0] < 27:
        lns = _select_best_lines(pts, 27)

    return pts, lns

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin vs 630s kill

    # 1. Initialize with 6x9 grid
    points = _build_grid_points(6, 9)
    
    # 2. Select best lines for this grid
    lines = _select_best_lines(points, 27)
    
    # 3. Calculate initial score
    current_total = _count_incidences(points, lines)
    
    # 4. Local search loop
    # Alternate between moving points and re-selecting lines
    # Since re-selecting lines is O(N^2 * M) which is ~ 54^2 * 27 ~ 78k ops, it's fast.
    # Moving points is O(N * R^2 * M) which is ~ 54 * 49 * 27 ~ 72k ops.
    
    last_total = current_total
    no_improve_rounds = 0
    
    while (time.time() - start_time) < deadline - 20:
        # Step A: Optimize points given current lines
        points, current_total = _local_search_points(points, lines, deadline, max_radius=3)
        
        if current_total > last_total:
            last_total = current_total
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1
            if no_improve_rounds > 5:
                break
                
        # Step B: Re-select lines given current points
        if (time.time() - start_time) < deadline - 20:
            new_lines = _select_best_lines(points, 27)
            new_total = _count_incidences(points, new_lines)
            
            if new_total >= current_total:
                lines = new_lines
                current_total = new_total
            else:
                # If re-selecting lines didn't help (shouldn't happen if greedy is good), keep old lines
                pass

    # 5. Final validation
    points, lines = _validate(points, lines)
    
    return points, lines

# EVOLVE-BLOCK-END
