# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from math import gcd

def _normalize_line(A, B, C):
    """Normalize a line so it's unique: divide by gcd, make first nonzero positive."""
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    # Make first nonzero entry positive
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidences(points, lines):
    """Count exact integer incidences."""
    I = 0
    for p in points:
        x, y = p[0], p[1]
        for line in lines:
            A, B, C = line[0], line[1], line[2]
            if A * x + B * y + C == 0:
                I += 1
    return I

def _generate_candidate_lines(points):
    """Generate all lines through pairs of points, plus some extra."""
    lines = set()
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            norm = _normalize_line(A, B, C)
            if norm != (0, 0, 0):
                lines.add(norm)
    return list(lines)

def _select_best_lines(points, num_lines, all_candidate_lines, time_budget=300.0):
    """Select best lines using greedy + local search."""
    start_time = time.time()
    
    # Greedy: pick line with most incidences, remove those points, repeat
    # But we want to maximize total, so use a different approach
    
    # For each candidate line, count how many points it hits
    line_point_counts = {}
    for line in all_candidate_lines:
        A, B, C = line
        count = 0
        for p in points:
            if A * p[0] + B * p[1] + C == 0:
                count += 1
        line_point_counts[line] = count
    
    # Sort by count descending
    sorted_lines = sorted(all_candidate_lines, key=lambda l: line_point_counts[l], reverse=True)
    
    # Greedy selection with diversity
    selected = []
    selected_set = set()
    remaining_points = set(range(len(points)))
    
    for line in sorted_lines:
        if len(selected) >= num_lines:
            break
        if line in selected_set:
            continue
        A, B, C = line
        # Count how many remaining points this line hits
        hits = 0
        for idx in remaining_points:
            if A * points[idx][0] + B * points[idx][1] + C == 0:
                hits += 1
        if hits > 0:
            selected.append(line)
            selected_set.add(line)
            # Remove hit points from remaining (soft - they can still be hit by other lines)
            # Actually for maximizing total incidences, we should NOT remove points
            # because a point can be on multiple lines. So just select top lines.
    
    # Better approach: just pick the top num_lines by point count
    selected = sorted_lines[:num_lines]
    
    # Local search: try swapping one line for another
    def total_incidence(line_list):
        total = 0
        for line in line_list:
            A, B, C = line
            for p in points:
                if A * p[0] + B * p[1] + C == 0:
                    total += 1
        return total
    
    current_best = selected[:]
    current_score = total_incidence(current_best)
    current_set = set(current_best)
    
    improved = True
    while improved:
        if time.time() - start_time > time_budget:
            break
        improved = False
        for i in range(len(current_best)):
            if time.time() - start_time > time_budget:
                break
            for j in range(len(all_candidate_lines)):
                if all_candidate_lines[j] in current_set:
                    continue
                new_line = all_candidate_lines[j]
                old_line = current_best[i]
                
                # Compute delta
                old_A, old_B, old_C = old_line
                new_A, new_B, new_C = new_line
                
                old_count = 0
                new_count = 0
                for p in points:
                    if old_A * p[0] + old_B * p[1] + old_C == 0:
                        old_count += 1
                    if new_A * p[0] + new_B * p[1] + new_C == 0:
                        new_count += 1
                
                if new_count > old_count:
                    current_best[i] = new_line
                    current_set.remove(old_line)
                    current_set.add(new_line)
                    current_score += new_count - old_count
                    improved = True
                    break
    
    return current_best

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization.
    Searches over grid dimensions and selects optimal lines.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit
    
    best_points = None
    best_lines = None
    best_score = 0
    
    # Try different grid dimensions that multiply to 54
    # 54 = 1*54, 2*27, 3*18, 6*9, 9*6, 18*3, 27*2, 54*1
    grid_dims = [(6, 9), (9, 6), (3, 18), (18, 3), (2, 27), (27, 2), (1, 54), (54, 1)]
    
    for (nx, ny) in grid_dims:
        if time.time() > deadline:
            break
        
        # Generate grid points
        xs = np.arange(1, nx + 1)
        ys = np.arange(1, ny + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        if points.shape != (54, 2):
            continue
        
        # Generate all candidate lines
        all_lines = _generate_candidate_lines(points)
        
        if len(all_lines) == 0:
            continue
        
        # Select best 27 lines
        selected = _select_best_lines(points, 27, all_lines, time_budget=min(60, deadline - time.time()))
        
        if len(selected) < 27:
            continue
        
        # Check for duplicates
        seen = set()
        unique_selected = []
        for line in selected:
            if line not in seen:
                seen.add(line)
                unique_selected.append(line)
        
        if len(unique_selected) < 27:
            # Pad with any other lines
            for line in all_lines:
                if len(unique_selected) >= 27:
                    break
                if line not in seen:
                    unique_selected.append(line)
                    seen.add(line)
        
        if len(unique_selected) < 27:
            continue
        
        lines = np.array(unique_selected[:27], dtype=np.int64)
        
        # Verify validity
        if points.shape != (54, 2) or lines.shape != (27, 3):
            continue
        
        # Check no A=B=0
        valid = True
        for line in lines:
            if line[0] == 0 and line[1] == 0:
                valid = False
                break
        if not valid:
            continue
        
        # Check no duplicate lines
        norm_lines = [_normalize_line(int(l[0]), int(l[1]), int(l[2])) for l in lines]
        if len(set(norm_lines)) != 27:
            continue
        
        # Count incidences
        score = _count_incidences(points, lines)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
            best_lines = lines.copy()
    
    # Fallback to original if search didn't find anything
    if best_points is None:
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        lines_list = []
        for i in range(1, 7):
            lines_list.append((1, 0, -i))
        for j in range(1, 10):
            lines_list.append((0, 1, -j))
        for k in [0, 1, 2, 3]:
            lines_list.append((-1, 1, -k))
        for k in [-1, 4]:
            lines_list.append((-1, 1, -k))
        for k in [7, 8, 9, 10]:
            lines_list.append((1, 1, -k))
        for k in [6, 11]:
            lines_list.append((1, 1, -k))
        best_lines = np.array(lines_list, dtype=np.int64)
    
    assert best_points.shape == (54, 2)
    assert best_lines.shape == (27, 3)
    
    return best_points, best_lines

# EVOLVE-BLOCK-END
