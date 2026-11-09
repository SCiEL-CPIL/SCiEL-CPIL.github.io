# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from itertools import combinations, product


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


def _normalize_line(A, B, C):
    """Normalize a line so it's unique: divide by gcd, make first nonzero positive."""
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _is_valid_line(A, B, C):
    """Check if a line is valid (not A=B=0)."""
    return not (A == 0 and B == 0)


def _build_grid_lines(a, b):
    """
    Build a grid of a columns x b rows, with points at (i+1, j+1) for i in [0,a), j in [0,b).
    Generate candidate lines: verticals, horizontals, and all diagonals through grid points.
    Return (points, candidate_lines) where candidate_lines is a list of (A,B,C) tuples.
    """
    # Points: (i+1, j+1) for i in range(a), j in range(b)
    points = []
    for i in range(a):
        for j in range(b):
            points.append((i + 1, j + 1))
    points = np.array(points, dtype=np.int64)
    
    n_points = len(points)
    
    # Candidate lines:
    # 1. Vertical lines: x = c, for c in 1..a -> (1, 0, -c)
    # 2. Horizontal lines: y = c, for c in 1..b -> (0, 1, -c)
    # 3. Diagonal lines through pairs of points
    
    candidate_lines = []
    seen = set()
    
    # Vertical lines
    for c in range(1, a + 1):
        line = (1, 0, -c)
        norm = _normalize_line(*line)
        if norm not in seen:
            seen.add(norm)
            candidate_lines.append(line)
    
    # Horizontal lines
    for c in range(1, b + 1):
        line = (0, 1, -c)
        norm = _normalize_line(*line)
        if norm not in seen:
            seen.add(norm)
            candidate_lines.append(line)
    
    # Diagonal lines through pairs of points
    # For each pair of points, compute the line through them
    for i in range(n_points):
        x1, y1 = points[i]
        for j in range(i + 1, n_points):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            # Line: dy*(x - x1) - dx*(y - y1) = 0
            # dy*x - dx*y - dy*x1 + dx*y1 = 0
            A = dy
            B = -dx
            C = -dy * x1 + dx * y1
            if not _is_valid_line(A, B, C):
                continue
            norm = _normalize_line(A, B, C)
            if norm not in seen:
                seen.add(norm)
                candidate_lines.append((A, B, C))
    
    return points, candidate_lines


def _evaluate_and_select(points, candidate_lines, m_lines=27):
    """
    Given points and candidate lines, select the best m_lines lines to maximize incidences.
    Use greedy approach with local search.
    """
    n_points = len(points)
    n_cand = len(candidate_lines)
    
    # Precompute incidence matrix: inc[p][l] = 1 if point p is on line l
    # To save memory, compute on the fly or use a bitset
    # For 54 points and ~few hundred lines, this is manageable
    
    # Count incidences for each candidate line
    line_inc_counts = np.zeros(n_cand, dtype=np.int64)
    for j in range(n_cand):
        A, B, C = candidate_lines[j]
        for i in range(n_points):
            x, y = points[i]
            if A * x + B * y + C == 0:
                line_inc_counts[j] += 1
    
    # Greedy: pick the line with most incidences, then next, etc.
    # But we need to be careful: picking a line doesn't reduce incidences of other lines
    # (incidences are independent per line), so greedy by count is optimal for maximizing sum!
    
    # Wait - that's true! Each line's incidence count is independent of which other lines we pick.
    # So the optimal set of m lines is simply the m lines with the highest incidence counts.
    
    # Sort by incidence count descending
    order = np.argsort(-line_inc_counts)
    selected = order[:m_lines]
    
    # Build the selected lines array
    selected_lines = np.array([candidate_lines[j] for j in selected], dtype=np.int64)
    
    # Verify no duplicates after normalization
    seen_norm = set()
    unique_selected = []
    for j in selected:
        norm = _normalize_line(*candidate_lines[j])
        if norm not in seen_norm:
            seen_norm.add(norm)
            unique_selected.append(j)
    
    # If we have duplicates, we need to handle that - but the problem says no duplicate lines
    # Let's just take the first m unique ones
    if len(unique_selected) < m_lines:
        # Add more lines from the rest
        for j in order:
            if j not in unique_selected:
                norm = _normalize_line(*candidate_lines[j])
                if norm not in seen_norm:
                    seen_norm.add(norm)
                    unique_selected.append(j)
                    if len(unique_selected) >= m_lines:
                        break
    
    unique_selected = unique_selected[:m_lines]
    selected_lines = np.array([candidate_lines[j] for j in unique_selected], dtype=np.int64)
    
    # Count total incidences
    total = sum(line_inc_counts[j] for j in unique_selected)
    
    return selected_lines, total


def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences with 54 points and 27 lines.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second budget
    
    best_points = None
    best_lines = None
    best_score = -1
    
    # Try different grid configurations that give 54 points
    # a * b = 54
    grid_configs = []
    for a in range(1, 55):
        if 54 % a == 0:
            b = 54 // a
            grid_configs.append((a, b))
    
    # Sort by more balanced aspect ratio first
    grid_configs.sort(key=lambda ab: abs(ab[0] - ab[1]))
    
    for a, b in grid_configs:
        if time.time() > deadline - 30:
            break
        
        points, candidate_lines = _build_grid_lines(a, b)
        
        # Evaluate and select best 27 lines
        selected_lines, score = _evaluate_and_select(points, candidate_lines, 27)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
            best_lines = selected_lines.copy()
    
    # Also try non-grid configurations with local search
    # Start from the best grid configuration and try swapping points/lines
    
    if best_points is not None:
        # Try to improve by adding points that create more collinearities
        # and removing points that don't contribute much
        current_points = best_points.copy()
        current_lines = best_lines.copy()
        
        # Compute current incidence
        current_score = _count_incidences(current_points, current_lines)
        
        # Try local search: for each point, try moving it to a nearby position
        # and see if we can increase incidences with existing lines or create new opportunities
        for iteration in range(200):
            if time.time() > deadline - 60:
                break
            
            improved = False
            for pi in range(54):
                if time.time() > deadline - 60:
                    break
                
                orig_x, orig_y = current_points[pi]
                # Try moving this point to nearby positions
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if dx == 0 and dy == 0:
                            continue
                        new_x = orig_x + dx
                        new_y = orig_y + dy
                        if new_x < 1 or new_y < 1:
                            continue
                        
                        # Check if this position is already occupied
                        if any(current_points[i][0] == new_x and current_points[i][1] == new_y 
                              for i in range(54) if i != pi):
                            continue
                        
                        # Temporarily move the point
                        saved = current_points[pi].copy()
                        current_points[pi] = [new_x, new_y]
                        
                        # Recompute incidences
                        new_score = _count_incidences(current_points, current_lines)
                        
                        if new_score > current_score:
                            current_score = new_score
                            improved = True
                        else:
                            current_points[pi] = saved
            
            if not improved:
                break
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
            best_lines = current_lines.copy()
    
    # Final: also try to optimize the lines given the best points
    if best_points is not None and time.time() < deadline - 30:
        # Generate more candidate lines from the current points
        n_points = len(best_points)
        candidate_lines = []
        seen = set()
        
        # All lines through pairs of points
        for i in range(n_points):
            x1, y1 = best_points[i]
            for j in range(i + 1, n_points):
                x2, y2 = best_points[j]
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    continue
                A = dy
                B = -dx
                C = -dy * x1 + dx * y1
                if not _is_valid_line(A, B, C):
                    continue
                norm = _normalize_line(A, B, C)
                if norm not in seen:
                    seen.add(norm)
                    candidate_lines.append((A, B, C))
        
        # Select best 27
        line_inc_counts = np.zeros(len(candidate_lines), dtype=np.int64)
        for j in range(len(candidate_lines)):
            A, B, C = candidate_lines[j]
            for i in range(n_points):
                x, y = best_points[i]
                if A * x + B * y + C == 0:
                    line_inc_counts[j] += 1
        
        order = np.argsort(-line_inc_counts)
        selected = []
        seen_norm = set()
        for j in order:
            norm = _normalize_line(*candidate_lines[j])
            if norm not in seen_norm:
                seen_norm.add(norm)
                selected.append(j)
            if len(selected) >= 27:
                break
        
        selected_lines = np.array([candidate_lines[j] for j in selected], dtype=np.int64)
        final_score = sum(line_inc_counts[j] for j in selected)
        
        if final_score > best_score:
            best_score = final_score
            best_lines = selected_lines
    
    # Ensure we have valid output
    if best_points is None:
        # Fallback to original
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
    
    # Ensure shapes are correct
    assert best_points.shape == (54, 2)
    assert best_lines.shape == (27, 3)
    
    return best_points, best_lines


# EVOLVE-BLOCK-END
