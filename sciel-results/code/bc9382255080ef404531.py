import numpy as np
import time
import itertools
from math import gcd

def _normalize_line(line):
    A, B, C = line
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g > 0:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _generate_candidate_lines(max_coeff=10):
    lines = set()
    # Verticals
    for x in range(-max_coeff, max_coeff + 1):
        lines.add(_normalize_line((1, 0, -x)))
    # Horizontals
    for y in range(-max_coeff, max_coeff + 1):
        lines.add(_normalize_line((0, 1, -y)))
    # Slopes
    for A in range(-max_coeff, max_coeff + 1):
        for B in range(-max_coeff, max_coeff + 1):
            if A == 0 and B == 0:
                continue
            for C in range(-max_coeff * 2, max_coeff * 2 + 1):
                lines.add(_normalize_line((A, B, C)))
    return list(lines)

def _calculate_incidence(points, lines):
    # points: (N, 2)
    # lines: (M, 3)
    # Efficient check: A*x + B*y + C == 0
    if len(points) == 0 or len(lines) == 0:
        return 0
    
    # Vectorized check
    # P: (N, 2), L: (M, 3)
    # Expand to (N, M)
    # A*x + B*y + C
    # x: (N, 1), y: (N, 1)
    # A: (1, M), B: (1, M), C: (1, M)
    
    x = points[:, 0:1]
    y = points[:, 1:2]
    A = lines[:, 0:1].T
    B = lines[:, 1:2].T
    C = lines[:, 2:3].T
    
    val = A * x + B * y + C
    count = np.sum(val == 0)
    return count

def _evaluate_configuration(points, lines):
    return _calculate_incidence(points, lines)

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    # Initial Configuration: 7x8 grid minus 2 points
    xs = np.arange(1, 8)
    ys = np.arange(1, 9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    all_points_grid = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Remove (4, 1) and (4, 8)
    idx1 = (4-1)*8 + (1-1) # 24
    idx2 = (4-1)*8 + (8-1) # 31
    mask = np.ones(56, dtype=bool)
    mask[idx1] = False
    mask[idx2] = False
    current_points = all_points_grid[mask]
    
    current_lines = [
        (1, 0, -1), (1, 0, -2), (1, 0, -3), (1, 0, -4), (1, 0, -5), (1, 0, -6), (1, 0, -7),
        (0, 1, -1), (0, 1, -2), (0, 1, -3), (0, 1, -4), (0, 1, -5), (0, 1, -6), (0, 1, -7), (0, 1, -8),
        (1, -1, 0), (1, -1, 1),
        (1, -1, -1), (1, -1, 2),
        (1, -1, -2), (1, -1, 3),
        (1, 1, -8), (1, 1, -9),
        (1, 1, -7), (1, 1, -10),
        (1, 1, -6), (1, 1, -11)
    ]
    current_lines = np.array(current_lines, dtype=np.int64)
    
    current_score = _evaluate_configuration(current_points, current_lines)
    best_points = current_points.copy()
    best_lines = current_lines.copy()
    best_score = current_score
    
    # Generate candidate lines for swapping
    candidate_lines = _generate_candidate_lines(max_coeff=10)
    candidate_lines = np.array(list(set(tuple(l) for l in candidate_lines)), dtype=np.int64)
    
    # Local Search
    # 1. Try swapping lines
    # 2. Try swapping points (removing one, adding another from grid)
    
    num_points = 54
    num_lines = 27
    
    # Pre-compute point indices for easy access
    point_to_idx = {tuple(p): i for i, p in enumerate(all_points_grid)}
    
    # We will iterate a few times
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Line Swapping
        # For each line in current_lines, try replacing it with a candidate line
        # To speed up, only try candidates that are "close" or likely to be better?
        # Just try all candidates for the worst-performing lines first?
        # Or just iterate through all pairs. 27 * ~500 candidates = 13500 checks.
        # Each check is O(N*M) = 54*27 = 1458 ops. Total ~20M ops. Fast in numpy.
        
        for i in range(num_lines):
            if time.time() > deadline:
                break
            line_to_replace = current_lines[i]
            
            # Calculate incidence of current line
            # Actually, we can just calculate the new score if we swap.
            # Score_new = Score_old - Incidence(line_to_replace, Points) + Incidence(candidate_line, Points)
            # But Incidence(line_to_replace) might overlap with other lines? No, incidence is sum of (point, line) pairs.
            # So it is additive.
            
            inc_old = np.sum((current_points[:, 0] * line_to_replace[0] + current_points[:, 1] * line_to_replace[1] + line_to_replace[2]) == 0)
            
            for j in range(len(candidate_lines)):
                if time.time() > deadline:
                    break
                cand = candidate_lines[j]
                # Skip if same line
                if np.array_equal(cand, line_to_replace):
                    continue
                # Skip if cand is already in current_lines (to avoid duplicates in the set of lines)
                # This check is expensive. We can skip it for now and verify at the end, 
                # or use a set of normalized lines.
                
                inc_new = np.sum((current_points[:, 0] * cand[0] + current_points[:, 1] * cand[1] + cand[2]) == 0)
                
                if inc_new > inc_old:
                    # Potential improvement
                    new_lines = current_lines.copy()
                    new_lines[i] = cand
                    
                    # Check for duplicate lines
                    unique_lines = set()
                    valid = True
                    for l in new_lines:
                        norm = _normalize_line(l)
                        if norm in unique_lines:
                            valid = False
                            break
                        unique_lines.add(norm)
                    
                    if valid:
                        new_score = current_score - inc_old + inc_new
                        if new_score > best_score:
                            best_score = new_score
                            best_lines = new_lines
                            current_lines = new_lines
                            current_score = new_score
                            improved = True
                            # Continue searching with the new best line set?
                            # Yes, update current_lines to the improved one.
                            break # Restart line swap loop for this i? Or just continue?
        
        # Point Swapping
        # Remove a point, add a different point from the grid
        # This is more complex because it affects all lines.
        # We can try removing each of the 54 points and adding each of the 2 missing points.
        # 54 * 2 = 108 swaps.
        # For each swap, we need to recalculate the score for all 27 lines.
        # Cost: 108 * 27 * 54 ops. ~150k ops. Very fast.
        
        missing_points = []
        for p in all_points_grid:
            if not np.any(np.all(p == current_points, axis=1)):
                missing_points.append(p)
        
        for i in range(num_points):
            if time.time() > deadline:
                break
            p_remove = current_points[i]
            
            for p_add in missing_points:
                if np.array_equal(p_remove, p_add):
                    continue
                
                new_points = current_points.copy()
                new_points[i] = p_add
                
                # Check for duplicate points
                # Since we are swapping with a missing point, and p_remove was unique, 
                # p_add is not in the set, so new_points should be unique.
                
                new_score = _evaluate_configuration(new_points, current_lines)
                
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points
                    current_points = new_points
                    current_score = new_score
                    # Update missing points list
                    missing_points = []
                    for p in all_points_grid:
                        if not np.any(np.all(p == current_points, axis=1)):
                            missing_points.append(p)
                    improved = True
                    break # Restart point swap loop?
            
            if improved:
                break # Restart outer loop
        
    return best_points, best_lines
