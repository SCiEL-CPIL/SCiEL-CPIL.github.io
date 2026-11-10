import numpy as np
import time
import random
from math import gcd

def normalize_line(A, B, C):
    """Normalize line coefficients to ensure uniqueness and integer validity."""
    if A == 0 and B == 0:
        return None
    
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        g = 1
    A //= g
    B //= g
    C //= g
    
    # Make first non-zero entry positive
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def count_incidences(points, lines):
    """
    points: (n, 2) array
    lines: (m, 3) array (A, B, C)
    Returns number of (p, l) such that A*x + B*y + C == 0
    """
    if len(points) == 0 or len(lines) == 0:
        return 0
    P = points.astype(np.int64)
    L = lines.astype(np.int64)
    
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    
    A = L[:, 0].reshape(1, -1)
    B = L[:, 1].reshape(1, -1)
    C = L[:, 2].reshape(1, -1)
    
    vals = A * px + B * py + C
    count = np.sum(vals == 0)
    return int(count)

def get_best_points_for_lines(lines, candidate_points, k=54):
    """
    Given a set of lines and a pool of candidate points,
    find the top k points that maximize incidences.
    Returns (best_points, max_incidence_count)
    """
    if len(candidate_points) == 0:
        return np.zeros((k, 2), dtype=np.int64), 0
    
    L = lines
    px = candidate_points[:, 0].reshape(-1, 1)
    py = candidate_points[:, 1].reshape(-1, 1)
    
    A = L[:, 0].reshape(1, -1)
    B = L[:, 1].reshape(1, -1)
    C = L[:, 2].reshape(1, -1)
    
    vals = A * px + B * py + C
    counts = np.sum(vals == 0, axis=1)
    
    # Get top k
    if len(counts) >= k:
        top_indices = np.argsort(counts)[::-1][:k]
    else:
        top_indices = np.arange(len(counts))
        
    best_points = candidate_points[top_indices]
    max_inc = int(counts[top_indices].max()) if len(counts) > 0 else 0
    
    return best_points, max_inc

def construct_grid_configuration():
    """
    Construct a configuration based on a 9x9 grid.
    Lines: 9 vertical, 9 horizontal, 9 diagonals.
    Points: Select 54 from the 81 grid points to maximize incidences.
    """
    # Grid indices 0..8
    grid_size = 9
    xs = np.arange(0, grid_size)
    ys = np.arange(0, grid_size)
    
    # Generate all 81 grid points
    XX, YY = np.meshgrid(xs, ys)
    all_points = np.column_stack([XX.ravel(), YY.ravel()])
    
    # Generate 27 lines
    lines = []
    
    # 9 Vertical lines: x = i
    for i in range(grid_size):
        lines.append((1, 0, -i))
        
    # 9 Horizontal lines: y = j
    for j in range(grid_size):
        lines.append((0, 1, -j))
        
    # 9 Diagonals: y = x + c and y = -x + c
    # We need 9 more lines.
    # Main diagonal y=x (c=0) passes through 9 points.
    # y = x + 1 (c=1) passes through 8 points.
    # y = x - 1 (c=-1) passes through 8 points.
    # y = x + 2 (c=2) passes through 7 points.
    # y = x - 2 (c=-2) passes through 7 points.
    # y = x + 3 (c=3) passes through 6 points.
    # y = x - 3 (c=-3) passes through 6 points.
    # y = -x + 8 (c=8) passes through 9 points (anti-diagonal).
    # y = -x + 7 (c=7) passes through 8 points.
    
    # Let's pick the ones with highest incidence counts on the full 9x9 grid.
    # Vertical/Horizontal each have 9 points.
    # Diagonals:
    # y = x + c: number of points is 9 - |c| for c in [-8, 8]
    # y = -x + c: number of points is 9 - |c-8| for c in [0, 16]
    
    diag_candidates = []
    for c in range(-8, 9):
        # Line: y - x - c = 0 => -x + y - c = 0 => A=-1, B=1, C=-c
        # Normalize: x - y + c = 0 => A=1, B=-1, C=c
        nl = normalize_line(1, -1, c)
        diag_candidates.append((nl, 9 - abs(c)))
        
    for c in range(0, 17):
        # Line: y + x - c = 0 => A=1, B=1, C=-c
        nl = normalize_line(1, 1, -c)
        # Incidence count on 9x9 grid:
        # x+y=c. x in [0,8], y in [0,8].
        # If c <= 8: x can be 0..c, so c+1 points.
        # If c > 8: x can be c-8..8, so 16-c points.
        inc = c + 1 if c <= 8 else 16 - c
        diag_candidates.append((nl, inc))
        
    # Sort by incidence count descending, then by line tuple for determinism
    diag_candidates.sort(key=lambda x: (-x[1], x[0]))
    
    # Take top 9 unique lines
    selected_diags = []
    seen = set()
    for nl, inc in diag_candidates:
        if nl not in seen:
            selected_diags.append(nl)
            seen.add(nl)
            if len(selected_diags) == 9:
                break
                
    lines.extend(selected_diags)
    
    lines_arr = np.array(lines, dtype=np.int64)
    
    # Select best 54 points from the 81 grid points
    best_points, _ = get_best_points_for_lines(lines_arr, all_points, k=54)
    
    return best_points, lines_arr

def polish_points(points, lines, candidate_points, time_limit):
    """
    Local search to improve point selection.
    Try swapping a point in the set with a point in the candidate pool.
    """
    current_points = points.copy()
    current_I = count_incidences(current_points, lines)
    best_points = current_points.copy()
    best_I = current_I
    
    start_time = time.time()
    no_improve = 0
    
    # Precompute incidence counts for all candidates against current lines
    # This is expensive, so we do it once initially, and update incrementally if needed.
    # But lines are fixed here.
    
    # To speed up, we can precompute which candidate point is on which line.
    # However, 81 candidates x 27 lines is small.
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try swapping each point in current set with each candidate not in set
        # This is O(54 * 81 * 27) ~ 117k ops per pass. Very fast.
        
        for i in range(54):
            p_old = current_points[i]
            for j in range(len(candidate_points)):
                p_new = candidate_points[j]
                
                # Check if p_new is already in current_points
                # Simple check: is p_new equal to any p in current_points?
                # We can use a set of tuples for faster lookup
                # But 54 points is small.
                is_dup = False
                for k in range(54):
                    if np.array_equal(current_points[k], p_new):
                        if k != i: # If it's the same index, it's the same point, no change
                            is_dup = True
                            break
                        else:
                            # Same point, no change in I
                            break
                if is_dup:
                    continue
                
                # Calculate delta I
                # I = sum over points of (number of lines containing the point)
                # Let score(p) = number of lines containing p
                # Then I = sum(score(p_i) for p_i in points)
                # Delta = score(p_new) - score(p_old)
                
                # We can precompute scores for all candidates
                # But p_old is in candidates, p_new is in candidates.
                
                # Let's just compute the scores directly if not cached.
                # For efficiency, let's precompute a score array for all candidates.
                
                pass # Placeholder for logic below
                
        # Implementing the efficient version:
        # Precompute scores for all candidate points
        # score[j] = number of lines in lines that contain candidate_points[j]
        
        # We'll do this outside the loop if possible, but here we are in a polish step.
        # Since lines are fixed, we can compute scores once.
        
        break # The loop above was a sketch. Let's implement the efficient version properly.
    
    # Efficient Polish
    # Precompute scores for all candidates
    scores = np.zeros(len(candidate_points), dtype=np.int64)
    px = candidate_points[:, 0].reshape(-1, 1)
    py = candidate_points[:, 1].reshape(-1, 1)
    A = lines[:, 0].reshape(1, -1)
    B = lines[:, 1].reshape(1, -1)
    C = lines[:, 2].reshape(1, -1)
    vals = A * px + B * py + C
    scores = np.sum(vals == 0, axis=1)
    
    # Map point tuple to index in candidate_points for O(1) lookup?
    # Or just use a set of tuples.
    cand_tuples = [tuple(p) for p in candidate_points]
    tuple_to_idx = {t: i for i, t in enumerate(cand_tuples)}
    
    current_point_tuples = [tuple(p) for p in current_points]
    current_scores = np.array([tuple_to_idx[t] for t in current_point_tuples])
    current_scores = scores[current_scores]
    
    while time.time() - start_time < time_limit:
        improved = False
        # Find the point in current set with lowest score
        min_idx = np.argmin(current_scores)
        min_score = current_scores[min_idx]
        
        # Find the candidate with highest score that is not in the set
        # Mask out candidates that are in the set
        in_set_mask = np.zeros(len(candidate_points), dtype=bool)
        for t in current_point_tuples:
            if t in tuple_to_idx:
                in_set_mask[tuple_to_idx[t]] = True
        
        # Find max score among not-in-set
        masked_scores = scores.copy()
        masked_scores[in_set_mask] = -1
        
        max_cand_idx = np.argmax(masked_scores)
        max_cand_score = scores[max_cand_idx]
        
        if max_cand_score > min_score:
            # Swap
            old_point = current_points[min_idx]
            new_point = candidate_points[max_cand_idx]
            
            current_points[min_idx] = new_point
            current_point_tuples[min_idx] = tuple(new_point)
            current_scores[min_idx] = max_cand_score
            
            # Update I
            current_I += max_cand_score - min_score
            
            if current_I > best_I:
                best_I = current_I
                best_points = current_points.copy()
            
            improved = True
        
        if not improved:
            break
            
    return best_points, best_I

def szemeredi_trotter_incidences():
    deadline = time.time() + 590  # Safety margin
    
    # 1. Construct the grid-based configuration
    points, lines = construct_grid_configuration()
    
    # 2. Polish the points using local search
    # The candidate pool is the 81 grid points
    # We need to extract the grid points from the construction or regenerate them
    grid_size = 9
    xs = np.arange(0, grid_size)
    ys = np.arange(0, grid_size)
    XX, YY = np.meshgrid(xs, ys)
    candidate_points = np.column_stack([XX.ravel(), YY.ravel()])
    
    # Run polish
    polished_points, polished_I = polish_points(points, lines, candidate_points, time_limit=10)
    
    # 3. Try a few variations of the diagonal selection to see if we can do better
    # The initial construction picked the "best" 9 diagonals.
    # Let's try swapping one diagonal for another and re-polishing.
    
    best_points = polished_points
    best_I = polished_I
    
    # Generate all possible diagonal lines
    all_diags = []
    for c in range(-8, 9):
        nl = normalize_line(1, -1, c)
        if nl:
            all_diags.append(nl)
    for c in range(0, 17):
        nl = normalize_line(1, 1, -c)
        if nl:
            all_diags.append(nl)
            
    # Remove duplicates
    unique_diags = list(set(all_diags))
    
    # Current lines: 9 vert, 9 horiz, 9 diags
    # Let's identify which are the current diags
    current_diag_set = set()
    for l in lines:
        A, B, C = l
        if A == 1 and B == -1:
            current_diag_set.add((A, B, C))
        elif A == 1 and B == 1:
            current_diag_set.add((A, B, C))
            
    # Try swapping each current diag with each other diag
    for i in range(9):
        if time.time() > deadline:
            break
            
        # Get the i-th diagonal from current lines
        # We need to know which index in lines array corresponds to the i-th diag
        # In construct_grid_configuration, lines are:
        # 0-8: Vertical
        # 9-17: Horizontal
        # 18-26: Diagonals
        
        old_diag = lines[18 + i]
        
        for new_diag in unique_diags:
            if tuple(new_diag) == tuple(old_diag):
                continue
            
            # Create new lines array
            new_lines = lines.copy()
            new_lines[18 + i] = new_diag
            
            # Re-polish points
            # To save time, we can just re-select best 54 points greedily
            new_points, _ = get_best_points_for_lines(new_lines, candidate_points, k=54)
            
            # Quick polish
            new_points, new_I = polish_points(new_points, new_lines, candidate_points, time_limit=2)
            
            if new_I > best_I:
                best_I = new_I
                best_points = new_points
                # Update lines to the new best lines
                lines = new_lines
                
    # Final validation
    # Ensure 54 unique points
    unique_points, indices = np.unique(best_points, axis=0, return_index=True)
    if len(unique_points) < 54:
        # Fallback to simple grid
        points, lines = construct_grid_configuration()
        best_points, best_I = polish_points(points, lines, candidate_points, time_limit=5)
        unique_points, indices = np.unique(best_points, axis=0, return_index=True)
        best_points = unique_points[indices]
    else:
        best_points = unique_points[indices]
        
    # Ensure 27 unique lines
    unique_lines = []
    for line in lines:
        n_line = normalize_line(line[0], line[1], line[2])
        if n_line and n_line not in unique_lines:
            unique_lines.append(n_line)
            
    if len(unique_lines) < 27:
        # Fallback
        points, lines = construct_grid_configuration()
        best_points, best_I = polish_points(points, lines, candidate_points, time_limit=5)
        unique_points, indices = np.unique(best_points, axis=0, return_index=True)
        best_points = unique_points[indices]
        unique_lines = []
        for line in lines:
            n_line = normalize_line(line[0], line[1], line[2])
            if n_line and n_line not in unique_lines:
                unique_lines.append(n_line)
                
    best_lines = np.array(unique_lines[:27], dtype=np.int64)
    
    return best_points, best_lines
