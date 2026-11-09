import numpy as np
import time
from math import gcd
from collections import defaultdict

def _normalize_line(A, B, C):
    """Normalize line coefficients (A,B,C) so that gcd(|A|,|B|,|C|)=1 and first nonzero is positive."""
    if A == 0 and B == 0:
        return None
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return None
    A //= g
    B //= g
    C //= g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _generate_grid(a, b):
    xs = np.arange(1, a + 1)
    ys = np.arange(1, b + 1)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def _compute_all_lines_and_counts(points):
    """
    Computes all unique lines defined by pairs of points and their incidence counts.
    Returns a dict {line_key: count}
    """
    n = points.shape[0]
    lines_dict = defaultdict(int)
    
    # Map point index to set of lines it's on? 
    # For now, just count incidences for each line.
    
    # Generate all pairs
    for i in range(n):
        x1, y1 = points[i, 0], points[i, 1]
        for j in range(i + 1, n):
            x2, y2 = points[j, 0], points[j, 1]
            dx = x2 - x1
            dy = y2 - y1
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            key = _normalize_line(A, B, C)
            if key is not None:
                lines_dict[key] = 0 # Initialize
                
    # Count incidences
    # To be fast, we can iterate lines and check all points, or iterate points and lines.
    # Since n=54, M~1400, M*N = 75k ops. Fast.
    for key in lines_dict:
        A, B, C = key
        vals = A * points[:, 0] + B * points[:, 1] + C
        lines_dict[key] = int(np.sum(vals == 0))
        
    return lines_dict

def _get_top_k_sum(lines_dict, k=27):
    if not lines_dict:
        return 0
    counts = list(lines_dict.values())
    counts.sort(reverse=True)
    return sum(counts[:k])

def _evaluate_configuration(points):
    """
    Returns (score, top_lines_array)
    """
    lines_dict = _compute_all_lines_and_counts(points)
    sorted_lines = sorted(lines_dict.items(), key=lambda x: (-x[1], x[0]))
    selected = [l for l, c in sorted_lines[:27]]
    score = sum(c for _, c in sorted_lines[:27])
    
    # Pad if necessary
    if len(selected) < 27:
        # This shouldn't happen for 54 points
        for k in range(1000000, 1000000 + (27 - len(selected))):
            selected.append((1, 0, k))
            
    return score, np.array(selected, dtype=np.int64)

def _local_search(points, deadline):
    """
    Perform local search by moving single points to nearby positions.
    """
    current_points = points.copy()
    current_score, _ = _evaluate_configuration(current_points)
    
    # Pre-compute lines and counts for fast updates?
    # For simplicity and safety, we re-evaluate fully, but only for promising moves.
    # However, full re-eval is ~0.01s. 54*24 = 1296 moves/pass. 12s/pass.
    # We have ~500s. So ~40 passes. This is feasible.
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        n = current_points.shape[0]
        
        # Try moving each point
        for i in range(n):
            if time.time() > deadline - 10:
                break
            orig_pt = current_points[i]
            
            # Candidate positions: neighbors in a 5x5 box around orig_pt
            # Exclude orig_pt itself
            candidates = []
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    new_pt = np.array([orig_pt[0] + dx, orig_pt[1] + dy])
                    # Basic sanity: keep coords positive? Not strictly required, but good for grid-like structures.
                    # The problem doesn't restrict coords to positive, but integers.
                    # However, large coords might cause issues with line normalization? No, int64.
                    # But "nearby" is a heuristic.
                    candidates.append(new_pt)
            
            # Check for duplicates
            existing_pts = set(map(tuple, current_points))
            
            for new_pt in candidates:
                if tuple(new_pt) in existing_pts:
                    continue
                
                # Apply move
                test_points = current_points.copy()
                test_points[i] = new_pt
                
                # Evaluate
                new_score, _ = _evaluate_configuration(test_points)
                
                if new_score > current_score:
                    current_points = test_points
                    current_score = new_score
                    improved = True
                    break # Restart search from this point? Or continue?
                    # Breaking out of candidate loop for this i is good.
                    # We can break out of i loop too to restart with new config.
            
            if improved:
                break # Restart outer loop
        
    return current_points, current_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580 # Safety margin
    
    best_score = -1
    best_points = None
    best_lines = None
    
    def update_best(pts, score):
        nonlocal best_score, best_points, best_lines
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            _, lines = _evaluate_configuration(pts)
            best_lines = lines
            
    # 1. Try standard grids
    grid_candidates = [
        (6, 9), (9, 6), (3, 18), (18, 3), (2, 27), (27, 2),
        (1, 54), (54, 1), (7, 8), (8, 7), (5, 11), (11, 5)
    ]
    
    for a, b in grid_candidates:
        if time.time() > deadline - 120:
            break
        pts = _generate_grid(a, b)
        score, _ = _evaluate_configuration(pts)
        update_best(pts, score)
        
    # 2. Local search on the best grid found so far
    if best_points is not None:
        if time.time() < deadline - 60:
            improved_pts, improved_score = _local_search(best_points, deadline)
            update_best(improved_pts, improved_score)
            
    # 3. If time remains, try local search on other promising grids
    if time.time() < deadline - 30:
        # Try 7x8 and 8x7 with removals?
        # The local search might naturally handle "removals" if it moves points to infinity?
        # No, we must keep 54 points.
        # So we stick to the best grid found.
        
        # Maybe try a different starting point for local search?
        # E.g. 7x8 grid (56 pts) -> remove 2 worst pts -> local search?
        # Let's try 7x8, remove 2 corner points, then local search.
        for a, b in [(7, 8), (8, 7)]:
            if time.time() > deadline - 30:
                break
            base_pts = _generate_grid(a, b)
            # Remove 2 points that are on fewest lines?
            # Just remove corners for simplicity
            # Corners are (1,1), (1,b), (a,1), (a,b)
            # Indices: 0, b-1, a*b-a, a*b-1
            pts = base_pts.copy()
            # Remove last two points (corners)
            pts = pts[:-2]
            if len(pts) != 54:
                continue
            score, _ = _evaluate_configuration(pts)
            if score > best_score:
                update_best(pts, score)
                # Local search on this
                if time.time() < deadline - 30:
                    improved_pts, improved_score = _local_search(pts, deadline)
                    update_best(improved_pts, improved_score)

    # Fallback
    if best_points is None:
        pts = _generate_grid(6, 9)
        score, lines = _evaluate_configuration(pts)
        best_points = pts
        best_lines = lines
        best_score = score
        
    return best_points, best_lines

