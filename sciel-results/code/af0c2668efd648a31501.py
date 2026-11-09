# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd
from itertools import combinations

def _normalize_line(A, B, C):
    """Normalize a line equation Ax + By + C = 0 to canonical form."""
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
    """Generate an a x b grid of points."""
    xs = np.arange(1, a + 1)
    ys = np.arange(1, b + 1)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def _build_universe(grid_points):
    """
    Build a universe of lines from grid_points.
    Returns:
        lines: np.array of (A, B, C) for each unique line
        point_line_map: list of lists, where point_line_map[i] contains indices of lines passing through grid_points[i]
        point_indices: mapping from (x,y) to index in grid_points
    """
    n = len(grid_points)
    line_to_idx = {}
    
    # Generate all lines
    for i in range(n):
        for j in range(i + 1, n):
            p1 = grid_points[i]
            p2 = grid_points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            A = dy
            B = -dx
            C = dx * p1[1] - dy * p1[0]
            key = _normalize_line(A, B, C)
            if key is not None:
                if key not in line_to_idx:
                    line_to_idx[key] = len(line_to_idx)
                    
    # Convert to array
    items = list(line_to_idx.items())
    lines = np.array([k for k, v in items], dtype=np.int64).reshape(-1, 3)
    
    # Build point_line_map
    point_line_map = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            p1 = grid_points[i]
            p2 = grid_points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            A = dy
            B = -dx
            C = dx * p1[1] - dy * p1[0]
            key = _normalize_line(A, B, C)
            if key is not None:
                idx = line_to_idx[key]
                point_line_map[i].append(idx)
                point_line_map[j].append(idx)
                
    # Create point to index map
    point_indices = {}
    for i, p in enumerate(grid_points):
        point_indices[(p[0], p[1])] = i
        
    return lines, point_line_map, point_indices

def _local_search(points, universe_points, universe_lines, universe_point_line_map, universe_point_indices, deadline):
    """
    Perform local search on points using the precomputed universe.
    """
    n_pts = len(points)
    M = len(universe_lines)
    
    # Map current points to universe indices
    current_universe_indices = []
    for p in points:
        key = (p[0], p[1])
        if key in universe_point_indices:
            current_universe_indices.append(universe_point_indices[key])
        else:
            return points # Cannot map, skip search
            
    current_universe_indices = np.array(current_universe_indices)
    
    # Initialize counts for all universe lines
    counts = np.zeros(M, dtype=np.int64)
    for ui in current_universe_indices:
        for li in universe_point_line_map[ui]:
            counts[li] += 1
            
    def _top_k_sum(c, k=27):
        if len(c) < k:
            return int(np.sum(c))
        return int(np.partition(c, -k)[-k:].sum())
        
    current_score = _top_k_sum(counts)
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(n_pts):
            old_ui = current_universe_indices[i]
            
            # Try moving to all other points in the universe
            for new_ui in range(len(universe_point_line_map)):
                if new_ui == old_ui:
                    continue
                
                # Check if new_ui is already occupied by another point
                if np.any(current_universe_indices == new_ui):
                    continue
                
                # Calculate change in counts
                old_lines = set(universe_point_line_map[old_ui])
                new_lines = set(universe_point_line_map[new_ui])
                
                lines_to_decrement = old_lines - new_lines
                lines_to_increment = new_lines - old_lines
                
                # Apply changes
                for li in lines_to_decrement:
                    counts[li] -= 1
                for li in lines_to_increment:
                    counts[li] += 1
                    
                new_score = _top_k_sum(counts)
                
                if new_score > current_score:
                    # Accept move
                    current_universe_indices[i] = new_ui
                    points[i] = universe_points[new_ui]
                    current_score = new_score
                    improved = True
                else:
                    # Revert changes
                    for li in lines_to_decrement:
                        counts[li] += 1
                    for li in lines_to_increment:
                        counts[li] -= 1
                        
    return points

def _get_all_lines_and_counts(points):
    """
    Generate all lines through pairs of points and count incidences.
    Returns:
        lines_dict: { (A,B,C): count }
    """
    n = points.shape[0]
    lines_dict = {}
    
    # Generate lines from all pairs
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            A = dy
            B = -dx
            C = dx * p1[1] - dy * p1[0]
            key = _normalize_line(A, B, C)
            if key is not None:
                if key not in lines_dict:
                    lines_dict[key] = 0
                
    # Count incidences accurately for each line using vectorized ops
    if lines_dict:
        keys = list(lines_dict.keys())
        A_arr = np.array([k[0] for k in keys], dtype=np.int64)
        B_arr = np.array([k[1] for k in keys], dtype=np.int64)
        C_arr = np.array([k[2] for k in keys], dtype=np.int64)
        xs = points[:, 0].astype(np.int64)
        ys = points[:, 1].astype(np.int64)
        vals = A_arr[:, None] * xs[None, :] + B_arr[:, None] * ys[None, :] + C_arr[:, None]
        counts = np.sum(vals == 0, axis=1)
        for idx, key in enumerate(keys):
            lines_dict[key] = int(counts[idx])
            
    return lines_dict

def _select_top_lines(lines_dict, k=27):
    """Select top k lines by incidence count."""
    sorted_lines = sorted(lines_dict.items(), key=lambda x: (-x[1], x[0]))
    selected = [l for l, c in sorted_lines[:k]]
    if len(selected) < k:
        # Pad with dummy lines that have 0 incidence
        sel_keys = set(selected)
        for key in lines_dict:
            if len(selected) >= k:
                break
            if key not in sel_keys:
                selected.append(key)
        while len(selected) < k:
            selected.append((1, 0, 1000000 + len(selected)))
    return np.array(selected, dtype=np.int64)

def _top_k_sum(counts, k=27):
    """Return sum of top k values in a dict."""
    if len(counts) < k:
        return sum(counts.values())
    vals = list(counts.values())
    return sum(sorted(vals, reverse=True)[:k])

def _try_grid(a, b, deadline):
    """Try a grid of size a*b, select top 54 points and top 27 lines."""
    if a * b < 54:
        return None, None, -1
        
    pts_base = _generate_grid(a, b)
    
    # If exact size, just use it
    if a * b == 54:
        pts = pts_base
    else:
        # Greedy removal: remove points that cause least damage
        # This is a placeholder, we should use a better method
        # For now, just take the first 54? Or center?
        # Let's use a simple heuristic: remove points on the border first?
        # Actually, let's just use the first 54 for now to keep it simple,
        # or better, use the local search to optimize from a random subset?
        # Let's just use the first 54 points for now.
        pts = pts_base[:54]
        
    lines_dict = _get_all_lines_and_counts(pts)
    lines_arr = _select_top_lines(lines_dict)
    
    # Calculate score
    I = 0
    for i in range(54):
        for j in range(27):
            if lines_arr[j, 0] * pts[i, 0] + lines_arr[j, 1] * pts[i, 1] + lines_arr[j, 2] == 0:
                I += 1
                
    return pts, lines_arr, I

def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines.
    """
    start_time = time.time()
    deadline = start_time + 600
    
    best_score = -1
    best_points = None
    best_lines = None
    
    def update_best(pts, lines, score):
        nonlocal best_score, best_points, best_lines
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            best_lines = lines.copy()
            
    # Phase 1: Try exact grids
    exact_grids = [
        (6, 9), (9, 6), (3, 18), (18, 3), (2, 27), (27, 2)
    ]
    for a, b in exact_grids:
        if time.time() > deadline - 60:
            break
        pts, lines, score = _try_grid(a, b, deadline)
        if pts is not None:
            update_best(pts, lines, score)
            
    # Phase 2: Try larger grids and use local search
    # Define a universe for local search
    # Use a 10x10 grid as universe
    universe_points = _generate_grid(10, 10)
    universe_lines, universe_point_line_map, universe_point_indices = _build_universe(universe_points)
    
    # Try starting from different subsets of the universe
    # 1. Centered 6x9
    # 2. Random subsets
    
    # For now, let's try to improve the best found so far using local search
    if best_points is not None:
        # Check if best_points are in universe
        # If not, we might need to expand universe or skip
        # Let's assume they are in 10x10 for now
        # If not, we create a larger universe
        max_x = np.max(best_points[:, 0])
        max_y = np.max(best_points[:, 1])
        min_x = np.min(best_points[:, 0])
        min_y = np.min(best_points[:, 1])
        
        # Expand universe to cover best_points plus a margin
        # Simple: just use 12x12
        universe_points = _generate_grid(12, 12)
        universe_lines, universe_point_line_map, universe_point_indices = _build_universe(universe_points)
        
        # Ensure best_points are in universe
        # If not, we can't use local search directly.
        # Let's just try local search if all points are in universe
        in_universe = True
        for p in best_points:
            if (p[0], p[1]) not in universe_point_indices:
                in_universe = False
                break
                
        if in_universe:
            # Perform local search
            pts_optimized = _local_search(best_points, universe_points, universe_lines, universe_point_line_map, universe_point_indices, deadline)
            
            # Calculate new score
            lines_dict = _get_all_lines_and_counts(pts_optimized)
            lines_arr = _select_top_lines(lines_dict)
            
            I = 0
            for i in range(54):
                for j in range(27):
                    if lines_arr[j, 0] * pts_optimized[i, 0] + lines_arr[j, 1] * pts_optimized[i, 1] + lines_arr[j, 2] == 0:
                        I += 1
                        
            update_best(pts_optimized, lines_arr, I)
            
    # Fallback
    if best_points is None:
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines_list = []
        for c in range(1, 7):
            lines_list.append((1, 0, -c))
        for c in range(1, 10):
            lines_list.append((0, 1, -c))
        for c in [0, 1, 2, 3]:
            lines_list.append((1, -1, c))
        for c in [7, 8, 9, 10]:
            lines_list.append((1, 1, -c))
        for c in [4, -1]:
            lines_list.append((1, -1, c))
        for c in [6, 11]:
            lines_list.append((1, 1, -c))
        best_lines = np.array(lines_list, dtype=np.int64)
        
    return best_points, best_lines

# EVOLVE-BLOCK-END
