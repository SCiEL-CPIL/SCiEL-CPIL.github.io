"""Improved program using structured grid construction."""
import numpy as np
from collections import Counter

def _compute_best_area_multiplicity(points):
    """Compute the largest multiplicity of a single positive doubled area."""
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                det = abs(int(pts[i][0])*(pts[j][1]-pts[k][1]) + 
                         int(pts[j][0])*(pts[k][1]-pts[i][1]) + 
                         int(pts[k][0])*(pts[i][1]-pts[j][1]))
                if det > 0:
                    areas.append(det)
    if not areas:
        return 0
    counter = Counter(areas)
    return counter.most_common(1)[0][1]

def _generate_grid(w, h, offset_x=0, offset_y=0):
    """Generate a w x h grid of points."""
    points = []
    for x in range(w):
        for y in range(h):
            points.append((x + offset_x, y + offset_y))
    return points

def _generate_subsets_of_grid(w, h, num_points=30):
    """Generate several subsets of a w x h grid, aiming for num_points."""
    if w * h < num_points:
        return []
    
    full_grid = _generate_grid(w, h)
    # We need to remove (w*h - num_points) points.
    # Strategy 1: Remove corners.
    # Strategy 2: Remove boundary points uniformly.
    
    subsets = []
    
    # If exact match, return the grid
    if w * h == num_points:
        subsets.append(full_grid)
        return subsets
        
    num_to_remove = w * h - num_points
    
    # Simple heuristic: remove points from the end of the list (which are top-right)
    # This creates a "cut corner" effect.
    # Or remove specific symmetric points.
    
    # Let's try removing the last N points from the flattened grid.
    # The grid is generated row by row.
    # Removing from the end removes from the last row, rightmost to leftmost.
    subset1 = full_grid[:num_points]
    subsets.append(subset1)
    
    # Try removing corners first.
    corners = [
        (0, 0),
        (w-1, 0),
        (0, h-1),
        (w-1, h-1)
    ]
    removed = set()
    for c in corners:
        if len(removed) < num_to_remove:
            removed.add(c)
            
    # If we need to remove more, remove from the boundary
    if len(removed) < num_to_remove:
        # Iterate through boundary points
        for x in range(w):
            if (x, 0) not in removed and len(removed) < num_to_remove:
                removed.add((x, 0))
            if (x, h-1) not in removed and len(removed) < num_to_remove:
                removed.add((x, h-1))
        for y in range(h):
            if (0, y) not in removed and len(removed) < num_to_remove:
                removed.add((0, y))
            if (w-1, y) not in removed and len(removed) < num_to_remove:
                removed.add((w-1, y))
                
    # If still not enough, remove from interior (arbitrary)
    if len(removed) < num_to_remove:
        for p in full_grid:
            if p not in removed and len(removed) < num_to_remove:
                removed.add(p)
                
    subset2 = [p for p in full_grid if p not in removed]
    if len(subset2) == num_points:
        subsets.append(subset2)
        
    return subsets

def construct():
    """
    Construct 30 points to maximize the largest multiplicity of a single positive doubled area.
    Uses structured grid generation and evaluation.
    """
    best_points = None
    best_score = -1
    
    # Candidate grid dimensions (w, h)
    # We want w * h >= 30
    # Common small grids:
    candidates = [
        (5, 6), (6, 5),
        (3, 10), (10, 3),
        (2, 15), (15, 2),
        (1, 30), (30, 1),
        (6, 6), # 36 points, need to remove 6
        (7, 5), # 35 points, need to remove 5
        (5, 7), # 35 points
        (8, 4), # 32 points, need to remove 2
        (4, 8), # 32 points
        (10, 4), # 40 points
        (4, 10), # 40 points
    ]
    
    for w, h in candidates:
        subsets = _generate_subsets_of_grid(w, h, 30)
        for points in subsets:
            if len(points) != 30:
                continue
            # Center the points roughly in [-30, 30]
            # Calculate center of bounding box
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            cx = (min_x + max_x) / 2.0
            cy = (min_y + max_y) / 2.0
            
            centered_points = [(int(p[0] - cx), int(p[1] - cy)) for p in points]
            
            # Check bounds
            valid = True
            for p in centered_points:
                if not (-30 <= p[0] <= 30 and -30 <= p[1] <= 30):
                    valid = False
                    break
            
            if not valid:
                # Shift to fit
                min_cx = min(p[0] for p in centered_points)
                min_cy = min(p[1] for p in centered_points)
                max_cx = max(p[0] for p in centered_points)
                max_cy = max(p[1] for p in centered_points)
                if max_cx - min_cx > 60 or max_cy - min_cy > 60:
                    continue # Too large
                # Shift to center
                shift_x = -min_cx + (30 - (max_cx - min_cx)) // 2
                shift_y = -min_cy + (30 - (max_cy - min_cy)) // 2
                centered_points = [(p[0] + shift_x, p[1] + shift_y) for p in centered_points]
                
            # Check distinctness
            if len(set(centered_points)) != 30:
                continue
                
            score = _compute_best_area_multiplicity(centered_points)
            if score > best_score:
                best_score = score
                best_points = centered_points
                
    if best_points is None:
        # Fallback: 5x6 grid
        best_points = _generate_grid(5, 6, -2, -2)
        
    return best_points
