"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import combinations

def _compute_best_area_multiplicity(points):
    """Compute the largest multiplicity of a single positive doubled area."""
    from collections import Counter
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

def _make_grid(w, h, offset_x=0, offset_y=0):
    """Create a w x h grid of points."""
    points = []
    for x in range(w):
        for y in range(h):
            points.append((x + offset_x, y + offset_y))
    return points

def _make_hexagonal(rows, cols):
    """Create a hexagonal-like grid."""
    points = []
    for r in range(rows):
        for c in range(cols):
            x = c + (r % 2) * 0  # simple grid for now
            y = r
            points.append((x, y))
    return points

def _try_configurations():
    """Try various configurations and return the best one."""
    candidates = []
    
    # Rectangular grids with w*h = 30
    grid_dims = [(1,30), (2,15), (3,10), (5,6), (6,5), (10,3), (15,2), (30,1)]
    for w, h in grid_dims:
        # Center the grid in [-30,30]
        ox = -w // 2
        oy = -h // 2
        pts = _make_grid(w, h, ox, oy)
        # Ensure all in [-30,30]
        if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts):
            candidates.append(pts)
    
    # Also try shifted/scaled variants
    # Try a 6x5 grid with different spacing
    for scale in [1, 2]:
        for w, h in [(6,5), (5,6)]:
            pts = []
            for x in range(w):
                for y in range(h):
                    px = x * scale - (w-1)*scale//2
                    py = y * scale - (h-1)*scale//2
                    if -30 <= px <= 30 and -30 <= py <= 30:
                        pts.append((px, py))
            if len(pts) == 30:
                candidates.append(pts)
    
    # Try a more compact arrangement: 3 rows of 10
    # Or 2 rows of 15
    # Try a "staircase" or other patterns
    
    # Evaluate all candidates
    best_pts = None
    best_mult = -1
    for pts in candidates:
        if len(pts) != 30:
            continue
        # Check distinctness
        if len(set(pts)) != 30:
            continue
        mult = _compute_best_area_multiplicity(pts)
        if mult > best_mult:
            best_mult = mult
            best_pts = pts
    
    if best_pts is None:
        # Fallback: 6x5 grid
        best_pts = _make_grid(6, 5)
    
    return best_pts

def construct():
    return _try_configurations()
