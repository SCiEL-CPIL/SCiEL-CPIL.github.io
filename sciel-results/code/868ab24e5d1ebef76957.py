"""
Rewritten program: maximize the largest multiplicity of a single positive doubled area
among all non-degenerate triangles formed by 30 distinct integer points in [-30,30]^2.

Strategy:
1. Use a lattice-based construction. Points on a small grid or arithmetic progression
   tend to produce many equal-area triangles.
2. The doubled area of triangle (a,b,c) is |det(b-a, c-a)|.
   For points on a line, area is 0 (excluded).
   For points forming a parallelogram or regular lattice, many triangles share the same area.

Key insight: If we take points from a k x m grid (with k*m = 30), the doubled areas
take values that are multiples of the grid cell area. By choosing the grid dimensions
and orientation carefully, we can maximize the count of triangles with the same doubled area.

A 5x6 grid gives 30 points. Let's try that and also try other arrangements.

Better idea: Use a rectangular grid with dimensions that give many equal-area triangles.
For a standard axis-aligned grid, the doubled area of a triangle with vertices at grid points
is determined by the determinant. The most common non-zero doubled area in a dense grid
tends to be the smallest positive one (e.g., 1 or 2 for unit grid).

Let's compute for the 5x6 grid and see what we get, then try to improve via local search.

Actually, let's think more carefully. The task says "maximize the largest multiplicity of a
single positive doubled area". So we want ONE area value that appears in as many triangles as possible.

For 30 points, total triangles = C(30,3) = 4060. We want to maximize the count for one specific area.

A well-known construction: take points on a circle or on a line (but line gives 0 area).
Take points forming a regular polygon? Integer coordinates constrain us.

Let's try: a 5x6 grid. Compute all triple areas, find the max multiplicity.
Then try local perturbations to improve.

Given the 90s time limit, I'll do:
1. Start with a 5x6 grid.
2. Try several grid orientations/sizes.
3. Do a limited local search (swap points) to improve the score.
"""

import time
import random
import numpy as np
from collections import Counter

def _make_grid(cols, rows, x0=0, y0=0):
    """Generate a cols x rows grid of points starting at (x0, y0)."""
    pts = []
    for r in range(rows):
        for c in range(cols):
            pts.append((x0 + c, y0 + r))
    return pts

def _make_sheared_grid(cols, rows, x0=0, y0=0, shear=1):
    """Generate a sheared cols x rows grid."""
    pts = []
    for r in range(rows):
        for c in range(cols):
            x = x0 + c + shear * r
            y = y0 + r
            pts.append((x, y))
    return pts

def _full_score_and_target(points):
    """Compute full histogram, return (max_count, target_area)."""
    n = len(points)
    counts = Counter()
    for i in range(n):
        xi, yi = points[i]
        for j in range(i+1, n):
            xj, yj = points[j]
            for k in range(j+1, n):
                xk, yk = points[k]
                d = abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
                if d > 0:
                    counts[d] += 1
    if not counts:
        return 0, 0
    target_area, max_count = counts.most_common(1)[0]
    return max_count, target_area

def _delta_count_for_target(points, idx, new_pt, target_area):
    """
    Calculate the change in the count of triangles with 'target_area'
    when points[idx] is replaced by new_pt. Uses NumPy for speed.
    """
    n = len(points)
    old_pt = points[idx]
    # Get all other points as arrays
    mask = np.ones(n, dtype=bool)
    mask[idx] = False
    other = np.array(points, dtype=np.int64)
    other = other[mask]
    
    ox, oy = old_pt
    nx, ny = new_pt
    
    # Vectors from old point to others
    vx_old = other[:, 0] - ox
    vy_old = other[:, 1] - oy
    # Vectors from new point to others
    vx_new = other[:, 0] - nx
    vy_new = other[:, 1] - ny
    
    # For each pair (j, k), compute determinant
    # We need to compute det for all pairs
    # This is expensive but we can vectorize partially
    # Actually, let's just do it in a tight loop since n=29
    delta = 0
    m = len(other)
    for j in range(m):
        xj, yj = other[j]
        for k in range(j + 1, m):
            xk, yk = other[k]
            d_old = abs((xj - ox) * (yk - oy) - (yj - oy) * (xk - ox))
            d_new = abs((xj - nx) * (yk - ny) - (yj - ny) * (xk - nx))
            if d_old == target_area and d_old > 0:
                delta -= 1
            if d_new == target_area and d_new > 0:
                delta += 1
    return delta

def _delta_count_for_swap(points, idx1, idx2, target_area):
    """
    Calculate the change in count of target_area triangles when swapping
    points[idx1] and points[idx2].
    """
    p1 = points[idx1]
    p2 = points[idx2]
    n = len(points)
    delta = 0
    for i in range(n):
        if i == idx1 or i == idx2:
            continue
        xi, yi = points[i]
        # Old: triangle (p1, p2, pi)
        d_old = abs((p2[0] - p1[0]) * (yi - p1[1]) - (p2[1] - p1[1]) * (xi - p1[0]))
        # New: triangle (p2, p1, pi) - same triangle, same area
        # Actually swapping two points in a triangle doesn't change the area
        # The area is the same! So delta from these triangles is 0.
        # But triangles involving p1 and some other point (not p2) change
        # and triangles involving p2 and some other point (not p1) change
        pass
    # Triangles involving p1 and some point k (k != idx1, idx2):
    # Old: (p1, pk, p2) or (p1, pk, pl) where pl != p2
    # Let's be more careful.
    # Triangles that include p1 but not p2: area changes from det(p1, pi, pj) to det(p2, pi, pj)
    # Triangles that include p2 but not p1: area changes from det(p2, pi, pj) to det(p1, pi, pj)
    # Triangles that include both p1 and p2: area unchanged
    # Triangles that include neither: unchanged
    
    for i in range(n):
        if i == idx1 or i == idx2:
            continue
        xi, yi = points[i]
        for j in range(n):
            if j == idx1 or j == idx2 or j == i:
                continue
            xj, yj = points[j]
            # Old triangle with p1: (p1, pi, pj)
            d_old = abs((xi - p1[0]) * (yj - p1[1]) - (yi - p1[1]) * (xj - p1[0]))
            # New triangle with p2 at p1's position: (p2, pi, pj)
            d_new = abs((xi - p2[0]) * (yj - p2[1]) - (yi - p2[1]) * (xj - p2[0]))
            if d_old == target_area and d_old > 0:
                delta -= 1
            if d_new == target_area and d_new > 0:
                delta += 1
            # Old triangle with p2: (p2, pi, pj)
            d_old2 = abs((xi - p2[0]) * (yj - p2[1]) - (yi - p2[1]) * (xj - p2[0]))
            # New triangle with p1 at p2's position: (p1, pi, pj)
            d_new2 = abs((xi - p1[0]) * (yj - p1[1]) - (yi - p1[1]) * (xj - p1[0]))
            if d_old2 == target_area and d_old2 > 0:
                delta -= 1
            if d_new2 == target_area and d_new2 > 0:
                delta += 1
    return delta

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin

    # Try multiple grid configurations including sheared ones
    best_pts = None
    best_count = 0
    target_area = 0
    
    configs = [
        ('grid', (6, 5, -2, -2, 0)),
        ('grid', (5, 6, -2, -2, 0)),
        ('grid', (10, 3, -4, -1, 0)),
        ('grid', (3, 10, -1, -4, 0)),
        ('grid', (15, 2, -7, 0, 0)),
        ('grid', (2, 15, 0, -7, 0)),
        ('sheared', (6, 5, -3, -2, 1)),
        ('sheared', (5, 6, -2, -3, 1)),
        ('sheared', (6, 5, -2, -2, 2)),
        ('sheared', (5, 6, -2, -2, 2)),
        ('sheared', (10, 3, -5, -1, 1)),
        ('sheared', (3, 10, -1, -5, 1)),
    ]
    
    for kind, params in configs:
        if kind == 'grid':
            cols, rows, x0, y0, _ = params
            if cols * rows != 30:
                continue
            pts = _make_grid(cols, rows, x0, y0)
        else:
            cols, rows, x0, y0, shear = params
            if cols * rows != 30:
                continue
            pts = _make_sheared_grid(cols, rows, x0, y0, shear)
        
        if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
            c, t = _full_score_and_target(pts)
            if c > best_count:
                best_count = c
                target_area = t
                best_pts = pts

    if best_pts is None:
        best_pts = _make_grid(6, 5, -2, -2)
        best_count, target_area = _full_score_and_target(best_pts)

    random.seed(42)
    no_improve_counter = 0
    
    while time.time() < deadline:
        # Periodically re-evaluate to see if a different area has become dominant
        if no_improve_counter > 100:
            c, t = _full_score_and_target(best_pts)
            if c > best_count:
                best_count = c
                target_area = t
            no_improve_counter = 0
            continue

        # Choose between move and swap
        use_swap = no_improve_counter > 50 and random.random() < 0.3
        
        improved = False
        
        if not use_swap:
            idx = random.randint(0, 29)
            orig = best_pts[idx]
            
            radius = 2 if no_improve_counter < 50 else 4
            
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = orig[0] + dx, orig[1] + dy
                    if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                        continue
                    if (nx, ny) in best_pts:
                        continue
                    
                    delta = _delta_count_for_target(best_pts, idx, (nx, ny), target_area)
                    if delta > 0:
                        best_pts[idx] = (nx, ny)
                        best_count += delta
                        no_improve_counter = 0
                        improved = True
                        break
                if improved:
                    break
        else:
            # Try a swap
            idx1 = random.randint(0, 29)
            idx2 = random.randint(0, 29)
            if idx1 == idx2:
                idx2 = (idx2 + 1) % 30
            
            delta = _delta_count_for_swap(best_pts, idx1, idx2, target_area)
            if delta > 0:
                best_pts[idx1], best_pts[idx2] = best_pts[idx2], best_pts[idx1]
                best_count += delta
                no_improve_counter = 0
                improved = True
        
        if not improved:
            no_improve_counter += 1
            
        # Occasionally try a random jump to escape local optima
        if no_improve_counter > 200 and random.random() < 0.1:
            idx = random.randint(0, 29)
            nx = random.randint(-30, 30)
            ny = random.randint(-30, 30)
            if (nx, ny) not in best_pts:
                delta = _delta_count_for_target(best_pts, idx, (nx, ny), target_area)
                if delta > 0 or (delta == 0 and random.random() < 0.05):
                    best_pts[idx] = (nx, ny)
                    best_count += delta
                    no_improve_counter = 0

    return [list(p) for p in best_pts]
