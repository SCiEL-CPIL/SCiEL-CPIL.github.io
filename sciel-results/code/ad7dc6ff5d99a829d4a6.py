"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np
from scipy.spatial import ConvexHull

N = 12
# Deadline guard: evaluator kills after 150s wall-clock. Use 135s to be safe.
DEADLINE = 135.0

def hex_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy), side s, rotated angle_deg."""
    a = math.radians(angle_deg)
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 (flat-top or pointy-top)
    # Standard: vertex i at angle = a + i*pi/3, distance from center to vertex = s (for regular hexagon, circumradius = side)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def convex_hull_area(vertices):
    """Compute area of convex hull of a set of 2D points."""
    pts = np.array(vertices, dtype=float)
    if len(pts) < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, .volume gives area
    except Exception:
        # Fallback: shoelace on sorted points won't work for non-convex, but we assume convex or use scipy
        return 0.0

def min_enclosing_hexagon_area(inner_verts_list):
    """
    Find the minimum area regular hexagon that encloses all given inner hexagon vertices.
    We search over rotation angles of the outer hexagon.
    For a fixed outer rotation, we find the minimum side length S such that all points are inside.
    Then area = (3*sqrt(3)/2) * S^2.
    """
    all_pts = []
    for verts in inner_verts_list:
        all_pts.extend(verts)
    if not all_pts:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    all_pts = np.array(all_pts, dtype=float)
    
    best_area = float('inf')
    best_S = float('inf')
    best_angle = 0.0
    best_center = (0.0, 0.0)
    
    # Search over outer hexagon rotation angles
    # The optimal orientation is likely aligned with one of the inner hexagons or symmetric.
    # We'll sample angles from 0 to 60 degrees (since hexagon has 60-degree symmetry).
    # For each angle, we need to find the minimum S and center.
    # This is a convex optimization problem for fixed angle? Not exactly, but we can use a grid or iterative method.
    # Simpler approach: For a fixed outer hexagon orientation, the minimum enclosing regular hexagon 
    # can be found by checking the constraints. However, this is complex.
    
    # Alternative: Use the fact that the minimum enclosing regular hexagon area is determined by the 
    # "width" of the set in 3 directions (or 6). 
    # Actually, a simpler heuristic: The bounding box of the points in a rotated frame.
    # Let's use a more robust method: 
    # For a regular hexagon with side S and rotation alpha, a point (x,y) is inside if:
    # max over 6 directions of projection <= S * sqrt(3)/2 * something?
    # The apothem (distance from center to side) of a regular hexagon with side S is S * sqrt(3)/2.
    # So the condition for a point to be inside a hexagon centered at (cx,cy) with rotation alpha is:
    # For each of the 6 normal directions n_i (unit vectors perpendicular to sides),
    # | (p - c) dot n_i | <= apothem = S * sqrt(3)/2.
    # The 6 normal directions are at angles alpha + 30 + i*60 degrees.
    
    # So for fixed alpha, we want to minimize S such that there exists c with:
    # max_i | (p_j - c) dot n_i(alpha) | <= S * sqrt(3)/2 for all j.
    # This is equivalent to finding the minimum width in each of the 6 directions, and taking the max.
    # Specifically, let w_i(c) = max_j (p_j dot n_i) - min_j (p_j dot n_i). Then we need S * sqrt(3) >= w_i(c) for all i.
    # So S >= max_i w_i(c) / sqrt(3).
    # We want to minimize max_i w_i(c) over c. This is a convex optimization problem in c.
    # We can solve it using scipy.optimize.minimize on f(c) = max_i w_i(c).
    
    # But this is expensive to do for many angles. Let's sample a few angles and use a simple grid for c.
    # Or, we can use the fact that the optimal c is likely near the centroid.
    
    # Let's try a simpler approach: 
    # 1. Compute the centroid of all points.
    # 2. For a set of candidate angles, compute the minimum S by finding the required apothem in each direction.
    # 3. The center c that minimizes the max width is the "center of the smallest enclosing interval" in each direction, 
    #    but coupled. This is the Chebyshev center of the intersection of strips.
    #    Actually, for fixed alpha, the problem is:
    #    Minimize t such that there exists c with:
    #    (p_j - c) dot n_i <= t for all i,j
    #    (c - p_j) dot n_i <= t for all i,j
    #    This is a linear program! We can solve it with scipy.optimize.linprog.
    
    # However, solving LP for many angles is slow. Let's use a heuristic:
    # The optimal center is often the centroid. Let's assume c = centroid for now, and optimize S.
    # Then refine c if time permits.
    
    centroid = all_pts.mean(axis=0)
    
    # Sample angles from 0 to 60 degrees
    angles = np.linspace(0, 60, 121)  # 121 samples, step 0.5 deg
    
    for alpha_deg in angles:
        alpha = math.radians(alpha_deg)
        # 6 normal directions
        normals = []
        for i in range(6):
            theta = alpha + math.radians(30 + i * 60)
            normals.append((math.cos(theta), math.sin(theta)))
        
        # Compute projections of all points onto each normal
        # proj[i][j] = p_j dot n_i
        proj = np.zeros((6, len(all_pts)))
        for i, n in enumerate(normals):
            proj[i] = all_pts[:, 0] * n[0] + all_pts[:, 1] * n[1]
        
        # For each direction, the width is max - min
        widths = proj.max(axis=1) - proj.min(axis=1)
        max_width = widths.max()
        
        # S * sqrt(3) >= max_width  =>  S >= max_width / sqrt(3)
        S = max_width / math.sqrt(3)
        area = (3 * math.sqrt(3) / 2) * S * S
        
        if area < best_area:
            best_area = area
            best_S = S
            best_angle = alpha_deg
            best_center = tuple(centroid)
    
    return best_area, best_S, best_angle, best_center[0], best_center[1]

def compute_score(inner):
    """Compute the score (outer side length) for a given inner configuration."""
    inner_verts_list = []
    for row in inner:
        x, y, ang = row
        verts = hex_vertices(x, y, 1.0, ang)
        inner_verts_list.append(verts)
    
    area, S, ang, cx, cy = min_enclosing_hexagon_area(inner_verts_list)
    return S, (cx, cy), ang

def check_overlap(inner, tol=1e-9):
    """Check if any two inner hexagons overlap. Return True if valid (no overlap)."""
    # For simplicity, we'll use a quick check: distance between centers >= 2.0 * something?
    # Actually, for two unit hexagons, the minimum distance between centers to avoid overlap depends on orientation.
    # The "radius" (max distance from center to vertex) is 1.0.
    # So if distance between centers > 2.0, they definitely don't overlap.
    # If < 2.0, they might. We need a more precise check.
    # For now, we'll use a conservative check: if distance < 1.9, assume overlap.
    # This is not perfect, but good enough for a heuristic.
    n = len(inner)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, _ = inner[i]
            x2, y2, _ = inner[j]
            d = math.hypot(x1 - x2, y1 - y2)
            if d < 1.9:  # Conservative threshold
                return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Initial configuration: 12 hexagons in a hexagonal arrangement
    # Side length 1 hexagons. A hexagonal packing of 12 hexagons:
    # Center + 6 around + 5 more? 
    # Actually, a regular hexagon of side 2 has 7 hexagons (1 center + 6).
    # For 12, we can do a 3-4-3 arrangement or similar.
    
    # Let's start with a grid-like arrangement.
    # Hexagonal lattice: 
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons? Total = 12.
    
    # Better: Use a hexagonal packing.
    # The distance between centers of adjacent hexagons in a hexagonal lattice is 2 * apothem = 2 * sqrt(3)/2 = sqrt(3) ≈ 1.732.
    # Wait, for unit side hexagons, the distance between centers of adjacent hexagons (sharing an edge) is 2 * apothem = sqrt(3).
    # But they must be disjoint, so we need a small gap. Let's use distance = 1.75.
    
    # Let's define a hexagonal lattice with spacing 1.75.
    # Actually, let's use a simpler initial guess: 12 hexagons in a line? No, that's bad.
    # Let's use a 3x4 grid with hexagonal offset.
    
    # Hexagonal lattice coordinates:
    # (i, j) -> (i * 1.75 + 0.875 * (j % 2), j * 1.5)
    # Let's try a 4x3 grid.
    
    inner = []
    # 4 columns, 3 rows
    for j in range(3):
        for i in range(4):
            x = i * 1.75 + (0.875 if j % 2 == 1 else 0)
            y = j * 1.5
            inner.append([x, y, 0.0])
    
    # Center the configuration
    xs = [p[0] for p in inner]
    ys = [p[1] for p in inner]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for p in inner:
        p[0] -= cx
        p[1] -= cy
    
    # Compute initial score
    S, center, ang = compute_score(inner)
    best_S = S
    best_inner = [p[:] for p in inner]
    best_center = center
    best_ang = ang
    
    # Local search: perturb positions and angles
    random.seed(42)
    step = 0.1
    improvements = 0
    
    while time.time() - start_time < DEADLINE - 5:
        # Try a random perturbation
        i = random.randint(0, N - 1)
        new_inner = [p[:] for p in inner]
        new_inner[i][0] += random.uniform(-step, step)
        new_inner[i][1] += random.uniform(-step, step)
        new_inner[i][2] += random.uniform(-15, 15)
        
        # Check overlap
        if not check_overlap(new_inner):
            continue
        
        # Compute score
        S_new, center_new, ang_new = compute_score(new_inner)
        
        if S_new < best_S - 1e-6:
            best_S = S_new
            best_inner = new_inner
            best_center = center_new
            best_ang = ang_new
            inner = new_inner
            improvements += 1
            # Reduce step size if improving
            step = max(0.01, step * 0.95)
        else:
            # Increase step size if stuck
            if improvements > 0:
                improvements = 0
            step = min(0.5, step * 1.1)
    
    # Final result
    return best_inner, best_center, best_S, best_ang
# EVOLVE-BLOCK-END
