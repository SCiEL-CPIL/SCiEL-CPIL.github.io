import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    """
    Returns vertices of a regular hexagon centered at (cx, cy) with side s
    and rotation angle_deg (degrees).
    """
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def _point_in_hex_dist_sq(px, py, cx, cy, s, angle_deg):
    """
    Returns squared distance from point (px, py) to the hexagon if outside,
    or 0 if inside.
    """
    a = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate by -a to align hexagon with axes
    rx = dx * math.cos(-a) - dy * math.sin(-a)
    ry = dx * math.sin(-a) + dy * math.cos(-a)
    
    apothem = s * math.sqrt(3) / 2
    sum_sq_viol = 0.0
    for i in range(3):
        theta_n = math.radians(30 + i * 60)
        proj = rx * math.cos(theta_n) + ry * math.sin(theta_n)
        if proj > apothem:
            sum_sq_viol += (proj - apothem)**2
        if proj < -apothem:
            sum_sq_viol += (-apothem - proj)**2
    return sum_sq_viol

def _get_cluster_vertices(cluster_angle_deg):
    """
    Generates vertices for all 11 unit hexagons in a fixed rigid lattice configuration,
    rotated by cluster_angle_deg around the origin.
    
    Lattice Configuration (Honeycomb-like):
    Center: (0,0)
    Ring 1 (6 hexes): distance 2*sqrt(3) from center, angles 0, 60, ...
    Ring 2 (4 hexes): selected from the next ring to maintain symmetry and density.
    
    Standard hex packing for 11:
    1 in center.
    6 in first ring.
    4 in second ring? 
    Actually, the standard "hexagonal number" H_3 = 19 is too big. H_2 = 7.
    11 is 7 + 4.
    Let's place 4 additional hexes in the gaps of the second ring or extend the first ring?
    
    A common dense packing for 11 is the 7-hex cluster (center + 6) plus 4 more attached.
    To keep it rigid and simple, let's use a specific known configuration or generate a valid one.
    
    Let's use the 7-hex cluster (radius 1 ring) and add 4 more at the "corners" of a larger hexagon?
    Or simply:
    (0,0)
    (2√3, 0), (√3, 3), (-√3, 3), (-2√3, 0), (-√3, -3), (√3, -3)  -> These are centers of the 6 surrounding.
    
    For the 4 extra, let's place them at:
    (2√3, 3√3) ? No, that's far.
    
    Let's look at the "11 hexes in a hexagon" problem specifically.
    The optimal packing is often cited as the lattice packing.
    
    Let's define the centers relative to the cluster origin:
    c0 = (0, 0)
    Ring 1 centers (distance d = 2 * apothem = sqrt(3)):
    Wait, distance between centers of touching unit hexes is 2 * apothem = sqrt(3).
    
    Centers:
    (0,0)
    (sqrt(3), 0)
    (sqrt(3)/2, 1.5)
    (-sqrt(3)/2, 1.5)
    (-sqrt(3), 0)
    (-sqrt(3)/2, -1.5)
    (sqrt(3)/2, -1.5)
    These 7 form a tight cluster.
    
    We need 4 more. Where do they go?
    In a hexagonal lattice, the next layer has 12 positions.
    Positions in layer 2 (distance 2*sqrt(3) from center, or adjacent to layer 1):
    Let's add 4 symmetrically.
    Maybe at angles 0, 90, 180, 270? No, hex symmetry is 60 deg.
    
    Let's try adding 4 at the "midpoints" of the arcs?
    Actually, a known optimal packing for 11 unit hexes in a hexagon is the lattice packing where you take a 19-hex cluster (radius 2) and remove 8? No.
    
    Let's assume the "Rigid Cluster" is the 7-hex flower + 4 attached to the outer ring.
    Let's place the 4 extra at:
    (2*sqrt(3), 0) -> adjacent to (sqrt(3), 0)
    (0, 2*sqrt(3)) -> not a valid lattice point for hex grid?
    
    Let's use a simpler rigid configuration that is guaranteed valid and dense:
    The 7-hex cluster is very dense.
    Let's just use the 7-hex cluster and 4 more placed at the "corners" of a square? No, must be hex lattice compatible to be dense.
    
    Let's use the following centers (in local coords, side=1):
    # Center
    (0, 0)
    # Ring 1
    (sqrt(3), 0)
    (sqrt(3)/2, 1.5)
    (-sqrt(3)/2, 1.5)
    (-sqrt(3), 0)
    (-sqrt(3)/2, -1.5)
    (sqrt(3)/2, -1.5)
    
    # Ring 2 (pick 4 that maintain some symmetry, e.g., along axes if possible, or 60 deg apart)
    # Valid neighbors of Ring 1:
    # (2*sqrt(3), 0) is neighbor of (sqrt(3), 0)
    # (sqrt(3), 3) is neighbor of (sqrt(3)/2, 1.5) and (sqrt(3)/2, -1.5)? No.
    
    Let's just pick 4 from the next ring that don't overlap.
    (2*sqrt(3), 0)
    (sqrt(3), 3)  # x=sqrt(3), y=3. Dist to (sqrt(3)/2, 1.5) is sqrt((sqrt(3)/2)^2 + 1.5^2) = sqrt(3/4 + 2.25) = sqrt(3) = 1.732. Correct.
    (-2*sqrt(3), 0)
    (-sqrt(3), 3)
    
    This creates an asymmetric shape. To allow the outer hexagon to rotate freely, the shape should ideally be symmetric or we just optimize the outer hex angle.
    
    Let's use a more symmetric set of 4:
    (2*sqrt(3), 0)
    (0, 2*sqrt(3)) ? Is (0, 2*sqrt(3)) a valid hex lattice point?
    Hex lattice points: x = sqrt(3)*(i + j/2), y = 1.5*j.
    If j=2, y=3. x = sqrt(3)*(i+1). If i=-1, x=0. So (0,3) is a valid lattice point.
    Distance from (0,0) is 3.
    Distance from (sqrt(3)/2, 1.5) to (0,3) is sqrt(3/4 + 2.25) = sqrt(3). OK.
    
    So let's use:
    c8 = (2*sqrt(3), 0)
    c9 = (0, 3)
    c10 = (-2*sqrt(3), 0)
    c11 = (0, -3)
    
    This has 4-fold symmetry (if we ignore the 6-fold of the base). Actually (0,3) and (0,-3) are opposite. (2s3, 0) and (-2s3, 0) are opposite.
    This shape is elongated in y? No, x-range is +/- 2*1.732 = 3.46. y-range is +/- 3.
    
    Let's verify no overlaps between new and old.
    (2*sqrt(3), 0) vs (sqrt(3), 0): dist sqrt(3). OK.
    (2*sqrt(3), 0) vs (sqrt(3)/2, 1.5): dist sqrt((1.5*sqrt(3))^2 + 1.5^2) = sqrt(6.75 + 2.25) = 3. OK.
    (0,3) vs (sqrt(3)/2, 1.5): dist sqrt(3/4 + 2.25) = sqrt(3). OK.
    (0,3) vs (-sqrt(3)/2, 1.5): dist sqrt(3/4 + 2.25) = sqrt(3). OK.
    (0,3) vs (0,0): dist 3. OK.
    
    This looks like a valid, dense, rigid cluster.
    """
    s3 = math.sqrt(3)
    centers = [
        (0, 0),
        (s3, 0),
        (s3/2, 1.5),
        (-s3/2, 1.5),
        (-s3, 0),
        (-s3/2, -1.5),
        (s3/2, -1.5),
        (2*s3, 0),
        (0, 3),
        (-2*s3, 0),
        (0, -3)
    ]
    
    a = math.radians(cluster_angle_deg)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    
    all_verts = []
    for cx, cy in centers:
        # Rotate center
        rx = cx * cos_a - cy * sin_a
        ry = cx * sin_a + cy * cos_a
        # Get vertices for unit hex at this rotated center, with same rotation
        verts = _hex_vertices(rx, ry, 1.0, cluster_angle_deg)
        all_verts.extend(verts)
        
    return all_verts

def _min_enclosing_hex_side(vertices):
    """
    Finds the minimum side length of a regular hexagon that contains all given vertices.
    We assume the outer hexagon is centered at the centroid of the vertices for simplicity in the search,
    but we optimize its angle.
    Actually, for a minimal enclosing regular hexagon, the center is usually the centroid of the "extreme" points.
    To be robust, we will optimize the outer hexagon's center (cx, cy) and angle (ang) and side (s).
    However, to keep the search space small, we can fix the center to the centroid of the cluster vertices
    and optimize only the angle and side.
    Is the optimal enclosing hexagon always centered at the centroid?
    For a symmetric cluster, yes. For an asymmetric one, maybe not.
    But our cluster is fairly symmetric (D2 symmetry).
    Let's optimize cx, cy, ang, s.
    """
    if not vertices:
        return 0.0, 0.0, 0.0, 0.0
        
    # Calculate centroid of all vertices
    cx_avg = sum(v[0] for v in vertices) / len(vertices)
    cy_avg = sum(v[1] for v in vertices) / len(vertices)
    
    def obj(x):
        ocx, ocy, ang, s = x
        pen = 0.0
        for v in vertices:
            pen += _point_in_hex_dist_sq(v[0], v[1], ocx, ocy, s, ang)
        return s + 100.0 * pen
        
    # Bounds
    # s should be at least the radius of the cluster
    max_r = max(math.hypot(v[0]-cx_avg, v[1]-cy_avg) for v in vertices)
    s_min = max_r / math.sqrt(3) * 2 # rough lower bound
    s_max = s_min * 3 + 5
    
    bounds = [
        (cx_avg - 2, cx_avg + 2),
        (cy_avg - 2, cy_avg + 2),
        (0, 180), # Hexagon has 60 deg symmetry, but let's allow 180 to be safe, or 60?
        (s_min, s_max)
    ]
    
    # Use DE for robustness
    res = differential_evolution(obj, bounds, maxiter=50, popsize=10, seed=42, polish=True, tol=1e-4)
    
    if res.success:
        ocx, ocy, ang, s = res.x
        return ocx, ocy, s, ang
    else:
        # Fallback to best found
        ocx, ocy, ang, s = res.x
        return ocx, ocy, s, ang

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # The rigid cluster is defined. We only need to optimize its global rotation.
    # The outer hexagon optimization is done inside _min_enclosing_hex_side for each cluster rotation.
    
    def total_objective(cluster_angle):
        # Get vertices for this cluster rotation
        verts = _get_cluster_vertices(cluster_angle[0])
        # Find min enclosing hex
        ocx, ocy, s, oang = _min_enclosing_hex_side(verts)
        return s
        
    # Optimize cluster rotation
    # The cluster has symmetry, so we only need to search 0 to 60 degrees (or less).
    # Let's search 0 to 90 to be safe.
    bounds = [(0, 90)]
    
    best_val = float('inf')
    best_angle = 0.0
    
    try:
        # DE for the single parameter cluster angle
        res = differential_evolution(total_objective, bounds, maxiter=20, popsize=20, seed=42, polish=True)
        if res.fun < best_val:
            best_val = res.fun
            best_angle = res.x[0]
    except Exception:
        pass
    
    if time.time() > deadline:
        # Emergency return
        verts = _get_cluster_vertices(best_angle)
        ocx, ocy, s, oang = _min_enclosing_hex_side(verts)
        return _format_result(best_angle, ocx, ocy, s, oang)
        
    # Polish the cluster angle with local search
    try:
        res2 = minimize(total_objective, [best_angle], method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 100})
        if res2.fun < best_val:
            best_val = res2.fun
            best_angle = res2.x[0]
    except Exception:
        pass
    
    # Final calculation
    verts = _get_cluster_vertices(best_angle)
    ocx, ocy, s, oang = _min_enclosing_hex_side(verts)
    
    return _format_result(best_angle, ocx, ocy, s, oang)

def _format_result(cluster_angle, ocx, ocy, os, oang):
    # Reconstruct the inner hexagon parameters for the return interface
    s3 = math.sqrt(3)
    centers = [
        (0, 0),
        (s3, 0),
        (s3/2, 1.5),
        (-s3/2, 1.5),
        (-s3, 0),
        (-s3/2, -1.5),
        (s3/2, -1.5),
        (2*s3, 0),
        (0, 3),
        (-2*s3, 0),
        (0, -3)
    ]
    
    a = math.radians(cluster_angle)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    
    inner = []
    for cx, cy in centers:
        rx = cx * cos_a - cy * sin_a
        ry = cx * sin_a + cy * cos_a
        inner.append([rx, ry, cluster_angle])
        
    return inner, (ocx, ocy), os, oang

if __name__ == "__main__":
    r = construct()
    print(r)
