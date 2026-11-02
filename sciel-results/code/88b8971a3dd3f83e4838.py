"""Improved program: place 11 unit hexagons in a regular hexagon, minimizing outer side."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

N = 11

# EVOLVE-BLOCK-START

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * math.pi / 3.0
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return np.array(verts)

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side and rotation."""
    # Transform point to hexagon's local frame
    angle = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate by -angle
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    # In local frame, hexagon vertices are at angles 0, 60, 120, 180, 240, 300
    # A regular hexagon with side s has vertices at distance s from center
    # The hexagon is the intersection of 6 half-planes
    # For a hexagon with vertex at angle 0, the half-plane constraints are:
    # x * cos(k*60) + y * sin(k*60) <= s * cos(30) = s * sqrt(3)/2 for k=0..5
    # Actually, for a regular hexagon with circumradius = side length s:
    # The apothem (inradius) = s * sqrt(3)/2
    # The constraint is: for each edge, the distance from center to edge is apothem
    # Edge normals are at angles 30, 90, 150, 210, 270, 330 degrees
    apothem = side * math.sqrt(3.0) / 2.0
    for k in range(6):
        normal_angle = math.radians(30 + k * 60)
        nx = math.cos(normal_angle)
        ny = math.sin(normal_angle)
        if lx * nx + ly * ny > apothem + 1e-10:
            return False
    return True

def hexagon_distance(p1, p2):
    """Minimum distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def hexagons_disjoint(v1, v2):
    """Check if two hexagons (as arrays of 6 vertices) are disjoint (no interior overlap).
    We check: no vertex of one is inside the other, and no edges intersect."""
    # Check vertices of v1 inside v2 and vice versa
    for i in range(6):
        for j in range(6):
            # We need to check if vertex is inside the other hexagon
            pass
    # This approach is complex; let's use a simpler check
    # Two convex polygons are disjoint if:
    # 1. No vertex of one is inside the other
    # 2. No edges intersect
    # For efficiency, we'll check vertex containment and edge intersection
    
    # Helper: point in convex polygon
    def point_in_convex_polygon(px, py, poly):
        n = len(poly)
        # Check if point is to the same side of all edges
        sign = 0
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1) % n]
            cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
            if cross > 1e-10:
                if sign < 0:
                    return False
                sign = 1
            elif cross < -1e-10:
                if sign > 0:
                    return False
                sign = -1
        return True
    
    # Check vertices
    for i in range(6):
        if point_in_convex_polygon(v1[i][0], v1[i][1], v2):
            return False
        if point_in_convex_polygon(v2[i][0], v2[i][1], v1):
            return False
    
    # Check edge intersections
    def segments_intersect(p1, p2, p3, p4):
        def orient(a, b, c):
            val = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
            if abs(val) < 1e-10:
                return 0
            return 1 if val > 0 else -1
        
        o1 = orient(p1, p2, p3)
        o2 = orient(p1, p2, p4)
        o3 = orient(p3, p4, p1)
        o4 = orient(p3, p4, p2)
        
        if o1 != o2 and o3 != o4:
            return True
        return False
    
    for i in range(6):
        for j in range(6):
            if segments_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return False
    
    return True

def compute_outer_side(inner_positions):
    """Given inner hexagon positions [(x,y,angle)], compute minimum outer hexagon side.
    The outer hexagon is centered at origin with 0 rotation for simplicity.
    We need to find the minimum side S such that all inner hexagons fit inside."""
    
    # For each inner hexagon, compute the maximum "extent" in each of the 6 directions
    # The outer hexagon (centered at origin, side S, angle 0) has the constraint:
    # For each edge normal direction k (at angles 30, 90, 150, 210, 270, 330),
    # the maximum projection of any inner hexagon vertex onto that normal must be <= S * sqrt(3)/2
    
    max_projections = [0.0] * 6
    
    for (x, y, angle_deg) in inner_positions:
        verts = hexagon_vertices(x, y, 1.0, angle_deg)
        for k in range(6):
            normal_angle = math.radians(30 + k * 60)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for v in verts:
                proj = v[0] * nx + v[1] * ny
                if proj > max_projections[k]:
                    max_projections[k] = proj
    
    # The required side S must satisfy: S * sqrt(3)/2 >= max(max_projections)
    # So S >= 2 * max(max_projections) / sqrt(3)
    max_proj = max(max_projections)
    if max_proj <= 0:
        return 0.0
    return 2.0 * max_proj / math.sqrt(3.0)

def objective(params):
    """Objective function: outer side length. params = [x1,y1,a1, x2,y2,a2, ..., xN,yN,aN]"""
    n = len(params) // 3
    inner = [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(n)]
    
    # Check disjointness - if any two overlap, return large penalty
    for i in range(n):
        for j in range(i+1, n):
            vi = hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            vj = hexagon_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            if not hexagons_disjoint(vi, vj):
                # Compute overlap penalty
                # Find minimum distance between the two hexagons
                min_dist = float('inf')
                for p in vi:
                    for q in vj:
                        d = hexagon_distance(p, q)
                        if d < min_dist:
                            min_dist = d
                return 100.0 + 10.0 * min_dist  # Large penalty
    
    return compute_outer_side(inner)

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Initial guess: hexagonal packing
    # Place hexagons in a honeycomb pattern
    hex_positions = []
    # Center
    hex_positions.append((0.0, 0.0, 0.0))
    # Ring 1: 6 hexagons at distance 2*sqrt(3) (touching)
    for k in range(6):
        angle = math.radians(30 + k * 60)
        x = 2 * math.sqrt(3) * math.cos(angle)
        y = 2 * math.sqrt(3) * math.sin(angle)
        hex_positions.append((x, y, 0.0))
    # Ring 2: need 5 more for total 11
    # Place at distance 4*sqrt(3) in alternating positions
    for k in range(5):
        angle = math.radians(0 + k * 72)  # Spread out
        x = 4 * math.sqrt(3) * math.cos(angle)
        y = 4 * math.sqrt(3) * math.sin(angle)
        hex_positions.append((x, y, 0.0))
    
    # Convert to parameter vector
    x0 = []
    for (x, y, a) in hex_positions:
        x0.extend([x, y, a])
    
    best_params = x0[:]
    best_val = objective(x0)
    
    # Try differential evolution first for global search
    bounds = [(-15, 15), (-15, 15), (0, 360)] * N
    
    try:
        result = differential_evolution(
            objective, bounds, maxiter=50, popsize=15, 
            tol=0.01, seed=42, polish=True,
            init='sobol'
        )
        elapsed = time.time() - start_time
        if elapsed < deadline:
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x[:]
    except Exception:
        pass
    
    # Local refinement with Nelder-Mead
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        try:
            result = minimize(
                objective, best_params, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x[:]
        except Exception:
            pass
    
    # Another pass with tighter tolerance
    elapsed = time.time() - start_time
    if elapsed < deadline - 2:
        try:
            result = minimize(
                objective, best_params, method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x[:]
        except Exception:
            pass
    
    # Extract final positions
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2] % 360.0
        inner.append([x, y, a])
    
    # Compute final outer side
    outer_side = compute_outer_side(inner)
    
    return inner, (0.0, 0.0), outer_side, 0.0

# EVOLVE-BLOCK-END
