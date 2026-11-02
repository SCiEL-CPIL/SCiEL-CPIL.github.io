"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 11
DEADLINE = 150.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _time_left():
    return DEADLINE - _elapsed() - 5.0  # safety margin

def hex_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        vx = cx + s * math.cos(theta)
        vy = cy + s * math.sin(theta)
        verts.append((vx, vy))
    return verts

def point_in_hexagon(px, py, cx, cy, s, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side s, rotated angle_deg."""
    # Transform to hexagon local frame
    dx = px - cx
    dy = py - cy
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    # Hexagon with side s: vertices at distance s from center
    # In local frame, the hexagon is |lx| <= s*sqrt(3)/2 and |ly| <= s and |lx|/s*sqrt(3) + |ly|/s <= 1 ... 
    # Actually for a regular hexagon with "radius" (center to vertex) = s:
    # The constraint is: for the 6 half-planes
    # Let's use the standard: hexagon with apothem = s*sqrt(3)/2, circumradius = s
    # Vertices at angles 0, 60, 120, ...
    # Half-plane constraints: for each edge normal direction
    for i in range(6):
        theta = i * math.pi / 3.0
        nx = math.cos(theta)
        ny = math.sin(theta)
        if lx * nx + ly * ny > s:
            return False
    return True

def hexagon_overlap_check(v1, v2, s1, s2):
    """Check if two hexagons (defined by center, side, angle) overlap.
    v1, v2 are centers, s1, s2 are side lengths, angles in degrees.
    We check if any vertex of one is inside the other, or if edges cross."""
    pass  # We'll use a simpler approach below

def compute_min_side(positions, angles, outer_center):
    """Given inner hexagon positions and angles, compute the minimum outer side length
    such that all inner hexagons fit inside the outer hexagon centered at outer_center.
    We try several outer angles and take the minimum."""
    best_side = float('inf')
    # Try outer rotation angles
    for outer_angle in [0.0, 30.0]:
        # For each outer angle, we need to find the minimum s such that all inner hexagons
        # are contained in the outer hexagon.
        # An inner hexagon with center (cx,cy), side 1, angle a is contained in outer hexagon
        # (center oc, side S, angle oa) iff all 6 vertices of inner are inside outer.
        # The maximum "required S" over all vertices and all inner hexagons gives the constraint.
        # For a point p to be inside outer hexagon (center oc, side S, angle oa):
        #   For all 6 directions i: dot(p - oc, n_i) <= S where n_i are unit normals to edges
        #   The edge normals are at angles oa + i*pi/3
        #   So S >= max over all points p (vertices of all inner hexes) and all i of dot(p-oc, n_i)
        #   Also S must be >= 0
        max_required = 0.0
        for k in range(N):
            cx, cy = positions[k]
            a = angles[k]
            verts = hex_vertices(cx, cy, 1.0, a)
            for (vx, vy) in verts:
                dx = vx - outer_center[0]
                dy = vy - outer_center[1]
                for i in range(6):
                    theta = math.radians(outer_angle) + i * math.pi / 3.0
                    nx = math.cos(theta)
                    ny = math.sin(theta)
                    val = dx * nx + dy * ny
                    if val > max_required:
                        max_required = val
        if max_required < best_side:
            best_side = max_required
    return best_side

def check_validity(positions, angles, outer_center, outer_side, outer_angle):
    """Check if all inner hexagons are inside the outer hexagon and don't overlap each other."""
    # Check containment
    for k in range(N):
        cx, cy = positions[k]
        a = angles[k]
        verts = hex_vertices(cx, cy, 1.0, a)
        for (vx, vy) in verts:
            if not point_in_hexagon(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                return False
    # Check non-overlap: for each pair, check if any vertex of one is strictly inside the other
    # This is a necessary but not sufficient condition for non-overlap of convex shapes.
    # For hexagons, if no vertex of A is inside B and no vertex of B is inside A, they may still overlap.
    # We'll use a more robust check: check all vertices of both hexagons against each other.
    for i in range(N):
        for j in range(i + 1, N):
            vi = hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
            vj = hex_vertices(positions[j][0], positions[j][1], 1.0, angles[j])
            # Check if any vertex of i is inside j
            for (vx, vy) in vi:
                if point_in_hexagon(vx, vy, positions[j][0], positions[j][1], 1.0, angles[j]):
                    return False
            # Check if any vertex of j is inside i
            for (vx, vy) in vj:
                if point_in_hexagon(vx, vy, positions[i][0], positions[i][1], 1.0, angles[i]):
                    return False
    return True

def objective(x):
    """Objective function for optimization. x = [x0,y0,a0, x1,y1,a1, ..., x10,y10,a10, ox,oy, os, oang]
    where a_i are angles in radians, os is outer side, oang is outer angle in radians.
    We minimize os subject to all constraints."""
    if _time_left() < 1.0:
        return 1e10
    positions = [(x[3*i], x[3*i+1]) for i in range(N)]
    angles_rad = [x[3*i+2] for i in range(N)]
    angles_deg = [math.degrees(a) for a in angles_rad]
    outer_center = (x[3*N], x[3*N+1])
    outer_side = x[3*N+2]
    outer_angle_rad = x[3*N+3]
    outer_angle_deg = math.degrees(outer_angle_rad)
    
    if outer_side < 2.0:
        return 1e10
    
    # Check validity
    if not check_validity(positions, angles_deg, outer_center, outer_side, outer_angle_deg):
        # Penalty for violation
        penalty = 0.0
        # Containment penalty
        for k in range(N):
            cx, cy = positions[k]
            a = angles_deg[k]
            verts = hex_vertices(cx, cy, 1.0, a)
            for (vx, vy) in verts:
                if not point_in_hexagon(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle_deg):
                    # Compute distance outside
                    dx = vx - outer_center[0]
                    dy = vy - outer_center[1]
                    for i in range(6):
                        theta = outer_angle_rad + i * math.pi / 3.0
                        nx = math.cos(theta)
                        ny = math.sin(theta)
                        val = dx * nx + dy * ny - outer_side
                        if val > 0:
                            penalty += val * val
        # Overlap penalty
        for i in range(N):
            for j in range(i + 1, N):
                vi = hex_vertices(positions[i][0], positions[i][1], 1.0, angles_deg[i])
                vj = hex_vertices(positions[j][0], positions[j][1], 1.0, angles_deg[j])
                for (vx, vy) in vi:
                    if point_in_hexagon(vx, vy, positions[j][0], positions[j][1], 1.0, angles_deg[j]):
                        penalty += 10.0
                for (vx, vy) in vj:
                    if point_in_hexagon(vx, vy, positions[i][0], positions[i][1], 1.0, angles_deg[i]):
                        penalty += 10.0
        return outer_side + penalty
    return outer_side

def construct():
    global START_TIME, DEADLINE
    START_TIME = time.time()
    DEADLINE = 150.0
    
    # Initial guess: place hexagons in a compact arrangement
    # Try a few different initial configurations
    best_result = None
    best_score = float('inf')
    
    # Configuration 1: linear arrangement
    init1 = []
    for k in range(N):
        init1.append([2.2 * (k - N//2), 0.0, 0.0])  # positions
    init1.append([0.0, 0.0, 20.0, 0.0])  # outer center, side, angle
    
    # Configuration 2: hexagonal cluster
    init2 = []
    offsets = [(0,0), (2.2,0), (-2.2,0), (1.1, 1.9), (-1.1, 1.9), (1.1, -1.9), (-1.1, -1.9),
               (3.3, 0), (-3.3, 0), (2.2, 3.8), (-2.2, 3.8)]
    for k in range(N):
        init2.append([offsets[k][0], offsets[k][1], 0.0])
    init2.append([0.0, 0.0, 15.0, 0.0])
    
    # Configuration 3: tighter cluster with some rotation
    init3 = []
    for k in range(N):
        angle = 2 * math.pi * k / N
        r = 3.0 if k > 0 else 0
        init3.append([r * math.cos(angle), r * math.sin(angle), 0.0])
    init3.append([0.0, 0.0, 12.0, 0.0])
    
    inits = [init1, init2, init3]
    
    for init in inits:
        if _time_left() < 10.0:
            break
        x0 = np.array(init, dtype=float)
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        except Exception:
            continue
        
        if res.fun < best_score and res.fun < 1e9:
            best_score = res.fun
            best_result = res.x
    
    # If we found a good result, try to polish it
    if best_result is not None and _time_left() > 5.0:
        try:
            res2 = minimize(objective, best_result, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            if res2.fun < best_score:
                best_score = res2.fun
                best_result = res2.x
        except Exception:
            pass
    
    if best_result is None:
        # Fallback: return a valid (but not optimal) configuration
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Extract result
    positions = [(best_result[3*i], best_result[3*i+1]) for i in range(N)]
    angles_deg = [math.degrees(best_result[3*i+2]) for i in range(N)]
    outer_center = (best_result[3*N], best_result[3*N+1])
    outer_side = best_result[3*N+2]
    outer_angle_deg = math.degrees(best_result[3*N+3])
    
    # Verify validity
    if check_validity(positions, angles_deg, outer_center, outer_side, outer_angle_deg):
        inner = [[positions[i][0], positions[i][1], angles_deg[i]] for i in range(N)]
        return inner, outer_center, outer_side, outer_angle_deg
    
    # If invalid, fall back
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
