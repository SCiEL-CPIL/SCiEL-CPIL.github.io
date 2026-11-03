"""Improved program: pack unit hexagons in a larger hexagon using grid + local optimization."""
import numpy as np
import time
from scipy.spatial import KDTree
from scipy.optimize import minimize

N = 11  # number of inner hexagons to place

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 115.0  # seconds, leaving 5s margin before 120s kill

def hex_vertex(cx, cy, angle_deg, s=1.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with side s, rotated by angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.pi / 3 * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def polygon_overlap(v1, v2, eps=1e-9):
    """Check if two convex polygons (lists of (x,y) np arrays) overlap.
    Returns True if they overlap (interiors intersect or one contains the other).
    We treat touching as non-overlapping (disjoint allows sharing boundary)."""
    # Use separating axis theorem for convex polygons
    def get_edges(p):
        n = len(p)
        edges = []
        for i in range(n):
            dx = p[(i+1)%n][0] - p[i][0]
            dy = p[(i+1)%n][1] - p[i][1]
            # Normal
            ln = np.hypot(dx, dy)
            if ln < eps:
                edges.append((1, 0))
            else:
                edges.append((-dy/ln, dx/ln))
        return edges
    
    def project(p, axis):
        dots = p[:,0]*axis[0] + p[:,1]*axis[1]
        return dots.min(), dots.max()
    
    axes = get_edges(v1) + get_edges(v2)
    for ax in axes:
        min1, max1 = project(v1, ax)
        min2, max2 = project(v2, ax)
        if max1 <= min2 + eps or max2 <= min1 + eps:
            return False  # separating axis found
    return True

def hexagon_distance(h1, h2):
    """Minimum distance between two hexagons (0 if overlapping or touching)."""
    v1 = hex_vertex(*h1)
    v2 = hex_vertex(*h2)
    if not polygon_overlap(v1, v2):
        # Compute min distance between edges
        min_d = np.inf
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            for j in range(6):
                q1 = v2[j]
                q2 = v2[(j+1)%6]
                # Segment-segment distance
                d = seg_seg_dist(p1, p2, q1, q2)
                min_d = min(min_d, d)
        return min_d
    else:
        return 0.0

def seg_seg_dist(p1, p2, q1, q2):
    """Minimum distance between segment p1-p2 and segment q1-q2."""
    def dist_point_seg(p, a, b):
        ab = b - a
        t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-15)
        t = max(0, min(1, t))
        proj = a + t * ab
        return np.linalg.norm(p - proj)
    
    d1 = dist_point_seg(p1, q1, q2)
    d2 = dist_point_seg(p2, q1, q2)
    d3 = dist_point_seg(q1, p1, p2)
    d4 = dist_point_seg(q2, p1, p2)
    
    # Check if segments intersect
    if segs_intersect(p1, p2, q1, q2):
        return 0.0
    
    return min(d1, d2, d3, d4)

def segs_intersect(p1, p2, q1, q2):
    """Check if segments p1-p2 and q1-q2 intersect."""
    def orient(a, b, c):
        val = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        if abs(val) < 1e-10:
            return 0
        return 1 if val > 0 else -1
    
    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)
    
    if o1 != o2 and o3 != o4:
        return True
    return False

def check_fit(positions, angles, outer_center, outer_side, outer_angle):
    """Check if all inner hexagons fit inside the outer hexagon and don't overlap each other.
    Returns True if valid."""
    outer_v = hex_vertex(outer_center[0], outer_center[1], outer_angle, outer_side)
    
    # Check each inner hexagon is inside outer
    for i in range(len(positions)):
        cx, cy = positions[i]
        ang = angles[i]
        inner_v = hex_vertex(cx, cy, ang, 1.0)
        # Check all vertices of inner are inside outer
        for pt in inner_v:
            if not point_in_polygon(pt, outer_v):
                return False
    
    # Check no overlaps between inner hexagons
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            if polygon_overlap(hex_vertex(*positions[i], angles[i]), 
                               hex_vertex(*positions[j], angles[j])):
                return False
    
    return True

def point_in_polygon(pt, poly):
    """Ray casting to check if point is inside polygon."""
    x, y = pt
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside

def initial_grid_layout(n):
    """Generate initial positions using a hexagonal grid pattern."""
    positions = []
    angles = []
    # Hexagonal packing: offset rows
    # For n hexagons, try to arrange in a compact cluster
    # Use a spiral or row-based approach
    
    # Row-based hex grid
    cols = 0
    row = 0
    while len(positions) < n:
        for col in range(cols, -cols-1 if cols > 0 else 0):
            if len(positions) >= n:
                break
            # Hex grid coordinates to cartesian
            x = col * 3.0  # horizontal spacing for side-1 hexagons touching
            y = row * np.sqrt(3)
            if row % 2 == 1:
                x += 1.5
            positions.append((x, y))
            angles.append(0.0)
        row += 1
        cols += 1
    
    return positions, angles

def objective(params, positions0, angles0, outer_center, outer_side, outer_angle):
    """Objective function to minimize: maximize minimum clearance or minimize overlap."""
    positions = []
    angles = []
    for i in range(N):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = params[3*i+2]
        positions.append((cx, cy))
        angles.append(ang)
    
    # Check outer containment
    outer_v = hex_vertex(outer_center[0], outer_center[1], outer_angle, outer_side)
    penalty = 0.0
    
    for i in range(N):
        inner_v = hex_vertex(positions[i][0], positions[i][1], angles[i], 1.0)
        for pt in inner_v:
            if not point_in_polygon(pt, outer_v):
                # Distance from point to outer polygon
                d = min_dist_to_polygon(pt, outer_v)
                penalty += max(0, d) ** 2
    
    # Check mutual overlaps
    for i in range(N):
        for j in range(i+1, N):
            v1 = hex_vertex(positions[i][0], positions[i][1], angles[i], 1.0)
            v2 = hex_vertex(positions[j][0], positions[j][1], angles[j], 1.0)
            if polygon_overlap(v1, v2):
                # Estimate overlap depth
                penalty += 10.0
    
    return penalty

def min_dist_to_polygon(pt, poly):
    """Minimum distance from point to polygon boundary (0 if inside)."""
    if point_in_polygon(pt, poly):
        # Compute min distance to edges
        min_d = np.inf
        for i in range(len(poly)):
            d = seg_seg_dist(pt, pt, poly[i], poly[(i+1)%len(poly)])
            min_d = min(min_d, d)
        return min_d
    else:
        # Point is outside, find min distance to polygon
        min_d = np.inf
        for i in range(len(poly)):
            d = seg_seg_dist(pt, pt, poly[i], poly[(i+1)%len(poly)])
            min_d = min(min_d, d)
        return min_d

def construct():
    global START_TIME
    start = time.time()
    
    # Initial layout
    positions, angles = initial_grid_layout(N)
    
    # Try to find a valid outer hexagon size
    # Start with a reasonable estimate and shrink
    
    best_side = np.inf
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Compute bounding box of initial layout to estimate outer size
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    radius = max(np.hypot(x-cx, y-cy) for x, y in zip(xs, ys))
    
    # For a regular hexagon containing points at max distance r from center,
    # the side length needed is approximately r / (sqrt(3)/2) = 2r/sqrt(3)
    # But we need to account for the hexagon shape
    est_side = radius / (np.sqrt(3)/2) + 2.0  # add margin for hexagon vertices
    
    # Try different outer angles and centers
    for outer_angle in [0.0, 30.0]:
        for center_offset in [(-1.0, -1.0), (0.0, 0.0), (1.0, 1.0)]:
            oc = (cx + center_offset[0], cy + center_offset[1])
            
            # Binary search for minimum side length
            lo = est_side * 0.5
            hi = est_side * 2.0
            
            # First verify hi works
            if not check_fit(positions, angles, oc, hi, outer_angle):
                hi *= 1.5
                if not check_fit(positions, angles, oc, hi, outer_angle):
                    continue
            
            # Binary search
            for _ in range(20):
                mid = (lo + hi) / 2
                if check_fit(positions, angles, oc, mid, outer_angle):
                    hi = mid
                    best_side = min(best_side, hi)
                    if best_side < hi:
                        best_positions = positions.copy()
                        best_angles = angles.copy()
                        best_center = oc
                        best_outer_angle = outer_angle
                else:
                    lo = mid
                
                if time.time() - start > TIME_LIMIT:
                    break
            
            if time.time() - start > TIME_LIMIT:
                break
        
        if time.time() - start > TIME_LIMIT:
            break
    
    # Local optimization to shrink further
    if time.time() - start < TIME_LIMIT - 30:
        # Try to optimize positions to allow smaller outer hexagon
        current_side = best_side
        current_positions = best_positions
        current_angles = best_angles
        current_center = best_center
        current_outer_angle = best_outer_angle
        
        # Parameterize: positions (2N) + angles (N)
        params = np.array([current_positions[i][0] for i in range(N)] + 
                         [current_positions[i][1] for i in range(N)] + 
                         current_angles)
        
        # Use a simple hill-climbing approach
        step_size = 0.1
        for iteration in range(500):
            if time.time() - start > TIME_LIMIT - 5:
                break
            
            improved = False
            for i in range(N):
                for dim in range(3):
                    for direction in [-1, 1]:
                        new_params = params.copy()
                        idx = 2*N + i if dim == 2 else i + (dim * N)
                        new_params[idx] += direction * step_size
                        
                        np_pos = [(new_params[i], new_params[N+i]) for i in range(N)]
                        np_ang = list(new_params[2*N:3*N])
                        
                        if check_fit(np_pos, np_ang, current_center, current_side, current_outer_angle):
                            # Try to shrink
                            new_side = current_side * 0.98
                            if check_fit(np_pos, np_ang, current_center, new_side, current_outer_angle):
                                params = new_params
                                current_positions = np_pos
                                current_angles = np_ang
                                current_side = new_side
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                step_size *= 0.9
                if step_size < 0.001:
                    break
            
            if current_side < best_side:
                best_side = current_side
                best_positions = current_positions
                best_angles = current_angles
                best_center = current_center
                best_outer_angle = current_outer_angle
    
    # Ensure we have a valid result
    if not np.isfinite(best_side) or best_side > 100:
        # Fallback: large hexagon
        best_side = 50.0
        best_center = (0.0, 0.0)
        best_positions = [(0.0, 0.0)] * N
        best_angles = [0.0] * N
        best_outer_angle = 0.0
    
    # Final verification
    if not check_fit(best_positions, best_angles, best_center, best_side, best_outer_angle):
        # Expand until valid
        while not check_fit(best_positions, best_angles, best_center, best_side, best_outer_angle):
            best_side *= 1.1
            if best_side > 1000:
                break
    
    # Format output
    inner = [[best_positions[i][0], best_positions[i][1], best_angles[i]] for i in range(N)]
    
    return inner, best_center, best_side, best_outer_angle
