"""Packing unit hexagons inside a regular hexagon."""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11
DEADLINE = 140.0  # safety margin before 150s kill

def hex_vertices(cx, cy, angle_deg=0.0):
    """
    Return the 6 vertices of a regular hexagon with side length 1,
    centered at (cx, cy), rotated by angle_deg degrees.
    A regular hexagon with side s has circumradius s.
    Vertices at angles 0, 60, 120, 180, 240, 300 degrees from center.
    """
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    x = cx + np.cos(angles)
    y = cy + np.sin(angles)
    return np.column_stack([x, y])

def polygon_area_signed(vertices):
    """Shoelace formula for signed area."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)

def point_in_polygon(px, py, vertices):
    """Ray casting to check if point (px,py) is inside polygon."""
    n = len(vertices)
    inside = False
    x1, y1 = vertices[0]
    for i in range(1, n + 1):
        x2, y2 = vertices[i % n]
        if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / (y2 - y1) + x1):
            inside = not inside
        x1, y1 = x2, y2
    return inside

def polygons_overlap(v1, v2):
    """
    Check if two convex polygons overlap.
    Use separating axis theorem.
    """
    # Get all edge normals (axes) from both polygons
    def get_axes(vertices):
        axes = []
        n = len(vertices)
        for i in range(n):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % n]
            # Edge vector
            ex, ey = x2 - x1, y2 - y1
            # Normal (perpendicular)
            length = np.sqrt(ex * ex + ey * ey)
            if length > 0:
                nx, ny = -ey / length, ex / length
                axes.append((nx, ny))
        return axes
    
    axes = get_axes(v1) + get_axes(v2)
    
    for ax, ay in axes:
        # Project both polygons onto this axis
        proj1 = v1 @ np.array([ax, ay])
        proj2 = v2 @ np.array([ax, ay])
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        # If there's a gap, no overlap
        if max1 < min2 or max2 < min1:
            return False
    return True

def check_validity(inner, outer_center, outer_side, outer_angle):
    """
    Check that all inner hexagons are inside the outer hexagon and don't overlap.
    inner: list of [x, y, angle_degrees]
    outer_center: (x, y)
    outer_side: side length of outer hexagon
    outer_angle: rotation of outer hexagon in degrees
    """
    # Get outer hexagon vertices
    outer_verts = hex_vertices(outer_center[0], outer_center[1], outer_angle)
    outer_verts = outer_verts * outer_side  # Scale by side length (circumradius = side for regular hex)
    
    # Check each inner hexagon
    inner_verts_list = []
    for i, (x, y, ang) in enumerate(inner):
        v = hex_vertices(x, y, ang)
        inner_verts_list.append(v)
        
        # Check if all vertices of inner hex are inside outer hex
        for vx, vy in v:
            if not point_in_polygon(vx, vy, outer_verts):
                return False
    
    # Check pairwise overlaps
    for i in range(len(inner_verts_list)):
        for j in range(i + 1, len(inner_verts_list)):
            if polygons_overlap(inner_verts_list[i], inner_verts_list[j]):
                return False
    
    return True

def compute_outer_side(inner):
    """
    Given positions and angles of inner hexagons, compute the minimum side length
    of a regular hexagon (with any center and rotation) that contains all of them.
    This is a simplified approach: we'll use the bounding hexagon approach.
    
    Actually, for a more tractable approach, let's fix the outer hexagon orientation
    and center, and just compute the minimum side needed.
    """
    # For simplicity, let's assume the outer hexagon is axis-aligned (angle=0)
    # and centered at the centroid of all inner hexagon centers.
    centers = np.array([[x, y] for x, y, _ in inner])
    outer_center = np.mean(centers, axis=0)
    
    # For a regular hexagon with side s and angle 0, the vertices are at:
    # (s*cos(0), s*sin(0)), (s*cos(60), s*sin(60)), etc.
    # The hexagon contains a point (px, py) if the point is within the hexagon.
    # For a hexagon centered at origin with side s, a point (x, y) is inside if:
    # |x| <= s and |y| <= s*sin(60) and |x| + |y|/tan(60) <= s
    
    # Actually, let's use a simpler check: compute the maximum "hexagonal distance"
    # from the center for all vertices of all inner hexagons.
    
    max_side = 0.0
    for x, y, ang in inner:
        v = hex_vertices(x, y, ang)
        for vx, vy in v:
            # Relative to outer center
            dx = vx - outer_center[0]
            dy = vy - outer_center[1]
            # For a regular hexagon with side s centered at origin, angle 0:
            # The hexagon is the intersection of 6 half-planes.
            # For a point to be inside, we need:
            # For each of the 6 directions, the projection onto that direction <= s
            # The 6 directions are at angles 0, 60, 120, 180, 240, 300
            # But actually, the apothem (distance from center to edge) is s * sqrt(3)/2
            # The circumradius is s.
            # A point (dx, dy) is inside a regular hexagon (side s, angle 0) centered at origin if:
            # Let's use the fact that a regular hexagon with circumradius s contains
            # all points whose "hexagonal norm" <= s.
            # The hexagonal norm for a point (x, y) relative to a hexagon with angle 0:
            # The hexagon vertices are at angles 0, 60, 120, 180, 240, 300.
            # The edges are perpendicular to directions 30, 90, 150, 210, 270, 330.
            # For each edge direction, the distance from center to edge (apothem) is s * sqrt(3)/2.
            # So for direction theta, we need |dx*cos(theta) + dy*sin(theta)| <= s * sqrt(3)/2
            # for theta in {30, 90, 150, 210, 270, 330}
            
            thetas = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
            projections = np.abs(dx * np.cos(thetas) + dy * np.sin(thetas))
            required_s = np.max(projections) / (np.sqrt(3) / 2)
            max_side = max(max_side, required_s)
    
    return max_side, tuple(outer_center)

def construct():
    """
    Place N unit hexagons inside a regular hexagon, minimizing the outer side length.
    Returns: (inner, center, side, angle)
    inner: list of [x, y, angle_degrees]
    center: (x, y) of outer hexagon
    side: side length of outer hexagon
    angle: rotation of outer hexagon in degrees
    """
    start_time = time.time()
    
    # Initial configuration: hexagonal lattice arrangement
    # For N hexagons, arrange in a hexagonal pattern
    # A hexagonal number pattern: 1, 7, 19, 37, ...
    # For N=11, let's use a roughly hexagonal arrangement
    
    # Simple initial arrangement: grid-like
    inner = []
    # Place in a 3x4 grid or similar
    dx = 2.0  # distance between centers for non-overlapping hexagons
    dy = 2.0 * np.sqrt(3) / 2  # vertical spacing for hexagonal packing
    
    # Let's try a hexagonal arrangement
    # Row 0: 2 hexagons
    # Row 1: 3 hexagons
    # Row 2: 3 hexagons
    # Row 3: 3 hexagons
    # Total: 11
    
    rows = [2, 3, 3, 3]
    y_pos = 0
    for r, count in enumerate(rows):
        x_offset = (count - 1) * dx / 2
        for c in range(count):
            x = x_offset + c * dx - dx / 2
            y = y_pos
            inner.append([x, y, 0.0])
        y_pos += dy
    
    # Optimize using gradient descent or similar
    # For simplicity, let's try a local optimization approach
    
    best_inner = [list(x) for x in inner]
    best_side, best_center = compute_outer_side(best_inner)
    best_angle = 0.0
    
    # Try simple local search: perturb positions and check if side decreases
    np.random.seed(42)
    step_size = 0.1
    iterations = 0
    max_iterations = 5000
    
    while time.time() - start_time < DEADLINE and iterations < max_iterations:
        iterations += 1
        
        # Pick a random inner hexagon and perturb it
        idx = np.random.randint(N)
        orig_x, orig_y, orig_ang = inner[idx]
        
        # Random perturbation
        new_x = orig_x + np.random.uniform(-step_size, step_size)
        new_y = orig_y + np.random.uniform(-step_size, step_size)
        new_ang = orig_ang + np.random.uniform(-10, 10)
        
        inner[idx] = [new_x, new_y, new_ang]
        
        side, center = compute_outer_side(inner)
        
        if side < best_side - 1e-9:
            best_inner = [list(x) for x in inner]
            best_side = side
            best_center = center
            step_size = max(step_size * 0.95, 0.01)
        else:
            # Revert
            inner[idx] = [orig_x, orig_y, orig_ang]
            step_size = max(step_size * 0.9, 0.01)
        
        # Check deadline
        if time.time() - start_time > DEADLINE - 5:
            break
    
    # Verify the best solution
    if not check_validity(best_inner, best_center, best_side, best_angle):
        # Fallback: use a known valid configuration
        # Simple grid arrangement with larger spacing
        inner_fallback = []
        dx = 2.1
        dy = 2.1 * np.sqrt(3) / 2
        rows = [2, 3, 3, 3]
        y_pos = 0
        for r, count in enumerate(rows):
            x_offset = (count - 1) * dx / 2
            for c in range(count):
                x = x_offset + c * dx - dx / 2
                y = y_pos
                inner_fallback.append([x, y, 0.0])
            y_pos += dy
        best_side, best_center = compute_outer_side(inner_fallback)
        best_inner = inner_fallback
    
    return best_inner, best_center, best_side, best_angle
