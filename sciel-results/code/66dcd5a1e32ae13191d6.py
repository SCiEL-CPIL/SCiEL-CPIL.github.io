"""Hexagon packing: place N unit regular hexagons inside a regular hexagon, minimizing outer side."""
import numpy as np
import time
from scipy.optimize import least_squares

N = 11
DEADLINE = 140.0  # seconds, keep safety margin under 150s

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    a = np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    # Unit hexagon vertices (side length 1, circumradius 1)
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    rx = np.cos(base_angles)
    ry = np.sin(base_angles)
    # Rotate
    x = cx + ca * rx - sa * ry
    y = cy + sa * rx + ca * ry
    return np.column_stack([x, y])

def _point_in_hex(px, py, cx, cy, angle_deg, side=1.0, eps=1e-9):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    a = np.radians(angle_deg)
    # Transform point to hexagon's local frame
    dx, dy = px - cx, py - cy
    lx = np.cos(a) * dx + np.sin(a) * dy
    ly = -np.sin(a) * dx + np.cos(a) * dy
    # Regular hexagon with circumradius R=side: |lx| <= R*cos(30)=R*sqrt(3)/2
    # and |ly| <= R, but also the hex shape constraint
    # A regular hexagon (pointy-top or flat-top) can be checked via:
    # max(|lx|, |ly|, |lx - ly|, |lx + ly|, |-lx - ly|, |-lx + ly|) <= R * sqrt(3)/2
    # Actually for a regular hexagon with circumradius R:
    # The hexagon is the intersection of 6 half-planes.
    # For a hexagon with vertices at angles 0,60,...,300 degrees:
    # |lx * cos(k*60) + ly * sin(k*60)| <= R * cos(30) = R * sqrt(3)/2 for k=0..5
    R = side
    limit = R * np.sqrt(3) / 2
    for k in range(6):
        ka = k * np.pi / 3
        val = abs(lx * np.cos(ka) + ly * np.sin(ka))
        if val > limit + eps:
            return False
    return True

def _hexagon_contains_point(px, py, cx, cy, angle_deg, side=1.0, eps=1e-9):
    """Same as point_in_hex but for outer hexagon."""
    return _point_in_hex(px, py, cx, cy, angle_deg, side, eps)

def _hex_disjoint(h1, h2, eps=1e-9):
    """Check if two unit hexagons are disjoint (no overlap).
    h1, h2: (cx, cy, angle_deg)
    """
    cx1, cy1, a1 = h1
    cx2, cy2, a2 = h2
    # Quick reject: if center distance > 2*sqrt(3) (max distance between two points in unit hex),
    # actually the max distance in a unit hex is 2 (diameter = 2*R = 2).
    # So if dist > 2 + 2 = 4, definitely disjoint. But let's use a tighter bound.
    # The inradius of a unit hex is sqrt(3)/2. If center distance > 2*inradius = sqrt(3),
    # they might still overlap if aligned. Let's just do vertex check.
    dist = np.hypot(cx1 - cx2, cy1 - cy2)
    if dist > 2.0 + 2.0 + eps:  # max circumradius sum
        return True
    # Check if any vertex of h1 is inside h2, or vice versa
    v1 = _hex_vertices(cx1, cy1, a1)
    v2 = _hex_vertices(cx2, cy2, a2)
    for i in range(6):
        if _point_in_hex(v1[i, 0], v1[i, 1], cx2, cy2, a2, 1.0, eps):
            return False
    for i in range(6):
        if _point_in_hex(v2[i, 0], v2[i, 1], cx1, cy1, a1, 1.0, eps):
            return False
    return True

def _all_inside_outer(inner, outer_center, outer_side, outer_angle, eps=1e-9):
    """Check all inner hexagons are inside the outer hexagon."""
    ocx, ocy = outer_center
    for i in range(len(inner)):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, ang)
        for j in range(6):
            if not _point_in_hex(verts[j, 0], verts[j, 1], ocx, ocy, outer_angle, outer_side, eps):
                return False
    return True

def _all_disjoint(inner, eps=1e-9):
    """Check all pairs of inner hexagons are disjoint."""
    n = len(inner)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hex_disjoint(inner[i], inner[j], eps):
                return False
    return True

def _is_valid(inner, outer_center, outer_side, outer_angle, eps=1e-9):
    """Full validity check."""
    if not _all_inside_outer(inner, outer_center, outer_side, outer_angle, eps):
        return False
    if not _all_disjoint(inner, eps):
        return False
    return True

def _build_initial_config(n=N):
    """Build a reasonable initial configuration: hexagonal lattice arrangement."""
    # Arrange in a hexagonal pattern
    # For n hexagons, use a centered hexagonal number arrangement
    # 1, 7, 19, 37, ... for k=0,1,2,3
    # For n=11, we can use a 3x4 grid or similar
    config = []
    # Use a rectangular grid with hexagonal offsets
    cols = 3
    rows = (n + cols - 1) // cols
    spacing_x = 1.75  # slightly more than sqrt(3) to avoid overlap
    spacing_y = 1.5
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= n:
                break
            x = c * spacing_x
            y = r * spacing_y
            if r % 2 == 1:
                x += spacing_x / 2
            config.append([x, y, 0.0])
            idx += 1
    # Center the configuration
    if config:
        xs = [p[0] for p in config]
        ys = [p[1] for p in config]
        cx0 = (min(xs) + max(xs)) / 2
        cy0 = (min(ys) + max(ys)) / 2
        for p in config:
            p[0] -= cx0
            p[1] -= cy0
    return config

def construct():
    start_time = time.time()
    best_score = float('inf')
    best_result = None
    
    # Start with a reasonable initial config
    inner = _build_initial_config(N)
    
    # Initial outer hexagon: estimate needed size
    xs = [p[0] for p in inner]
    ys = [p[1] for p in inner]
    margin = 2.0  # buffer for hexagon extent
    outer_side_init = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + margin + 1.0
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Check initial validity
    if _is_valid(inner, outer_center, outer_side_init, outer_angle):
        best_score = outer_side_init
        best_result = (inner, outer_center, outer_side_init, outer_angle)
    
    # Try to shrink the outer hexagon using optimization
    # Variables: inner positions (2N), inner angles (N), outer center (2), outer side (1)
    # Objective: minimize outer_side subject to constraints
    
    # Use a simpler approach: iteratively shrink outer side and solve for inner positions
    # that fit inside
    
    # Alternative: use least squares to find a valid tight packing
    # Let's try a grid search over outer side with relaxation
    
    def _relax(inner_config, outer_center, outer_side, outer_angle, max_iter=200, tol=1e-6):
        """Relax positions to minimize penetration into outer and between inner hexes."""
        n = len(inner_config)
        positions = np.array([[p[0], p[1]] for p in inner_config], dtype=float)
        angles = np.array([p[2] for p in inner_config], dtype=float)
        
        for it in range(max_iter):
            if time.time() - start_time > DEADLINE - 5:
                break
            grad_x = np.zeros(n)
            grad_y = np.zeros(n)
            
            for i in range(n):
                cx, cy = positions[i]
                ang = angles[i]
                verts = _hex_vertices(cx, cy, ang)
                
                # Force from outer boundary
                for j in range(6):
                    vx, vy = verts[j]
                    # Check if vertex is outside outer hex
                    if not _point_in_hex(vx, vy, outer_center[0], outer_center[1], outer_angle, outer_side, 0):
                        # Push toward center
                        dx = outer_center[0] - vx
                        dy = outer_center[1] - vy
                        d = np.hypot(dx, dy)
                        if d > 1e-9:
                            grad_x[i] += dx / d
                            grad_y[i] += dy / d
                
                # Force from other hexagons
                for k in range(n):
                    if k == i:
                        continue
                    cxk, cyk = positions[k]
                    angk = angles[k]
                    dist = np.hypot(cx - cxk, cy - cyk)
                    if dist < 2.0 + 0.1:  # potential overlap
                        # Check for actual overlap
                        h1 = (cx, cy, ang)
                        h2 = (cxk, cyk, angk)
                        if not _hex_disjoint(h1, h2, 0):
                            # Push apart
                            dx = cx - cxk
                            dy = cy - cyk
                            d = np.hypot(dx, dy)
                            if d > 1e-9:
                                grad_x[i] += dx / d * 2
                                grad_y[i] += dy / d * 2
            
            # Update positions
            step = 0.1
            for i in range(n):
                positions[i, 0] += step * grad_x[i]
                positions[i, 1] += step * grad_y[i]
            
            # Build new config
            new_config = []
            for i in range(n):
                new_config.append([positions[i, 0], positions[i, 1], angles[i]])
            
            # Check if valid
            if _is_valid(new_config, outer_center, outer_side, outer_angle):
                return new_config
            
            # Reduce step if not improving
            step *= 0.95
        
        return inner_config
    
    # Strategy: start with large outer, shrink gradually
    current_side = outer_side_init
    current_inner = [list(p) for p in inner]
    current_center = outer_center
    current_angle = outer_angle
    
    shrink_factor = 0.95
    min_side = 2.5  # theoretical minimum is around 2*sqrt(3)/2 + something
    
    while time.time() - start_time < DEADLINE - 10:
        new_side = current_side * shrink_factor
        if new_side < min_side:
            break
        
        # Try to relax inner config to fit in smaller outer
        relaxed = _relax(current_inner, current_center, new_side, current_angle, max_iter=100)
        
        if _is_valid(relaxed, current_center, new_side, current_angle):
            current_inner = relaxed
            current_side = new_side
            if new_side < best_score:
                best_score = new_side
                best_result = (current_inner, current_center, current_side, current_angle)
        else:
            # Try with small perturbation
            perturbed = [list(p) for p in current_inner]
            for p in perturbed:
                p[0] += np.random.uniform(-0.1, 0.1)
                p[1] += np.random.uniform(-0.1, 0.1)
            relaxed2 = _relax(perturbed, current_center, new_side, current_angle, max_iter=100)
            if _is_valid(relaxed2, current_center, new_side, current_angle):
                current_inner = relaxed2
                current_side = new_side
                if new_side < best_score:
                    best_score = new_side
                    best_result = (current_inner, current_center, current_side, current_angle)
            else:
                # Can't shrink further, try smaller shrink factor
                shrink_factor *= 0.9
                if shrink_factor < 0.99:
                    break
    
    # Also try optimizing the outer angle and center
    if best_result is None:
        # Fallback: return initial if valid
        if _is_valid(inner, outer_center, outer_side_init, outer_angle):
            best_result = (inner, outer_center, outer_side_init, outer_angle)
        else:
            # Make it valid by increasing side
            outer_side_init += 5.0
            best_result = (inner, outer_center, outer_side_init, outer_angle)
    
    # Final check
    if not _is_valid(best_result[0], best_result[1], best_result[2], best_result[3]):
        # Increase side until valid
        side = best_result[2]
        for _ in range(100):
            side += 0.5
            if _is_valid(best_result[0], best_result[1], side, best_result[3]):
                best_result = (best_result[0], best_result[1], side, best_result[3])
                break
    
    return best_result[0], best_result[1], best_result[2], best_result[3]
