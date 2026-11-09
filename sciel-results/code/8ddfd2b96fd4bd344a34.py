"""Hexagon packing: place 11 unit hexagons in a minimal regular hexagon."""
import math
import time

N = 11
DEADLINE = 145.0  # safety margin before 150s kill

def _hex_verts(cx, cy, angle_deg, r=1.0):
    """Vertices of a regular hexagon (circumradius r) centered at (cx,cy)."""
    a = math.radians(angle_deg)
    return [(cx + r * math.cos(a + k * math.pi / 3.0),
             cy + r * math.sin(a + k * math.pi / 3.0)) for k in range(6)]

def _seg_intersect(p1, p2, p3, p4):
    """Return True if segment p1-p2 properly intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def _point_in_hex(px, py, verts):
    """Ray-casting point in polygon."""
    inside = False
    n = len(verts)
    j = n - 1
    for i in range(n):
        xi, yi = verts[i]
        xj, yj = verts[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def _overlap_dist(v1, v2):
    """
    Approximate minimum distance between two hexagons (as polygons).
    Returns (distance, push_x, push_y) where distance < 0 means overlap.
    Uses closest-point-on-segment approach for all edge pairs.
    """
    # First check if one center is inside the other
    c1 = ((v1[0][0]+v1[2][0]+v1[4][0])/3.0, (v1[0][1]+v1[2][1]+v1[4][1])/3.0)
    c2 = ((v2[0][0]+v2[2][0]+v2[4][0])/3.0, (v2[0][1]+v2[2][1]+v2[4][1])/3.0)
    if _point_in_hex(c1[0], c1[1], v2):
        return (-1.0, 0.0, 0.0)  # deep overlap
    if _point_in_hex(c2[0], c2[1], v1):
        return (-1.0, 0.0, 0.0)

    # Check edge-edge intersections
    n1, n2 = len(v1), len(v2)
    for i in range(n1):
        for j in range(n2):
            if _seg_intersect(v1[i], v1[(i+1)%n1], v2[j], v2[(j+1)%n2]):
                return (-1.0, 0.0, 0.0)

    # Find minimum distance between edges
    min_d = float('inf')
    min_px, min_py = 0.0, 0.0
    for i in range(n1):
        a1, a2 = v1[i], v1[(i+1)%n1]
        for j in range(n2):
            b1, b2 = v2[j], v2[(j+1)%n2]
            # Closest points between two segments
            d = _seg_seg_dist(a1, a2, b1, b2)
            if d < min_d:
                min_d = d
                # Direction from mid1 to mid2 for push
                m1x, m1y = (a1[0]+a2[0])/2, (a1[1]+a2[1])/2
                m2x, m2y = (b1[0]+b2[0])/2, (b1[1]+b2[1])/2
                dx, dy = m2x - m1x, m2y - m1y
                ln = math.sqrt(dx*dx + dy*dy)
                if ln > 1e-12:
                    min_px, min_py = dx/ln, dy/ln
    return (min_d, min_px, min_py)

def _seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between two segments."""
    d12 = (p2[0]-p1[0])**2 + (p2[1]-p1[1])**2
    d34 = (p4[0]-p3[0])**2 + (p4[1]-p3[1])**2
    d13 = (p3[0]-p1[0])**2 + (p3[1]-p1[1])**2
    d14 = (p4[0]-p1[0])**2 + (p4[1]-p1[1])**2
    d23 = (p3[0]-p2[0])**2 + (p3[1]-p2[1])**2
    d24 = (p4[0]-p2[0])**2 + (p4[1]-p2[1])**2
    if d12 < 1e-12 or d34 < 1e-12:
        return min(d13, d14, d23, d24) ** 0.5
    # Project
    def clamp(t): return max(0.0, min(1.0, t))
    # Parametric
    dx12 = p2[0]-p1[0]; dy12 = p2[1]-p1[1]
    dx34 = p4[0]-p3[0]; dy34 = p4[1]-p3[1]
    # Solve for closest points
    t = ((p3[0]-p1[0])*dx12 + (p3[1]-p1[1])*dy12) / d12
    t = clamp(t)
    cx1 = p1[0] + t*dx12; cy1 = p1[1] + t*dy12
    s = ((p1[0]-p3[0])*dx34 + (p1[1]-p3[1])*dy34) / d34
    s = clamp(s)
    cx2 = p3[0] + s*dx34; cy2 = p3[1] + s*dy34
    # Refine
    t = ((p3[0]-p1[0])*dx12 + (p3[1]-p1[1])*dy12) / d12
    t = clamp(t)
    cx1 = p1[0] + t*dx12; cy1 = p1[1] + t*dy12
    s = ((cx1-p3[0])*dx34 + (cy1-p3[1])*dy34) / d34
    s = clamp(s)
    cx2 = p3[0] + s*dx34; cy2 = p3[1] + s*dy34
    return ((cx1-cx2)**2 + (cy1-cy2)**2) ** 0.5

def _hex_boundary_dist(cx, cy, angle, side):
    """
    Check if a unit hexagon at (cx,cy,angle) fits inside a regular hexagon
    of side `side` centered at origin with 0 rotation.
    Returns (min_clearance, push_x, push_y) where clearance < 0 means violation.
    """
    hverts = _hex_verts(cx, cy, angle, 1.0)
    # Outer hexagon vertices (flat-top, centered at origin)
    outer = [(side * math.cos(k * math.pi/3.0), side * math.sin(k * math.pi/3.0)) for k in range(6)]
    
    # Check each vertex of inner hex is inside outer
    for vx, vy in hverts:
        if not _point_in_hex(vx, vy, outer):
            # Find closest point on outer boundary
            min_d = float('inf')
            px, py = 0.0, 0.0
            for i in range(6):
                a, b = outer[i], outer[(i+1)%6]
                d = _pt_seg_dist(vx, vy, a, b)
                if d < min_d:
                    min_d = d
                    px, py = a[0], a[1]
            # Push direction: from boundary point toward hex center
            dx, dy = cx - px, cy - py
            ln = math.sqrt(dx*dx + dy*dy)
            if ln > 1e-12:
                return (-min_d, dx/ln, dy/ln)
            return (-min_d, 0.0, 0.0)
    return (1.0, 0.0, 0.0)  # fully inside, positive clearance

def _pt_seg_dist(px, py, a, b):
    """Distance from point to segment, and closest point."""
    dx, dy = b[0]-a[0], b[1]-a[1]
    d2 = dx*dx + dy*dy
    if d2 < 1e-12:
        return math.sqrt((px-a[0])**2 + (py-a[1])**2)
    t = max(0.0, min(1.0, ((px-a[0])*dx + (py-a[1])*dy) / d2))
    cx, cy = a[0]+t*dx, a[1]+t*dy
    return math.sqrt((px-cx)**2 + (py-cy)**2)

def _initial_grid():
    """3-4-4 grid of hexagons, flat-top orientation."""
    hexes = []
    # 3 in row 0, 4 in row 1, 4 in row 2
    # Spacing: horizontal = 2*sqrt(3) for flat-top, vertical = 3
    sx = 2 * math.sqrt(3)
    sy = 3.0
    rows = [3, 4, 4]
    y_offs = [-sy, 0, sy]
    for ri, (count, yoff) in enumerate(zip(rows, y_offs)):
        x_start = -(count-1) * sx / 2.0
        for ci in range(count):
            x = x_start + ci * sx
            hexes.append([x, yoff, 0.0])
    return hexes

def construct():
    t0 = time.time()
    
    # Start from grid configuration
    hexes = _initial_grid()
    best_side = 40.0
    best_hexes = [h[:] for h in hexes]
    
    # Initial estimate of required side length
    # Bounding box of grid
    xs = [h[0] for h in hexes]
    ys = [h[1] for h in hexes]
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    # For flat-top hexagons, width = x_range + 2*sqrt(3), height = y_range + 3
    # Outer hex side needs to contain this
    # Conservative start
    side = max(x_range + 2*math.sqrt(3), y_range + 3) / math.sqrt(3) + 1.0
    side = max(side, 4.5)
    
    # Force-directed relaxation
    dt = 0.02
    damping = 0.9
    repel_strength = 2.0
    attract_strength = 0.05
    
    # We'll iteratively:
    # 1. Apply forces (repulsion between overlapping hexes, attraction to center)
    # 2. Integrate positions
    # 3. Check if all hexes fit in outer hexagon of given side
    # 4. Try to shrink side
    
    velocities = [[0.0, 0.0] for _ in range(N)]
    
    for iteration in range(5000):
        if time.time() - t0 > DEADLINE:
            break
        
        # Compute forces
        forces = [[0.0, 0.0] for _ in range(N)]
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ai = hexes[i]
                xj, yj, aj = hexes[j]
                dx = xj - xi
                dy = yj - yi
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 1e-6:
                    dx, dy = 0.01, 0.01
                    dist = math.sqrt(dx*dx + dy*dy)
                # For same orientation, minimum distance is 2.0
                min_dist = 2.0
                if dist < min_dist:
                    # Overlap - strong repulsion
                    overlap = min_dist - dist
                    f = repel_strength * overlap
                    forces[i][0] -= f * dx / dist
                    forces[i][1] -= f * dy / dist
                    forces[j][0] += f * dx / dist
                    forces[j][1] += f * dy / dist
                elif dist < min_dist * 1.5:
                    # Soft repulsion to keep them separated
                    f = 0.1 * (min_dist * 1.5 - dist)
                    forces[i][0] -= f * dx / dist
                    forces[i][1] -= f * dy / dist
                    forces[j][0] += f * dx / dist
                    forces[j][1] += f * dy / dist
        
        # Attraction to center (keeps them compact)
        for i in range(N):
            forces[i][0] -= attract_strength * hexes[i][0]
            forces[i][1] -= attract_strength * hexes[i][1]
        
        # Boundary constraint: push back if outside outer hexagon
        for i in range(N):
            clearance, px, py = _hex_boundary_dist(hexes[i][0], hexes[i][1], hexes[i][2], side)
            if clearance < 0:
                forces[i][0] += 5.0 * (-clearance) * px
                forces[i][1] += 5.0 * (-clearance) * py
        
        # Integrate
        for i in range(N):
            velocities[i][0] = (velocities[i][0] + forces[i][0] * dt) * damping
            velocities[i][1] = (velocities[i][1] + forces[i][1] * dt) * damping
            hexes[i][0] += velocities[i][0] * dt
            hexes[i][1] += velocities[i][1] * dt
        
        # Every 100 iterations, try to shrink the outer hexagon
        if iteration % 100 == 99 and iteration > 0:
            # Check if current configuration fits
            max_violation = 0.0
            for i in range(N):
                clearance, _, _ = _hex_boundary_dist(hexes[i][0], hexes[i][1], hexes[i][2], side)
                if clearance < max_violation:
                    max_violation = clearance
            if max_violation >= 0:
                # All fit! Try to shrink
                new_side = side * 0.98
                # Check if still fits with smaller side
                fits = True
                for i in range(N):
                    clearance, _, _ = _hex_boundary_dist(hexes[i][0], hexes[i][1], hexes[i][2], new_side)
                    if clearance < 0:
                        fits = False
                        break
                if fits:
                    side = new_side
                    best_side = side
                    best_hexes = [h[:] for h in hexes]
                else:
                    # Binary search for max shrink
                    lo, hi = new_side, side
                    for _ in range(10):
                        mid = (lo + hi) / 2
                        fits = True
                        for i in range(N):
                            clearance, _, _ = _hex_boundary_dist(hexes[i][0], hexes[i][1], hexes[i][2], mid)
                            if clearance < 0:
                                fits = False
                                break
                        if fits:
                            lo = mid
                        else:
                            hi = mid
                    side = lo
                    best_side = side
                    best_hexes = [h[:] for h in hexes]
    
    # Final verification and return
    # Use the best configuration found
    if best_hexes:
        hexes = best_hexes
    
    # Final fit check
    for i in range(N):
        clearance, _, _ = _hex_boundary_dist(hexes[i][0], hexes[i][1], hexes[i][2], best_side)
        if clearance < 0:
            # Expand slightly
            best_side *= 1.01
    
    inner = [[hexes[i][0], hexes[i][1], hexes[i][2]] for i in range(N)]
    return inner, (0.0, 0.0), best_side, 0.0
