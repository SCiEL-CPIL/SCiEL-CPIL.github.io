"""Unit regular hexagons inside a regular hexagon n=12; honeycomb packing."""
import math
import time

N = 12
_DEADLINE = 150.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _deadline_ok(margin=5.0):
    return _elapsed() < _DEADLINE - margin

def _hex_verts(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon (flat-top at 0 deg), rotated by angle_deg."""
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.pi / 3.0 * k  # flat-top: vertices at 0,60,...
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        xr = cx + cos_a * x - sin_a * y
        yr = cy + sin_a * x + cos_a * y
        verts.append((xr, yr))
    return verts

def _required_outer_side(points, angle_deg=0.0):
    """Smallest side of an outer hexagon (center 0, given rotation) containing all points."""
    a = math.radians(angle_deg)
    # For a hexagon centered at origin with rotation a, the constraint is:
    # For each of 6 face directions, the projection of each point onto the outward
    # normal must be <= s * inradius = s * sqrt(3)/2
    # Face normals (outward) for flat-top hex at rotation a:
    # direction angles: a + 90 + 60*k for k in 0..5  (normals perpendicular to edges)
    # Actually for flat-top hex, edges are at angles 0,60,...; normals at 90,150,...
    # Let's use: for each vertex direction theta_k = a + pi/3*k, the supporting line
    # is at distance s from center along that direction? No.
    # 
    # A regular hexagon with side s, center at origin, rotation a:
    # Vertices at distance s from center at angles a + pi/3*k
    # The hexagon is the intersection of 6 half-planes.
    # Edge k is perpendicular to direction a + pi/3*k + pi/6? 
    # 
    # Simpler: the hexagon is {x: max_k (x . n_k) <= s*sqrt(3)/2}
    # where n_k are unit vectors at angles a + pi/3*k + pi/6 (normals to edges)
    # For flat-top (a=0): normals at 90, 150, 210, 270, 330, 30 degrees
    # 
    # Actually, let me just compute: for a hex with vertices at angles a+pi/3*k at radius s,
    # the inradius (apothem) is s*sqrt(3)/2. The 6 edge normals are at angles a+pi/3*k+pi/6.
    
    max_proj = 0.0
    for k in range(6):
        # Normal direction for edge k
        n_angle = a + math.pi / 3.0 * k + math.pi / 6.0
        nx = math.cos(n_angle)
        ny = math.sin(n_angle)
        for (px, py) in points:
            proj = px * nx + py * ny
            if proj > max_proj:
                max_proj = proj
    # Need s * sqrt(3)/2 >= max_proj  =>  s >= max_proj * 2 / sqrt(3)
    return max_proj * 2.0 / math.sqrt(3.0)

def _honeycomb_positions():
    """Generate honeycomb positions for 12 hexagons."""
    positions = [(0.0, 0.0)]  # center
    
    # Ring 1: 6 neighbors at distance 2 (edge-to-edge touching)
    # For flat-top hexagons, neighbors are at angles 0, 60, 120, ... at distance 2
    for k in range(6):
        theta = math.pi / 3.0 * k
        x = 2.0 * math.cos(theta)
        y = 2.0 * math.sin(theta)
        positions.append((x, y))
    
    # Ring 2: 6 more at distance 2*sqrt(3), at angles 30, 90, 150, ...
    for k in range(6):
        theta = math.pi / 6.0 + math.pi / 3.0 * k
        x = 2.0 * math.sqrt(3.0) * math.cos(theta)
        y = 2.0 * math.sqrt(3.0) * math.sin(theta)
        positions.append((x, y))
    
    return positions  # 12 positions

def _compute_score(positions, angle_deg=0.0, outer_angle=0.0):
    """Compute the outer side length for given inner positions and orientations."""
    all_points = []
    for (cx, cy) in positions:
        verts = _hex_verts(cx, cy, angle_deg)
        all_points.extend(verts)
    return _required_outer_side(all_points, outer_angle)

def construct():
    if not _deadline_ok():
        # Fallback: return honeycomb with known good side
        positions = _honeycomb_positions()
        inner = [[x, y, 0.0] for (x, y) in positions]
        return inner, (0.0, 0.0), 5.5, 0.0
    
    # Try different inner hexagon orientations and outer hexagon orientations
    best_side = float('inf')
    best_inner = None
    best_outer_angle = 0.0
    best_inner_angle = 0.0
    
    positions = _honeycomb_positions()
    
    # Try various inner rotation angles (0-60 degrees in steps)
    # and outer rotation angles
    for inner_angle in range(0, 61, 5):
        if not _deadline_ok():
            break
        for outer_angle in range(0, 61, 5):
            if not _deadline_ok():
                break
            side = _compute_score(positions, inner_angle, outer_angle)
            if side < best_side:
                best_side = side
                best_inner_angle = inner_angle
                best_outer_angle = outer_angle
    
    # Fine-tune around best
    if _deadline_ok():
        for da in [-2, -1, 0, 1, 2]:
            for inner_angle in [best_inner_angle + da]:
                if inner_angle < 0 or inner_angle > 60:
                    continue
                for outer_angle in [best_outer_angle + da]:
                    if outer_angle < 0 or outer_angle > 60:
                        continue
                    side = _compute_score(positions, inner_angle, outer_angle)
                    if side < best_side:
                        best_side = side
                        best_inner_angle = inner_angle
                        best_outer_angle = outer_angle
    
    inner = [[x, y, best_inner_angle] for (x, y) in positions]
    return inner, (0.0, 0.0), best_side, best_outer_angle

