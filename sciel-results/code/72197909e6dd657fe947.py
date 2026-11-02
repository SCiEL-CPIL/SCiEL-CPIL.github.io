import numpy as np
import time

N = 11

def _hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle + i * np.pi / 3
        verts.append((cx + side * np.cos(a), cy + side * np.sin(a)))
    return np.array(verts)

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with given side and rotation."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    angle = np.radians(angle_deg)
    # Rotate point into hexagon frame
    ca, sa = np.cos(-angle), np.sin(-angle)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    # Hexagon with flat-top orientation (vertex at angle 0) has vertices at angles 0,60,...,300
    # The 6 supporting lines: for each edge, check distance
    # For a regular hexagon with circumradius R=side, the 6 half-planes are:
    # cos(60*i)*x + sin(60*i)*y <= R * cos(30) = side * sqrt(3)/2 for i=0..5
    # Actually, the apothem is side * sqrt(3)/2
    apothem = side * np.sqrt(3) / 2
    for i in range(6):
        a = i * np.pi / 3
        if lx * np.cos(a) + ly * np.sin(a) > apothem + 1e-9:
            return False
    return True

def _hex_fits_in_hex(inner_cx, inner_cy, inner_angle, outer_cx, outer_cy, outer_side, outer_angle):
    """Check if inner unit hexagon fits inside outer hexagon."""
    # Check all 6 vertices of inner hexagon are inside outer hexagon
    inner_verts = _hexagon_vertices(inner_cx, inner_cy, 1.0, inner_angle)
    for vx, vy in inner_verts:
        if not _point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_side, outer_angle):
            return False
    return True

def _generate_hex_lattice_positions(n):
    """Generate n positions on a hexagonal lattice, centered at origin."""
    # Hexagonal numbering: center + rings
    # Ring 0: 1 point
    # Ring k: 6k points
    # Total up to ring r: 1 + 3r(r+1)
    positions = [(0.0, 0.0)]
    if n <= 1:
        return positions[:n]
    
    # Lattice vectors for hexagonal packing of unit hexagons
    # Distance between adjacent centers: sqrt(3) for side-1 hexagons (touching)
    a = np.sqrt(3)
    # Basis vectors
    e1 = np.array([a, 0.0])
    e2 = np.array([a/2, 3*a/2])  # This gives 60 degree angle... let me reconsider
    
    # For regular hexagons of side 1 packed densely:
    # The center-to-center distance for touching hexagons is 2 * apothem = 2 * sqrt(3)/2 = sqrt(3)
    # Lattice: axial coordinates. Neighbors at 6 directions.
    # Let's use: basis1 = (sqrt(3), 0), basis2 = (sqrt(3)/2, 3/2)
    # Wait, the y-spacing: for hexagons with flat top, vertical spacing is 3/2 * side = 3/2
    # Horizontal spacing is sqrt(3) * side = sqrt(3)
    # Offset rows shift by sqrt(3)/2
    
    # Simpler: generate all lattice points within a bounding box, sort by distance from center
    b1 = np.array([np.sqrt(3), 0.0])
    b2 = np.array([np.sqrt(3)/2, 1.5])  # 1.5 = 3/2
    
    # Generate candidates
    candidates = []
    max_ring = 10
    for i in range(-max_ring, max_ring + 1):
        for j in range(-max_ring, max_ring + 1):
            pos = i * b1 + j * b2
            candidates.append((float(pos[0]), float(pos[1])))
    
    # Sort by distance from origin
    candidates.sort(key=lambda p: p[0]*p[0] + p[1]*p[1])
    return candidates[:n]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Generate inner hexagon positions on hexagonal lattice
    inner_positions = _generate_hex_lattice_positions(N)
    
    # For each inner hexagon, we can rotate it. Try a few angles.
    # The optimal rotation for fitting in a hexagonal container is likely 0 or 30 degrees.
    inner_angles = [0.0, 30.0, 15.0, 45.0]
    
    # Binary search on outer side length
    lo = 1.0
    hi = 50.0
    
    # First check if hi is feasible
    def feasible(S, outer_angle=0.0):
        for (ix, iy), angle in zip(inner_positions, [0.0]*N):
            if not _hex_fits_in_hex(ix, iy, angle, 0.0, 0.0, S, outer_angle):
                return False
        return True
    
    # Try a few outer rotations
    best_side = hi
    best_config = None
    
    for outer_angle in [0.0, 30.0, 15.0, 45.0]:
        if time.time() > deadline:
            break
        
        lo2, hi2 = 1.0, hi
        # Check hi2
        if not feasible(hi2, outer_angle):
            # Try larger
            hi2 *= 2
            while not feasible(hi2, outer_angle) and hi2 < 100:
                hi2 *= 2
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        
        # Binary search
        for _ in range(60):
            if time.time() > deadline:
                break
            mid = (lo2 + hi2) / 2
            if feasible(mid, outer_angle):
                hi2 = mid
            else:
                lo2 = mid
        
        if hi2 < best_side:
            best_side = hi2
            best_config = outer_angle
    
    # Build result
    inner = [[x, y, 0.0] for (x, y) in inner_positions]
    outer_center = (0.0, 0.0)
    outer_side = best_side
    outer_angle = best_config if best_config is not None else 0.0
    
    return inner, outer_center, outer_side, outer_angle

