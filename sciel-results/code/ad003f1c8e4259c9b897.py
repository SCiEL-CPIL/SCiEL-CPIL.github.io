import numpy as np
import time

N = 11

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy)."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return cx + side * np.cos(angles), cy + side * np.sin(angles)

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point is inside hexagon."""
    a = np.radians(angle_deg)
    dx, dy = px - cx, py - cy
    rx = dx * np.cos(a) + dy * np.sin(a)
    ry = -dx * np.sin(a) + dy * np.cos(a)
    apothem = side * np.sqrt(3) / 2
    for k in range(6):
        theta = np.radians(30 + k * 60)
        if rx * np.cos(theta) + ry * np.sin(theta) > apothem + 1e-10:
            return False
    return True

def _hex_in_hex(cx1, cy1, a1, cx2, cy2, a2, s1, s2):
    """Check if hex2 (side s2) is inside hex1 (side s1)."""
    vx, vy = _hex_vertices(cx2, cy2, s2, a2)
    for i in range(6):
        if not _point_in_hex(vx[i], vy[i], cx1, cy1, s1, a1):
            return False
    return True

def _hexes_disjoint(cx1, cy1, a1, cx2, cy2, a2, s=1.0):
    """Check if two unit hexagons are disjoint."""
    d = np.hypot(cx1 - cx2, cy1 - cy2)
    if d > 2.0 * s + 1e-9:
        return True
    vx1, vy1 = _hex_vertices(cx1, cy1, s, a1)
    vx2, vy2 = _hex_vertices(cx2, cy2, s, a2)
    for i in range(6):
        if _point_in_hex(vx1[i], vy1[i], cx2, cy2, s, a2):
            return False
    for i in range(6):
        if _point_in_hex(vx2[i], vy2[i], cx1, cy1, s, a1):
            return False
    return True

def _min_outer_side(positions, angles, center, outer_angle):
    """Minimum outer hex side to contain all inner hexes."""
    max_s = 0.0
    a_rad = np.radians(outer_angle)
    for i in range(len(positions)):
        vx, vy = _hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
        for j in range(6):
            dx, dy = vx[j] - center[0], vy[j] - center[1]
            rx = dx * np.cos(a_rad) + dy * np.sin(a_rad)
            ry = -dx * np.sin(a_rad) + dy * np.cos(a_rad)
            for k in range(6):
                theta = np.radians(30 + k * 60)
                proj = rx * np.cos(theta) + ry * np.sin(theta)
                s_req = 2.0 * abs(proj) / np.sqrt(3)
                if s_req > max_s:
                    max_s = s_req
    return max_s

def _is_valid(positions, angles, center, outer_angle, side):
    """Check validity: all inner hexes inside outer, and disjoint from each other."""
    # Check containment
    for i in range(len(positions)):
        if not _hex_in_hex(center[0], center[1], outer_angle, positions[i][0], positions[i][1], angles[i], side, 1.0):
            return False
    # Check disjointness
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if not _hexes_disjoint(positions[i][0], positions[i][1], angles[i],
                                   positions[j][0], positions[j][1], angles[j]):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    sqrt3 = np.sqrt(3)
    
    # Hexagonal lattice packing: 11 hexagons in a compact hexagonal arrangement
    # For flat-top hexagons of side 1, edge-adjacent centers are sqrt(3) apart.
    # A 3-3-3-2 arrangement in hex grid:
    
    # Let's define positions in a hexagonal coordinate system
    # Using axial coordinates (q, r) for hex grid
    # Convert to Cartesian: x = sqrt(3)*(q + r/2), y = 1.5*r
    
    hex_coords = [
        (0, 0), (1, 0), (2, 0),      # row 0: 3 hexes
        (0, 1), (1, 1), (2, 1),      # row 1: 3 hexes  
        (0, 2), (1, 2), (2, 2),      # row 2: 3 hexes
        (0, 3), (1, 3),              # row 3: 2 hexes
    ]
    
    positions = []
    for q, r in hex_coords:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        positions.append((x, y))
    
    # Center the arrangement
    cx_mean = np.mean([p[0] for p in positions])
    cy_mean = np.mean([p[1] for p in positions])
    positions = [(p[0] - cx_mean, p[1] - cy_mean) for p in positions]
    
    angles = [0.0] * N
    center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Compute required outer side
    side = _min_outer_side(positions, angles, center, outer_angle)
    
    # Verify validity
    if _is_valid(positions, angles, center, outer_angle, side + 1e-9):
        inner = [[float(p[0]), float(p[1]), float(a)] for p, a in zip(positions, angles)]
        return inner, (0.0, 0.0), float(side + 1e-6), float(outer_angle)
    
    # Fallback: try different arrangements
    # Try pointy-top arrangement
    hex_coords2 = [
        (0, 0), (1, 0), (2, 0),
        (-0.5, 1), (0.5, 1), (1.5, 1),
        (0, 2), (1, 2), (2, 2),
        (0.5, 3), (1.5, 3),
    ]
    positions2 = []
    for q, r in hex_coords2:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        positions2.append((x, y))
    cx_mean = np.mean([p[0] for p in positions2])
    cy_mean = np.mean([p[1] for p in positions2])
    positions2 = [(p[0] - cx_mean, p[1] - cy_mean) for p in positions2]
    
    side2 = _min_outer_side(positions2, angles, center, outer_angle)
    if _is_valid(positions2, angles, center, outer_angle, side2 + 1e-9):
        inner = [[float(p[0]), float(p[1]), float(a)] for p, a in zip(positions2, angles)]
        return inner, (0.0, 0.0), float(side2 + 1e-6), float(outer_angle)
    
    # Return best found
    best_side = min(side, side2)
    best_positions = positions if side <= side2 else positions2
    inner = [[float(p[0]), float(p[1]), float(a)] for p, a in zip(best_positions, angles)]
    return inner, (0.0, 0.0), float(best_side + 1e-6), float(outer_angle)
