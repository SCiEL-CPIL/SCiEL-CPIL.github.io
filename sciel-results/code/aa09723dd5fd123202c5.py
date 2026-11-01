"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product, combinations
from math import sqrt, cos, sin, pi

N = 11

def hex_to_cartesian(q, r, s=1.0):
    """Convert axial coordinates to Cartesian for hex lattice with side length s."""
    # For a hexagon with flat top, axial to Cartesian
    x = s * (q + r / 2.0) * 2 * sqrt(3) / 2  # Actually let me use standard
    # Standard: for pointy-top hexagons
    # x = size * (sqrt(3) * q + sqrt(3)/2 * r)
    # y = size * (3/2 * r)
    # But for flat-top:
    # x = size * (3/2 * q)
    # y = size * (sqrt(3)/2 * q + sqrt(3) * r)
    # Let me use a consistent system. For unit hexagons (side=1), center-to-center distance = sqrt(3)
    # Using axial coords where neighbors are at (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    x = sqrt(3) * (q + r / 2.0)
    y = 1.5 * r
    return x, y

def cartesian_to_hex(x, y):
    """Convert Cartesian to axial coordinates."""
    q = (2.0 / 3.0 * x) / sqrt(3)  # This is getting messy, let me just work in Cartesian directly
    return None

def compute_enclosing_hexagon_side(points, angle_deg):
    """
    Given a set of points (centers of unit hexagons), compute the minimum side length
    of a regular hexagon (with the given rotation angle) that contains all points.
    The hexagon is centered at the centroid of the points (for simplicity, we can also optimize center).
    """
    angle = np.radians(angle_deg)
    cos_a = cos(angle)
    sin_a = sin(angle)
    
    # For a regular hexagon with side length s and rotation angle, the supporting lines
    # are at 6 directions. The hexagon contains a point if for each of the 6 normals,
    # the projection is within the apothem.
    
    # The 6 outward normals of a regular hexagon (flat-top, angle 0) are at angles:
    # 0, 60, 120, 180, 240, 300 degrees (for the edges)
    # Actually, for a regular hexagon, the 6 face normals are at angles 0, 60, 120, 180, 240, 300
    # if we define it appropriately.
    
    # Let's define the hexagon by its 6 face normals. For a regular hexagon with side s,
    # the apothem is s * sqrt(3)/2.
    
    # The 6 normal directions (unit vectors) for a hexagon rotated by angle:
    normals = []
    for i in range(6):
        theta = angle + i * pi / 3
        normals.append((cos(theta), sin(theta)))
    
    pts = np.array(points)
    # Try center at centroid
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    
    max_proj = 0
    for nx, ny in normals:
        projections = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
        max_proj = max(max_proj, abs(projections).max())
    
    # The apothem must be at least max_proj
    # Apothem = s * sqrt(3) / 2
    s = max_proj * 2.0 / sqrt(3)
    return s

def try_configuration(hex_centers, n_angles=12):
    """Try multiple rotations and return the best (min side length)."""
    best_s = float('inf')
    best_angle = 0
    for i in range(n_angles):
        angle = i * 360.0 / n_angles
        s = compute_enclosing_hexagon_side(hex_centers, angle)
        if s < best_s:
            best_s = s
            best_angle = angle
    return best_s, best_angle

def generate_candidate_configurations():
    """Generate several candidate configurations of 11 unit hexagons."""
    configs = []
    
    # Configuration 1: Hexagonal cluster with radius 1 (7 hexagons) + 4 more
    # Radius 1: center (0,0) and 6 neighbors
    base = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    # Add 4 more: extend in one direction
    # Add (2,0), (2,-1), (1,1), (0,2) or similar
    config1 = base + [(2,0), (2,-1), (1,1), (0,2)]
    configs.append(config1)
    
    # Configuration 2: Linear arrangement
    config2 = [(i, 0) for i in range(11)]
    configs.append(config2)
    
    # Configuration 3: Two rows of 5-6
    config3 = [(i, 0) for i in range(6)] + [(i, 1) for i in range(5)]
    configs.append(config3)
    
    # Configuration 4: More compact - center + 6 + 4 in a cross pattern
    config4 = base + [(2,0), (-2,0), (0,2), (0,-2)]
    configs.append(config4)
    
    # Configuration 5: 3-4-3-1 pattern
    config5 = [(0,0), (1,0), (2,0), (3,0),
               (0,1), (1,1), (2,1), (3,1),
               (0,2), (1,2), (2,2)]
    configs.append(config5)
    
    # Configuration 6: Compact blob
    config6 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
               (2,0), (2,-1), (1,1), (-1,2)]
    configs.append(config6)
    
    # Configuration 7: Hexagon with radius 1, then add 4 in a line
    config7 = base + [(2,0), (3,0), (4,0), (5,0)]
    configs.append(config7)
    
    # Configuration 8: 4-3-4 arrangement
    config8 = [(i,0) for i in range(4)] + [(i,1) for i in range(1,4)] + [(i,2) for i in range(4)]
    configs.append(config8)
    
    return configs

def construct():
    best_score = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_side = float('inf')
    best_angle = 0.0
    
    configs = generate_candidate_configurations()
    
    for config in configs:
        # Convert axial to Cartesian
        hex_centers = []
        for q, r in config:
            x, y = hex_to_cartesian(q, r)
            hex_centers.append((x, y))
        
        s, angle = try_configuration(hex_centers, n_angles=24)
        
        if s < best_score:
            best_score = s
            best_inner = [[x, y, angle] for x, y in hex_centers]
            # Center the outer hexagon at centroid
            cx = np.mean([p[0] for p in hex_centers])
            cy = np.mean([p[1] for p in hex_centers])
            best_outer_center = (cx, cy)
            best_side = s
            best_angle = angle
    
    if best_inner is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_outer_center, best_side, best_angle

# EVOLVE-BLOCK-END
