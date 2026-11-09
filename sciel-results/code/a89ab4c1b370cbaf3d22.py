# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar


def _hex_vertices(cx, cy, side, angle_deg):
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))


def hexagon_packing_12():
    """
    Packs 12 unit regular hexagons into the smallest possible regular hexagon.
    Returns:
        inner: (12, 3) array of (x, y, angle_deg)
        outer: (3,) array of (x, y, angle_deg)
        s: outer hexagon side length
    """
    # Define 12 hexagon centers in a hexagonal lattice arrangement
    # This gives a compact cluster of 12 hexagons
    sqrt3 = np.sqrt(3)
    # Hexagonal lattice with spacing 2 (so hexagons of side 1 just touch)
    # Centers:
    #   Row 0: 3 hexagons
    #   Row 1: 4 hexagons  
    #   Row 2: 3 hexagons
    #   Row 3: 2 hexagons
    # Total: 12
    
    # Let's use a more standard compact arrangement:
    # A hexagonal number arrangement isn't possible for 12, so we use a 
    # rectangular-ish cluster that's compact.
    
    # Actually, let me use a proven compact arrangement:
    # Place hexagons in a pattern that minimizes the bounding hexagon.
    # Use a 3-4-3-2 arrangement or similar.
    
    # Let me try a different approach: use a hexagonal lattice and pick 12 
    # centers that form a compact shape.
    
    # Standard hexagonal lattice points (for hexagons of side 1, centers are 
    # 2 units apart horizontally, and sqrt(3) apart vertically)
    
    # Let me define centers manually for a compact 12-hexagon cluster:
    # Think of it as a hexagon with 3 rings minus some, or a compact blob.
    
    # Compact arrangement: 
    # Center at origin, then 6 around it, then 5 more in a ring but not complete
    # That's 1 + 6 + 5 = 12
    
    # Actually, let me use a cleaner approach: a 4x3 grid with hexagonal offset
    # But that might not be optimal.
    
    # Let me just use the classic "hexagonal packing" of 12:
    # Arrange in rows: 2, 3, 3, 2, 2 or similar.
    
    # For now, let me use a simple but compact arrangement and optimize the outer hex.
    
    # Hexagonal lattice: 
    # x = 2 * (i + j/2), y = sqrt(3) * j
    # for integer i, j
    
    # Let me pick 12 points that form a compact cluster:
    centers = []
    # Row j=0: i = -1, 0, 1 (3 hexagons)
    for i in [-1, 0, 1]:
        centers.append((2 * i, 0))
    # Row j=1: i = -1, 0, 1, 2 (4 hexagons) -- wait, this might be too wide
    # Let me reconsider.
    
    # Better: use a hexagonal number pattern. For 12, it's not a perfect hexagon.
    # Let me use: 
    # j=0: i=0 (1)
    # j=1: i=-1,0,1 (3)  
    # j=2: i=-1,0,1,2 (4)
    # j=3: i=-1,0,1 (3)
    # j=4: i=0 (1)
    # Total: 1+3+4+3+1 = 12
    
    centers = []
    # j=0: center
    centers.append((0, 0))
    # j=1: 3 hexagons
    for i in [-1, 0, 1]:
        x = 2 * (i + 1/2)
        y = sqrt3
        centers.append((x, y))
    # j=2: 4 hexagons  
    for i in [-1, 0, 1, 2]:
        x = 2 * (i + 2/2)
        y = 2 * sqrt3
        centers.append((x, y))
    # j=3: 3 hexagons
    for i in [-1, 0, 1]:
        x = 2 * (i + 1/2)
        y = 3 * sqrt3
        centers.append((x, y))
    # j=4: 1 hexagon
    centers.append((0, 4 * sqrt3))
    
    # Hmm, this is getting complicated. Let me just use a simple verified arrangement.
    
    # Let me use the arrangement from the seed but make it more compact,
    # then optimize the outer hexagon rotation.
    
    # Actually, let me start with a known compact configuration:
    # 12 hexagons in a 3-3-3-3 grid (rectangular with hex offset)
    
    # For simplicity, let me use centers that I know work:
    # A 4x3 arrangement with hexagonal offset
    centers = []
    for row in range(3):
        for col in range(4):
            x = col * 2 + (row % 2) * 1
            y = row * sqrt3
            centers.append((x, y))
    
    # That's 12 centers. Let me center them.
    centers = np.array(centers)
    centers -= np.mean(centers, axis=0)
    
    # Now, find the optimal outer hexagon
    # The outer hexagon has center at origin, angle theta, side s
    # We need to find theta and s such that all inner hexagons fit inside.
    
    def required_s(theta_rad):
        """Given outer hexagon angle, compute minimum side length needed."""
        # Outer hexagon vertices
        outer_angles = theta_rad + np.deg2rad(60 * np.arange(6))
        # For a regular hexagon with side s and center at origin,
        # the vertices are at distance s from center (since for regular hexagon, 
        # side length = radius)
        # Wait, no. For a regular hexagon, the distance from center to vertex 
        # equals the side length. So vertices are at distance s.
        
        # Actually, let me reconsider. The task says "side d" and vertices are 
        # at distance d from center. So for outer hex with side s, vertices 
        # are at distance s from center.
        
        # The constraint is that all inner hexagon vertices must be inside 
        # the outer hexagon.
        
        # For a regular hexagon with vertices at distance s, the apothem 
        # (distance from center to edge) is s * cos(pi/6) = s * sqrt(3)/2.
        
        # To check if a point (px, py) is inside the hexagon, we can use 
        # the fact that the hexagon is the intersection of 6 half-planes.
        # Each edge has a normal direction.
        
        # The 6 edges have normals at angles: theta + pi/6 + k*pi/3, k=0..5
        # Wait, let me think again.
        
        # Vertices at angles: theta + k*60 deg, k=0..5, at distance s
        # Edge between vertex k and k+1 has midpoint at angle theta + (2k+1)*30 deg
        # The normal to this edge points outward at angle theta + (2k+1)*30 deg
        # The distance from center to this edge is the apothem = s * cos(30 deg) = s * sqrt(3)/2
        
        # So for a point P to be inside, for each of the 6 edges, 
        # the projection of P onto the outward normal must be <= apothem.
        
        normals = theta_rad + np.deg2rad(30 + 60 * np.arange(6))
        apothem = s * np.sqrt(3) / 2 if False else None  # We're solving for s
        
        # For each inner hexagon, get all 6 vertices
        # For each vertex, compute max projection onto each normal
        # s must satisfy: max_proj <= s * sqrt(3)/2
        # So s >= max_proj * 2/sqrt(3)
        
        all_max_projs = []
        for cx, cy in centers:
            verts = _hex_vertices(cx, cy, 1, 0)  # angle 0 for inner hexagons
            for px, py in verts:
                for n_angle in normals:
                    proj = px * np.cos(n_angle) + py * np.sin(n_angle)
                    all_max_projs.append(proj)
        
        max_proj = max(all_max_projs)
        # s * sqrt(3)/2 >= max_proj
        # s >= max_proj * 2/sqrt(3)
        return max_proj * 2 / sqrt3
    
    # Minimize required_s over theta in [0, 60 degrees)
    result = minimize_scalar(
        lambda theta_deg: required_s(np.deg2rad(theta_deg)),
        bounds=(0, 60),
        method='bounded',
        options={'xatol': 1e-10}
    )
    
    optimal_theta_deg = result.x
    optimal_s = result.fun
    
    # Build inner hexagon data
    inner_hex_data = np.zeros((12, 3))
    for i, (cx, cy) in enumerate(centers):
        inner_hex_data[i] = [cx, cy, 0]
    
    # Build outer hexagon data
    outer_hex_data = np.array([0, 0, optimal_theta_deg])
    
    return inner_hex_data, outer_hex_data, optimal_s


# EVOLVE-BLOCK-END
