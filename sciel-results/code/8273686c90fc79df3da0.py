# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    using a hexagonal cluster topology (7 hexagons in a hex number pattern + 4 satellites).
    Optimizes the global rotation of the entire cluster to minimize the outer hexagon side length.
    """
    # Hexagonal cluster: 7 hexagons in a hexagonal number pattern (center + 6 around)
    # plus 4 satellites at alternating outer positions
    # Unit hexagons with side length 1, flat-top orientation (angle=0)
    # For flat-top hexagons, the distance between centers of adjacent hexagons is 2 * (sqrt(3)/2) = sqrt(3)
    # Actually for side length s=1, the distance between centers of adjacent hexagons is 2*s*sin(60) = sqrt(3)
    # Wait, let me recalculate. For two adjacent unit hexagons sharing an edge, the center-to-center distance is 2 * apothem = 2 * (sqrt(3)/2) * 1 = sqrt(3)
    
    s = 1.0  # side length of inner hexagons
    d = s * np.sqrt(3)  # center-to-center distance for adjacent hexagons
    
    # Hexagonal cluster positions (relative to center hexagon)
    # Center at (0,0)
    # 6 neighbors at angles 0, 60, 120, 180, 240, 300 degrees, distance d
    # For flat-top hexagons, the 6 neighbors are at:
    # (d, 0), (d/2, d*sqrt(3)/2), (-d/2, d*sqrt(3)/2), (-d, 0), (-d/2, -d*sqrt(3)/2), (d/2, -d*sqrt(3)/2)
    # Wait, for flat-top hexagons, the vertices are at angles 0, 60, 120, 180, 240, 300 from center
    # The neighbors share edges, so they are at the midpoints of the edges extended... 
    # Actually, for two hexagons sharing an edge, the center-to-center vector is perpendicular to the shared edge,
    # with length 2*apothem = sqrt(3) for unit side.
    # For flat-top hexagon, the edges are at angles 30, 90, 150, 210, 270, 330 (perpendicular to vertices at 0,60,...)
    # So neighbors are at angles 30, 90, 150, 210, 270, 330 degrees from center, distance sqrt(3)
    
    angles_neighbor = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180
    neighbor_offsets = np.column_stack([d * np.cos(angles_neighbor), d * np.sin(angles_neighbor)])
    
    # 7 hexagons: center + 6 neighbors
    cluster_positions = np.vstack([np.array([[0.0, 0.0]]), neighbor_offsets])
    
    # 4 satellites: place them at alternating outer positions
    # The outer ring of the 7-hexagon cluster has 6 "outer" positions where additional hexagons can fit
    # Let's place 4 satellites at positions that are 2*d away in alternating directions
    # For a hexagonal lattice, the next ring positions are at distance 2*d in the 6 neighbor directions,
    # plus 2*d in the 6 edge-normal directions (30, 90, ...)
    # Let's try placing satellites at 4 of the 6 "corner" positions of the hexagonal cluster
    # The corners of the 7-hexagon cluster are at the 6 neighbor positions extended by d more
    # Actually, let's place them at positions that are adjacent to 2 of the outer hexagons
    
    # For a hexagonal lattice, positions at "distance 2" from center in the lattice sense:
    # These are at 2*d in the 6 neighbor directions, and at d*sqrt(3) in the 6 edge-normal directions... 
    # Let me just use the standard hex lattice basis vectors
    # Basis: e1 = (d, 0), e2 = (d/2, d*sqrt(3)/2)
    # Lattice points: m*e1 + n*e2 for integers m, n
    # The 7-hexagon cluster uses (0,0), (1,0), (0,1), (-1,0), (0,-1), (1,-1), (-1,1)
    # Wait, that's not right. Let me use axial coordinates.
    
    # Let me use a simpler approach: the 11 hexagons in a compact arrangement
    # Use the known optimal or near-optimal packing for 11 hexagons
    # A common arrangement is a 3-4-3 pattern or similar
    
    # Let me define positions using a hex lattice with basis vectors
    e1 = np.array([d, 0.0])
    e2 = np.array([d/2, d*np.sqrt(3)/2])
    
    # Axial coordinates (q, r) -> position = q*e1 + r*e2
    # 7-hexagon cluster (hex number 7): (0,0), (1,0), (0,1), (-1,0), (0,-1), (1,-1), (-1,1)
    # That's 7 positions. Add 4 more for 11 total.
    # Let's add: (2,-1), (2,0), (-2,1), (-2,0) or similar
    
    # Let me try a symmetric arrangement:
    # Row 1 (top): 3 hexagons
    # Row 2: 4 hexagons  
    # Row 3: 3 hexagons
    # Total: 10... need 11
    
    # Let me use: a 2-3-2-1-2-1 pattern or just optimize a good initial guess
    # For now, use the 7-cluster + 4 satellites at (2,0), (2,-1), (-2,0), (-2,1) in axial coords
    
    axial_coords = np.array([
        [0, 0],    # center
        [1, 0],    # right
        [0, 1],    # top-right
        [-1, 0],   # left
        [0, -1],   # bottom-left
        [1, -1],   # bottom-right
        [-1, 1],   # top-left
        [2, 0],    # satellite 1
        [2, -1],   # satellite 2
        [-2, 0],   # satellite 3
        [-2, 1],   # satellite 4
    ])
    
    # Convert axial to Cartesian
    positions = np.array([q*e1 + r*e2 for q, r in axial_coords])
    
    # All inner hexagons have angle 0 (flat-top)
    inner_hex_data = np.hstack([positions, np.zeros((11, 1))])
    
    # Function to compute the required outer hexagon side length for a given rotation
    def compute_outer_side(inner_positions, outer_angle_deg):
        """Compute minimum outer hexagon side length to contain all inner hexagons."""
        outer_angle = np.radians(outer_angle_deg)
        
        # Get vertices of each inner hexagon
        all_vertices = []
        for i in range(len(inner_positions)):
            cx, cy = inner_positions[i]
            for j in range(6):
                theta = outer_angle + np.radians(60 * j)  # inner hexagons also rotated? No, they're fixed
                # Actually, inner hexagons have fixed angle 0, so their vertices are at fixed angles
                theta_j = np.radians(60 * j)
                vx = cx + s * np.cos(theta_j)
                vy = cy + s * np.sin(theta_j)
                all_vertices.append([vx, vy])
        all_vertices = np.array(all_vertices)
        
        # For a regular hexagon with center at origin, side length L, angle a:
        # A point (x,y) is inside if for each of the 6 edge normals n_k, dot((x,y), n_k) <= L*sin(60) = L*sqrt(3)/2
        # The edge normals are at angles a + 30 + 60*k for k=0..5
        # Actually, for a regular hexagon with vertices at angles a, a+60, ..., the edges are perpendicular to directions a+30, a+90, ...
        # The constraint is: for each edge normal direction n_k, max over all vertices of dot(v, n_k) <= apothem = L*sqrt(3)/2
        # So L >= 2/sqrt(3) * max_k max_v dot(v, n_k)
        
        max_projection = 0.0
        for k in range(6):
            normal_angle = outer_angle + np.radians(30 + 60 * k)
            nx = np.cos(normal_angle)
            ny = np.sin(normal_angle)
            projections = all_vertices[:, 0] * nx + all_vertices[:, 1] * ny
            max_proj = np.max(projections)
            if max_proj > max_projection:
                max_projection = max_proj
        
        L = 2.0 / np.sqrt(3) * max_projection
        return L
    
    # Optimize the outer hexagon angle to minimize side length
    # Also consider rotating the entire inner cluster
    def objective(params):
        inner_rotation = params[0]
        outer_rotation = params[1]
        
        # Rotate inner positions
        cos_r = np.cos(inner_rotation)
        sin_r = np.sin(inner_rotation)
        rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        rotated_positions = positions @ rot_matrix.T
        
        return compute_outer_side(rotated_positions, np.degrees(outer_rotation))
    
    # Try multiple initial guesses
    best_side = np.inf
    best_inner = None
    best_outer = None
    
    for trial in range(20):
        init_inner_rot = np.random.uniform(0, 2*np.pi) if trial > 0 else 0.0
        init_outer_rot = np.random.uniform(0, np.pi/3) if trial > 0 else 0.0
        
        result = minimize(objective, [init_inner_rot, init_outer_rot], method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12})
        
        if result.fun < best_side:
            best_side = result.fun
            inner_rot = result.x[0]
            outer_rot = result.x[1]
    
    # Apply best rotation to inner positions
    cos_r = np.cos(inner_rot)
    sin_r = np.sin(inner_rot)
    rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    rotated_positions = positions @ rot_matrix.T
    
    inner_hex_data = np.hstack([rotated_positions, np.zeros((11, 1))])
    outer_hex_data = np.array([0.0, 0.0, np.degrees(outer_rot)])
    outer_hex_side_length = float(best_side)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
