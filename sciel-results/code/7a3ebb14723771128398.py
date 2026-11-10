# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    n = 11

    # Start with a good initial configuration: honeycomb-like arrangement
    # Unit hexagons have circumradius = 1 (since side length = 1)
    # Distance between centers of touching hexagons in honeycomb = 2 * sqrt(3)/2 * 2 = 2*sqrt(3) ≈ 3.464? 
    # Actually, for regular hexagons with side s=1, the distance between centers of adjacent hexagons
    # in a honeycomb packing is 2 * (apothem) = 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.732
    # Wait, let me reconsider. A regular hexagon with side length 1 has:
    # - circumradius (center to vertex) = 1
    # - inradius (center to edge midpoint) = sqrt(3)/2
    # When two hexagons share an edge, the distance between centers = 2 * inradius = sqrt(3)
    
    # Let's use a compact arrangement. For 11 hexagons, a common arrangement is:
    # Center one, surrounded by 6 in a ring, then 4 more around.
    # But let's try a more systematic approach.
    
    # Initial guess: place hexagons in a rough honeycomb pattern
    sqrt3 = np.sqrt(3)
    
    # Position 11 hexagons in a compact cluster
    # Using axial coordinates on a triangular lattice
    # Each hexagon center is at distance sqrt(3) from its neighbors
    
    # Let me define positions relative to center
    # A good starting config: one in center, 6 around it, 4 more
    init_positions = np.array([
        [0, 0],           # center
        [sqrt3, 0],       # right
        [sqrt3/2, 1.5],   # top-right
        [-sqrt3/2, 1.5],  # top-left
        [-sqrt3, 0],      # left
        [-sqrt3/2, -1.5], # bottom-left
        [sqrt3/2, -1.5],  # bottom-right
        [2*sqrt3, 0],     # far right
        [sqrt3/2, 3.0],   # far top-right
        [-sqrt3/2, 3.0],  # far top-left
        [-sqrt3/2, -3.0], # far bottom-left
    ])
    
    # All angles are 0 initially
    init_angles = np.zeros(n)
    
    # Initial outer hexagon parameters
    # The outer hexagon needs to contain all inner hexagons
    # Let's estimate initial side length
    # Max distance from origin to any inner hexagon vertex
    def hexagon_vertices(cx, cy, s, angle_deg):
        """Return 6 vertices of a regular hexagon with center (cx,cy), side s, rotation angle_deg."""
        angle_rad = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle_rad + i * np.pi / 3
            vertices.append([cx + s * np.cos(theta), cy + s * np.sin(theta)])
        return np.array(vertices)
    
    def get_all_inner_vertices(positions, angles, side=1.0):
        """Get all vertices of all inner hexagons."""
        all_verts = []
        for i in range(len(positions)):
            verts = hexagon_vertices(positions[i, 0], positions[i, 1], side, angles[i])
            all_verts.append(verts)
        return all_verts
    
    def compute_min_bounding_hexagon(vertices_list, n_sides=6):
        """
        Find the minimum side length of a regular hexagon (with variable center and rotation)
        that contains all given vertices.
        We optimize over center (cx, cy) and rotation angle.
        """
        # Flatten all vertices
        all_verts = np.vstack(vertices_list)
        
        def objective(params):
            cx, cy, angle_deg = params
            angle_rad = np.radians(angle_deg)
            
            # For a regular hexagon with center (cx,cy), side s, rotation angle_rad,
            # a point (x,y) is inside if for each of the 6 edges, the point is on the correct side.
            # The 6 edge normals are at angles angle_rad + i*pi/3 + pi/6 (pointing outward)
            # Actually, let's think of it differently.
            # The hexagon can be defined by 6 half-planes.
            # Edge i has normal direction at angle (angle_rad + i*pi/3) and the supporting line
            # is at distance s * sqrt(3)/2 from center (the apothem).
            
            apothem = None  # we'll solve for this
            
            # For containment, we need to find the minimum apothem such that all points
            # are inside. For a given orientation, the required apothem is the max over all
            # points and all 6 directions of the projection.
            
            # The 6 outward normal directions for the hexagon edges
            # If the hexagon has vertices at angles angle_rad + i*pi/3, then the edges
            # are between consecutive vertices. The normal to edge i points outward.
            # Edge between vertex i and i+1: the outward normal is at angle angle_rad + (i+0.5)*pi/3
            
            normal_angles = angle_rad + (np.arange(6) + 0.5) * np.pi / 3
            
            # For each point, project onto each normal direction
            # The distance from center in that direction must be <= apothem
            # So required apothem = max over all points and all normals of (dot(point - center, normal))
            
            dx = all_verts[:, 0] - cx
            dy = all_verts[:, 1] - cy
            
            # Project onto each normal
            max_proj = 0
            for na in normal_angles:
                proj = dx * np.cos(na) + dy * np.sin(na)
                max_proj = max(max_proj, np.max(proj))
            
            if max_proj < 0:
                max_proj = 0
            
            apothem = max_proj
            side = apothem / (np.sqrt(3) / 2)
            
            return side
        
        # Optimize over center and rotation
        best_side = np.inf
        best_params = None
        
        # Try multiple initial guesses for rotation
        for rot_init in [0, 30, 60, 90, 120, 150]:
            # Estimate center as centroid of all vertices
            cx0 = np.mean(all_verts[:, 0])
            cy0 = np.mean(all_verts[:, 1])
            
            res = minimize(objective, [cx0, cy0, float(rot_init)], 
                          method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_params = res.x
        
        return best_side, best_params
    
    # Optimization: we'll jointly optimize inner positions, angles, and outer hexagon
    # to minimize the outer side length while maintaining separation
    
    def separation_violation(pos_i, angle_i, pos_j, angle_j):
        """
        Check if two unit hexagons with centers pos_i, pos_j and angles angle_i, angle_j
        are separated. Returns the minimum gap (positive means separated, negative means overlapping).
        We check all 12 edge normals (6 from each hexagon).
        """
        # Get vertices of both hexagons
        verts_i = hexagon_vertices(pos_i[0], pos_i[1], 1.0, angle_i)
        verts_j = hexagon_vertices(pos_j[0], pos_j[1], 1.0, angle_j)
        
        # For each edge of hexagon i, check if all vertices of hexagon j are outside (or on)
        # The edge normal points outward from hexagon i
        # A point is inside hexagon i if it's on the inward side of all 6 edges
        
        # Actually, the condition is: for separation, on at least one of the 12 edge normals,
        # the projected intervals are apart by more than 1e-6.
        
        # Let's compute the projection intervals for both hexagons on each of the 12 normals
        # and check if any pair of intervals are disjoint with gap > 1e-6
        
        # Normals for hexagon i: 6 outward normals
        angle_i_rad = np.radians(angle_i)
        angle_j_rad = np.radians(angle_j)
        
        normals_i = []
        for k in range(6):
            na = angle_i_rad + (k + 0.5) * np.pi / 3
            normals_i.append([np.cos(na), np.sin(na)])
        
        normals_j = []
        for k in range(6):
            na = angle_j_rad + (k + 0.5) * np.pi / 3
            normals_j.append([np.cos(na), np.sin(na)])
        
        all_normals = normals_i + normals_j
        
        for nrm in all_normals:
            # Project all vertices of hex i onto this normal
            proj_i = verts_i @ np.array(nrm)
            # Project all vertices of hex j onto this normal
            proj_j = verts_j @ np.array(nrm)
            
            min_i = np.min(proj_i)
            max_i = np.max(proj_i)
            min_j = np.min(proj_j)
            max_j = np.max(proj_j)
            
            # Check if intervals are disjoint
            # Either max_i < min_j - 1e-6 or max_j < min_i - 1e-6
            if max_i < min_j - 1e-6 or max_j < min_i - 1e-6:
                return 1e-6  # separated
        
        return -1.0  # not separated (overlapping or touching)
    
    def objective_function(params):
        """
        Objective: minimize outer hexagon side length.
        params: [x0, y0, a0, x1, y1, a1, ..., x10, y10, a10, cx, cy, outer_angle]
        where (xi, yi, ai) are inner hexagon centers and angles (degrees)
        and (cx, cy, outer_angle) are outer hexagon center and rotation angle (degrees)
        """
        n_inner = 11
        inner_params = params[:3 * n_inner]
        outer_params = params[3 * n_inner:]
        
        positions = np.array([[inner_params[3*i], inner_params[3*i+1]] for i in range(n_inner)])
        angles = np.array([inner_params[3*i+2] for i in range(n_inner)])
        
        cx, cy, outer_angle_deg = outer_params
        outer_angle_rad = np.radians(outer_angle_deg)
        
        # Check separation between all pairs of inner hexagons
        for i in range(n_inner):
            for j in range(i + 1, n_inner):
                gap = separation_violation(positions[i], angles[i], positions[j], angles[j])
                if gap < 0:
                    # Penalty for overlap
                    return 1e6 + abs(gap)
        
        # Compute the minimum bounding hexagon side length
        all_verts = []
        for i in range(n_inner):
            verts = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            all_verts.append(verts)
        
        # For the given outer hexagon orientation, compute required side length
        normal_angles = outer_angle_rad + (np.arange(6) + 0.5) * np.pi / 3
        
        max_proj = 0
        for verts in all_verts:
            dx = verts[:, 0] - cx
            dy = verts[:, 1] - cy
            for na in normal_angles:
                proj = dx * np.cos(na) + dy * np.sin(na)
                max_proj = max(max_proj, np.max(proj))
        
        if max_proj < 0:
            max_proj = 0
        
        side = max_proj / (np.sqrt(3) / 2)
        
        return side
    
    # Initial parameters
    init_params = np.concatenate([
        init_positions.flatten(),
        init_angles,
        [0, 0, 0]  # outer hex center and angle
    ])
    
    # Optimize
    res = minimize(objective_function, init_params, method='Nelder-Mead',
                  options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    # Extract results
    opt_params = res.x
    n_inner = 11
    positions = np.array([[opt_params[3*i], opt_params[3*i+1]] for i in range(n_inner)])
    angles = np.array([opt_params[3*i+2] for i in range(n_inner)])
    cx, cy, outer_angle_deg = opt_params[3*n_inner:]
    
    # Build inner hex data
    inner_hex_data = np.column_stack([positions, angles])
    
    # Build outer hex data
    outer_hex_data = np.array([cx, cy, outer_angle_deg])
    
    # Compute side length
    all_verts = []
    for i in range(n_inner):
        verts = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
        all_verts.append(verts)
    
    outer_angle_rad = np.radians(outer_angle_deg)
    normal_angles = outer_angle_rad + (np.arange(6) + 0.5) * np.pi / 3
    
    max_proj = 0
    for verts in all_verts:
        dx = verts[:, 0] - cx
        dy = verts[:, 1] - cy
        for na in normal_angles:
            proj = dx * np.cos(na) + dy * np.sin(na)
            max_proj = max(max_proj, np.max(proj))
    
    if max_proj < 0:
        max_proj = 0
    
    side = max_proj / (np.sqrt(3) / 2)
    
    # Ensure side is at least sqrt(11)
    side = max(side, np.sqrt(11))
    
    return inner_hex_data, outer_hex_data, float(side)


# EVOLVE-BLOCK-END
