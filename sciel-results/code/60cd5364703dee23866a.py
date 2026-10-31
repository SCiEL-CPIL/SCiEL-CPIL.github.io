import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define the 11 hexagon centers in axial coordinates (q, r)
    # This forms a compact cluster:
    # Center (0,0)
    # Neighbors: (1,0), (-1,0), (0,1), (0,-1), (-1,1), (1,-1)
    # Extensions: (1,1), (-1,-1), (0,2), (0,-2)
    # Total: 1 + 6 + 4 = 11
    
    axial_coords = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (-1, 1), (1, -1),
        (1, 1), (-1, -1),
        (0, 2), (0, -2)
    ]
    
    # Convert axial to Cartesian (pointy-top, side length 1)
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    centers = []
    for q, r in axial_coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    centers = np.array(centers)
    
    # Function to calculate the required side length S for a given outer rotation
    def min_side_length(centers, outer_angle_deg):
        outer_angle_rad = np.deg2rad(outer_angle_deg)
        
        # Normals to the sides of the outer hexagon
        # For a pointy-top hexagon, the sides are at angles 30, 90, 150, 210, 270, 330 degrees from x-axis?
        # No, the vertices are at 90, 30, -30...
        # The normals to the sides are at 0, 60, 120, 180, 240, 300 degrees.
        # Let's verify: Side connecting (sqrt(3)/2, 1/2) and (sqrt(3)/2, -1/2) is vertical. Normal is (1, 0) -> 0 deg.
        # Side connecting (sqrt(3)/2, 1/2) and (0, 1) has normal at 60 deg?
        # Vector along side: (-sqrt(3)/2, 1/2). Normal: (1/2, sqrt(3)/2) -> 60 deg.
        # Yes, normals are at 0, 60, 120, 180, 240, 300.
        
        base_normals = [np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))]) for a in [0, 60, 120, 180, 240, 300]]
        
        # Rotate normals by outer_angle_rad
        cos_a = np.cos(outer_angle_rad)
        sin_a = np.sin(outer_angle_rad)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        rotated_normals = [rot_matrix @ n for n in base_normals]
        
        # Support function of a unit hexagon (side 1, angle 0) in direction u
        # h(u) = max dot(v, u) for v in unit hexagon vertices
        # Vertices of unit hexagon (pointy-top):
        # (0, 1), (sqrt(3)/2, 0.5), (sqrt(3)/2, -0.5), (0, -1), (-sqrt(3)/2, -0.5), (-sqrt(3)/2, 0.5)
        unit_vertices = np.array([
            [0, 1],
            [np.sqrt(3)/2, 0.5],
            [np.sqrt(3)/2, -0.5],
            [0, -1],
            [-np.sqrt(3)/2, -0.5],
            [-np.sqrt(3)/2, 0.5]
        ])
        
        max_required_inradius = 0
        
        for u in rotated_normals:
            # Support of unit hexagon in direction u
            h_unit = np.max(unit_vertices @ u)
            
            # Support of each inner hexagon center in direction u
            # center @ u
            center_projections = centers @ u
            
            # Total support for each inner hexagon
            total_supports = center_projections + h_unit
            
            # Max over all inner hexagons
            max_support = np.max(total_supports)
            
            if max_support > max_required_inradius:
                max_required_inradius = max_support
                
        # The inradius of the outer hexagon is (sqrt(3)/2) * S
        # So (sqrt(3)/2) * S >= max_required_inradius
        S = max_required_inradius / (np.sqrt(3) / 2.0)
        
        return S

    # Try different outer rotations to minimize S
    best_S = float('inf')
    best_angle = 0.0
    
    # Check a range of angles. Due to symmetry, 0 to 60 is sufficient, but we'll check 0-180 in steps of 15
    for angle in range(0, 180, 15):
        S = min_side_length(centers, angle)
        if S < best_S:
            best_S = S
            best_angle = angle

    # Construct the return values
    inner = [[x, y, 0.0] for x, y in centers]
    center = (0.0, 0.0)
    side = best_S
    angle = best_angle
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
