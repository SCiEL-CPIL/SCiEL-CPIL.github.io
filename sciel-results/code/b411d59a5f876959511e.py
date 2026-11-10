# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize_scalar

_START_TIME = time.time()
_DEADLINE = 630.0
_SAFETY_MARGIN = 30.0

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), (x, y, angle_deg)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_deg)
        outer_hex_side_length: float
    """
    # Define the rigid cluster of 12 hexagons.
    # Topology: Center (0) + Ring 1 (6) + Ring 2 (5 specific ones to keep compact)
    # Hexagonal lattice coordinates (axial):
    # Center: (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 candidates: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    # We pick 5 from Ring 2 to make a compact shape.
    # Let's pick: (2,0), (1,1), (0,2), (-1,2), (-2,1) -> This forms a "fan" or partial ring.
    # Actually, a more symmetric compact shape might be better.
    # Let's try: Center + 6 Ring 1 + 5 Ring 2.
    # To minimize bounding box, we want to fill corners.
    
    # Convert axial (q, r) to Cartesian (x, y) for flat-top hexagons (side 1)
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    
    sqrt3 = np.sqrt(3.0)
    
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return x, y

    # Define the 12 centers in the rigid frame
    # 0: Center
    centers_axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        # Ring 2: Pick 5 that create a compact cluster
        # (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
        (2, 0),
        (1, 1),
        (0, 2),
        (-1, 2),
        (-2, 1)
    ]
    
    # Convert to Cartesian
    rigid_centers = np.array([axial_to_cartesian(q, r) for q, r in centers_axial])
    
    # Center the rigid body at the origin for easier rotation
    center_of_mass = np.mean(rigid_centers, axis=0)
    rigid_centers = rigid_centers - center_of_mass
    
    # Precompute the vertices of the unit hexagons relative to their centers
    # A unit hexagon with side 1, angle 0 has vertices at angles 0, 60, 120, 180, 240, 300
    # Vertices: (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    hex_angles = np.deg2rad(60.0 * np.arange(6))
    unit_hex_verts = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    # All 12 hexagons, 6 vertices each
    # rigid_all_verts[i, j, :] is the j-th vertex of the i-th hexagon in the rigid frame
    rigid_all_verts = rigid_centers[:, np.newaxis, :] + unit_hex_verts[np.newaxis, :, :]
    
    # Function to compute the minimal enclosing regular hexagon side length
    # given a rotation angle theta (radians)
    def min_enclosing_hex_side(theta):
        # Rotation matrix
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        
        # Rotate all vertices
        # rigid_all_verts shape: (12, 6, 2)
        # R @ v for each v
        rotated_verts = (R @ rigid_all_verts.reshape(-1, 2).T).T.reshape(12, 6, 2)
        
        # The outer hexagon is centered at the origin, angle 0.
        # Its edges are defined by normals at 30, 90, 150, 210, 270, 330 degrees.
        # For a regular hexagon of side s centered at origin with angle 0:
        # The distance from center to each edge (apothem) is a = s * sqrt(3) / 2.
        # So s = 2 * a / sqrt(3).
        # The condition for a point v to be inside is:
        # n_k . v <= a for all k, where n_k are the unit outward normals.
        
        normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Project all rotated vertices onto each normal
        # rotated_verts_flat: (72, 2)
        verts_flat = rotated_verts.reshape(-1, 2)
        
        # Projections: (6, 72)
        projections = normals @ verts_flat.T
        
        # For each normal, find the max projection
        max_projections = np.max(projections, axis=1)
        
        # The required apothem is the max of these projections
        required_apothem = np.max(max_projections)
        
        # Side length s = 2 * apothem / sqrt(3)
        s = 2.0 * required_apothem / np.sqrt(3.0)
        
        return s

    # Optimize theta in [0, pi/3] due to 6-fold symmetry of the outer hexagon
    # and the periodicity of the inner cluster (though the cluster is not 6-fold symmetric,
    # the outer hexagon is, so we only need to search one sector of the outer hexagon's symmetry).
    # Actually, the inner cluster breaks the symmetry, but the outer hexagon's constraint
    # is periodic with period pi/3 (60 degrees).
    # So we search theta in [0, pi/3].
    
    best_theta = 0.0
    best_s = np.inf
    
    # Use scipy minimize_scalar for robustness
    try:
        result = minimize_scalar(
            min_enclosing_hex_side,
            bounds=(0, np.pi / 3),
            method='bounded',
            options={'xatol': 1e-10}
        )
        if result.success:
            best_theta = result.x
            best_s = result.fun
        else:
            # Fallback to grid search if optimization fails
            thetas = np.linspace(0, np.pi / 3, 1000)
            s_vals = np.array([min_enclosing_hex_side(t) for t in thetas])
            idx = np.argmin(s_vals)
            best_theta = thetas[idx]
            best_s = s_vals[idx]
    except Exception:
        # Fallback to grid search
        thetas = np.linspace(0, np.pi / 3, 1000)
        s_vals = np.array([min_enclosing_hex_side(t) for t in thetas])
        idx = np.argmin(s_vals)
        best_theta = thetas[idx]
        best_s = s_vals[idx]

    # Apply the best rotation to the centers
    c, s = np.cos(best_theta), np.sin(best_theta)
    R = np.array([[c, -s], [s, c]])
    
    final_centers = (R @ rigid_centers.T).T
    
    # The outer hexagon is centered at the origin, angle 0
    outer_center = np.array([0.0, 0.0])
    outer_angle = 0.0
    
    # Construct the output arrays
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = final_centers[i, 0]
        inner_hex_data[i, 1] = final_centers[i, 1]
        inner_hex_data[i, 2] = best_theta * 180.0 / np.pi  # Rotate the hexagons too
        
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = best_s
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
