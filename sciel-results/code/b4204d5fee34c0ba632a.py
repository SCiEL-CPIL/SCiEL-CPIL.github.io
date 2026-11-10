# EVOLVE-BLOCK-START
import numpy as np
import time


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    Uses a fixed honeycomb lattice configuration and optimizes the outer hexagon orientation.
    """
    sqrt3 = np.sqrt(3)
    
    # Honeycomb lattice vectors for edge-adjacent unit hexagons
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3 / 2.0, 1.5])

    # Define 12 lattice coordinates (i, j) that form a compact cluster
    # Center + Ring 1 (6) + 5 from Ring 2
    coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (-1, 2), (-2, 1)
    ]

    # Calculate centers
    centers = np.array([i * a1 + j * a2 for i, j in coords])
    
    # Unit hexagon vertices (angle 0, centered at origin)
    hex_angles = np.arange(6) * np.pi / 3.0
    unit_verts = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])  # (6,2)

    # Generate all vertices of the 12 hexagons
    # Centers shape (12, 2), unit_verts shape (6, 2)
    all_verts = (centers[:, np.newaxis, :] + unit_verts[np.newaxis, :, :]).reshape(-1, 2)
    
    # The cluster is centered around origin. 
    # We will find the minimal enclosing regular hexagon for these points.
    # The outer hexagon is centered at (0,0) with angle alpha.
    # For a regular hexagon with apothem h and orientation alpha, the constraints are:
    # |P dot u_k| <= h for k=0,1,2, where u_k are unit normals at alpha, alpha+pi/3, alpha+2pi/3
    # h = s * sqrt(3) / 2  => s = 2 * h / sqrt(3)
    
    def min_side_for_angle(alpha):
        # Normals for the 3 pairs of parallel sides
        angles = np.array([alpha, alpha + np.pi/3, alpha + 2*np.pi/3])
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Project all vertices onto these normals
        # all_verts: (72, 2), normals: (3, 2)
        projections = all_verts @ normals.T  # (72, 3)
        
        # Max absolute projection
        max_proj = np.max(np.abs(projections))
        
        # Apothem h must be at least max_proj
        h = max_proj
        s = 2 * h / sqrt3
        return s

    # Scan for optimal angle
    # Symmetry of the problem might allow reducing range, but full scan is fast
    # The lattice has some symmetry, but the selection of 5 from ring 2 breaks full 6-fold symmetry.
    # We scan 0 to pi (180 deg) with fine steps.
    best_s = np.inf
    best_alpha = 0.0
    
    # Use a fine grid
    alphas = np.linspace(0, np.pi, 1800)
    for a in alphas:
        s_val = min_side_for_angle(a)
        if s_val < best_s:
            best_s = s_val
            best_alpha = a
            
    # Refine with a few steps around best_alpha
    ref_alphas = np.linspace(best_alpha - np.pi/180, best_alpha + np.pi/180, 180)
    for a in ref_alphas:
        s_val = min_side_for_angle(a)
        if s_val < best_s:
            best_s = s_val
            best_alpha = a

    # Construct output
    # Inner hexagons: centers are 'centers', angle is 0 (since we didn't rotate the cluster)
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i] = [centers[i, 0], centers[i, 1], 0.0]
        
    # Outer hexagon: center (0,0), angle best_alpha, side best_s
    outer_hex_data = np.array([0.0, 0.0, np.degrees(best_alpha)])
    
    return inner_hex_data, outer_hex_data, best_s

# EVOLVE-BLOCK-END
