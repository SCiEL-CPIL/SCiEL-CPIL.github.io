# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def hexagon_packing_12():
    # 1. Define 12 unit hexagon centers in a compact cluster
    # Axial coordinates (q, r)
    # This is a "flower" (7 hexes) + 5 more on the perimeter
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (2, -1), (1, -2), (-1, -2), (-2, -1)
    ]
    
    # Scale by a small factor to ensure separation (gap > 1e-6)
    # Distance between adjacent centers is sqrt(3).
    # Scaling by (1+delta) creates a gap of sqrt(3)*delta.
    # We need sqrt(3)*delta > 1e-6 => delta > ~6e-7.
    # Use 1e-4 for safety.
    scale = 1.0 + 1e-4
    
    # Convert to Cartesian
    centers = []
    for q, r in axial_coords:
        x = 1.5 * q * scale
        y = np.sqrt(3) * (r + 0.5 * q) * scale
        centers.append([x, y])
    centers = np.array(centers)
    
    # Generate vertices for all 12 hexagons
    # Hexagon vertices relative to center, side 1, angle 0
    angles = np.deg2rad(np.arange(0, 360, 60))
    rel_vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # We will optimize rotation of the entire cluster.
    # The vertices of each hexagon are center + rel_vertices.
    # If we rotate the cluster by theta, the centers rotate, and the hexagons themselves rotate?
    # The problem states: "inner a (12, 3) array of rows (x, y, angle_deg)".
    # The angle_deg is the orientation of the inner hexagon.
    # If we rotate the cluster, we can either:
    # A) Keep inner hexagons at angle 0 and rotate centers. (This changes relative orientation of hexagon faces to cluster symmetry? No, hexagons are identical. But the *packing* geometry depends on the shape of the union. The union of rotated hexagons is different from the union of unrotated hexagons at rotated centers?
    # Actually, a hexagon rotated by 60k is the same. A hexagon rotated by 30 is different (pointy vs flat top).
    # The standard "flat-top" hexagon has vertices at 0, 60, 120...
    # If we rotate the centers but keep hexagons at 0, the gaps change shape.
    # However, the most compact packing usually aligns the hexagons with the outer container's symmetry or in a lattice.
    # The current code assumes all inner hexagons are at angle 0.
    # If we rotate the *centers* but keep hexagons at 0, the "shape" of the cluster of hexagons is effectively rotated only if the hexagons are also rotated.
    # Wait. The union of hexagons is the object.
    # If I rotate the centers by theta, but keep each hexagon at angle 0, the union is NOT a rotated version of the original union. It is a "twisted" cluster.
    # To rotate the *cluster* properly, I must rotate both the centers AND the orientation of each hexagon.
    # So, inner_angle = theta for all hexagons?
    # If all inner hexagons are rotated by theta, the union is the original union rotated by theta.
    # This is the standard approach.
    
    all_vertices_unrotated = []
    for c in centers:
        verts = c + rel_vertices
        all_vertices_unrotated.append(verts)
    all_vertices_unrotated = np.vstack(all_vertices_unrotated) # Shape (72, 2)
    
    # 2. Define the objective function to minimize
    # Outer hexagon normals (perpendicular to edges)
    # Outer hexagon is fixed at angle 0.
    normal_angles = np.deg2rad(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)]) # Shape (6, 2)
    
    def objective(T):
        tx, ty, theta = T
        # Rotate vertices
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        rotated_vertices = all_vertices_unrotated @ R.T
        shifted_vertices = rotated_vertices + np.array([tx, ty])
        
        # Project onto all 6 normals
        projections = shifted_vertices @ normals.T
        max_abs_proj = np.max(np.abs(projections))
        # s_outer = (2/sqrt(3)) * max_abs_proj
        return (2.0 / np.sqrt(3)) * max_abs_proj

    # 3. Optimize translation T and rotation theta
    # Initial guess: center of mass at 0, theta = 0
    initial_T = [-np.mean(centers, axis=0)[0], -np.mean(centers, axis=0)[1], 0.0]
    
    # Use Nelder-Mead for 3D optimization
    res = minimize(objective, initial_T, method='Nelder-Mead', options={'xatol': 1e-9, 'fatol': 1e-9, 'maxiter': 5000})
    
    optimal_T = res.x
    optimal_s = res.fun
    optimal_theta = optimal_T[2]
    
    # 4. Construct the result
    # Rotate centers
    c = np.cos(optimal_theta)
    s = np.sin(optimal_theta)
    R = np.array([[c, -s], [s, c]])
    
    rotated_centers = centers @ R.T
    final_centers = rotated_centers + np.array([optimal_T[0], optimal_T[1]])
    
    # Inner hexagons: (x, y, angle_deg)
    # All inner hexagons are rotated by optimal_theta
    inner_angle_deg = np.degrees(optimal_theta)
    inner_hex_data = np.column_stack([final_centers, np.full(12, inner_angle_deg)])
    
    # Outer hexagon: centered at (0,0), angle 0
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, optimal_s
# EVOLVE-BLOCK-END
