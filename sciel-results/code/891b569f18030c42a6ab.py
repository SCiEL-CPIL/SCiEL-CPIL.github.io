import numpy as np
from scipy.optimize import minimize

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    using a 3-4-3 honeycomb rigid cluster with optimized global rotation and outer hexagon angle.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), (x, y, angle_deg) for each inner hexagon.
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_deg) for the outer hexagon.
        s: float, side length of the outer hexagon.
    """
    # Define the 3-4-3 honeycomb topology for 12 unit hexagons (side=1, flat-top, angle=0)
    # Middle row (4 hexagons) at y=0
    sqrt3 = np.sqrt(3)
    middle_x = np.array([-1.5*sqrt3, -0.5*sqrt3, 0.5*sqrt3, 1.5*sqrt3])
    middle_y = np.zeros(4)
    
    # Top row (3 hexagons) at y = 1.5
    top_x = np.array([-sqrt3, 0, sqrt3])
    top_y = np.full(3, 1.5)
    
    # Bottom row (3 hexagons) at y = -1.5
    bottom_x = np.array([-sqrt3, 0, sqrt3])
    bottom_y = np.full(3, -1.5)
    
    # Combine all 12 centers
    base_centers = np.vstack([
        np.column_stack([middle_x, middle_y]),
        np.column_stack([top_x, top_y]),
        np.column_stack([bottom_x, bottom_y])
    ])  # shape (12, 2)
    
    # All inner hexagons have angle 0 (flat-top)
    base_angles = np.zeros(12)
    
    def get_hex_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of a regular hexagon with center (cx,cy), side, angle in degrees."""
        a = np.radians(angle_deg)
        angles = a + np.radians(60) * np.arange(6)
        vx = cx + side * np.cos(angles)
        vy = cy + side * np.sin(angles)
        return np.column_stack([vx, vy])  # (6, 2)
    
    def enclosing_hex_side(centers, angles, outer_angle_deg):
        """
        Given inner hexagon centers and angles, and outer hexagon angle,
        compute the minimum side length s of a regular hexagon (centered at origin)
        with the given angle that contains all inner hexagon vertices.
        """
        # Compute all vertices of all inner hexagons
        all_vertices = []
        for i in range(len(centers)):
            cx, cy = centers[i]
            verts = get_hex_vertices(cx, cy, 1.0, angles[i])
            all_vertices.append(verts)
        all_vertices = np.vstack(all_vertices)  # (72, 2)
        
        # Outer hexagon: flat-top with rotation outer_angle_deg
        # The 6 outward normals for a flat-top hexagon (vertex at angle 0) are at angles 0, 60, 120, 180, 240, 300
        # With rotation, they are at outer_angle + 60k
        outer_a = np.radians(outer_angle_deg)
        normal_angles = outer_a + np.radians(60) * np.arange(6)
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)
        
        # For each vertex, compute max dot product with each normal
        # dot_products: (72, 6)
        dots = all_vertices @ normals.T  # (72, 6)
        max_dot = np.max(dots, axis=0)  # (6,)
        max_over_all = np.max(max_dot)
        
        # Apothem = s * sqrt(3)/2, so s = max_over_all / (sqrt(3)/2) = 2 * max_over_all / sqrt(3)
        s = 2.0 * max_over_all / sqrt3
        return s
    
    def objective(params):
        """
        params: [cluster_rotation_deg, outer_angle_deg]
        """
        theta = params[0]
        phi = params[1]
        
        # Rotate the base centers by theta
        theta_rad = np.radians(theta)
        R = np.array([
            [np.cos(theta_rad), -np.sin(theta_rad)],
            [np.sin(theta_rad), np.cos(theta_rad)]
        ])
        rotated_centers = base_centers @ R.T  # (12, 2)
        
        # Rotate the inner hexagon angles too (they were all 0, now they become theta)
        rotated_angles = np.full(12, theta)
        
        # Center the cluster at origin (subtract mean of centers)
        center_offset = np.mean(rotated_centers, axis=0)
        rotated_centers = rotated_centers - center_offset
        
        s = enclosing_hex_side(rotated_centers, rotated_angles, phi)
        return s
    
    # Optimize over [cluster_rotation, outer_angle]
    # Use multiple starting points to avoid local minima
    best_s = np.inf
    best_params = None
    
    for theta0 in np.linspace(0, 60, 13):
        for phi0 in np.linspace(0, 60, 13):
            res = minimize(objective, [theta0, phi0], method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_s:
                best_s = res.fun
                best_params = res.x
    
    theta_opt, phi_opt = best_params
    
    # Compute final centers and angles
    theta_rad = np.radians(theta_opt)
    R = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad), np.cos(theta_rad)]
    ])
    final_centers = base_centers @ R.T
    final_angles = np.full(12, theta_opt)
    
    # Center at origin
    center_offset = np.mean(final_centers, axis=0)
    final_centers = final_centers - center_offset
    
    # Build output arrays
    inner_hex_data = np.column_stack([final_centers[:, 0], final_centers[:, 1], final_angles])
    outer_hex_data = np.array([0.0, 0.0, phi_opt])
    s = best_s
    
    return inner_hex_data, outer_hex_data, s
