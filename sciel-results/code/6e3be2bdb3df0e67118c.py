<<<<<<< SEARCH
    # Final refinement: find optimal alpha for the best centers
    centers_final = best_params.reshape(11, 2)
    
    # Optimize alpha
    def S_alpha(alpha):
        return compute_outer_side(centers_final, alpha)
    
    # Check alpha in [0, pi/3]
    alphas = np.linspace(0, np.pi/3, 100)
    S_vals = [S_alpha(a) for a in alphas]
    best_alpha = alphas[np.argmin(S_vals)]
    best_S = min(S_vals)
    
    # Build output
    inner_hex_data = np.zeros((11, 3))
    for i, c in enumerate(centers_final):
        inner_hex_data[i, 0] = c[0]
        inner_hex_data[i, 1] = c[1]
        inner_hex_data[i, 2] = 0.0 # Angle 0
    
    outer_hex_data = np.array([0.0, 0.0, np.rad2deg(best_alpha)])
    outer_hex_side_length = float(best_S)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
=======
    # Final refinement: find optimal alpha for the best centers
    centers_final = best_params.reshape(11, 2)
    
    # Optimize alpha
    def S_alpha(alpha):
        return compute_outer_side(centers_final, alpha)
    
    # Check alpha in [0, pi/3]
    alphas = np.linspace(0, np.pi/3, 200)
    S_vals = [S_alpha(a) for a in alphas]
    best_alpha = alphas[np.argmin(S_vals)]
    best_S = min(S_vals)
    
    # Verify containment: check that all inner vertices are inside the outer hexagon
    all_inner_verts = np.vstack([get_hex_vertices(c[0], c[1], 0.0) for c in centers_final])
    outer_center = np.array([0.0, 0.0])
    outer_alpha = best_alpha
    outer_s = best_S
    
    # Check each inner vertex against each outer edge
    for i in range(6):
        theta1 = outer_alpha + i * np.pi / 3
        theta2 = outer_alpha + (i + 1) * np.pi / 3
        p1 = outer_center + outer_s * np.array([np.cos(theta1), np.sin(theta1)])
        p2 = outer_center + outer_s * np.array([np.cos(theta2), np.sin(theta2)])
        
        edge = p2 - p1
        # Normal pointing inward
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normal = normal / norm
            # Check if normal points toward center
            if np.dot(normal, outer_center - p1) < 0:
                normal = -normal
            
            # For each inner vertex, check cross product
            for p in all_inner_verts:
                cross_val = np.cross(p2 - p1, p - p1)
                if cross_val < -1e-6:
                    # Vertex is outside, increase side length slightly
                    best_S *= 1.01
                    break
    
    # Build output
    inner_hex_data = np.zeros((11, 3))
    for i, c in enumerate(centers_final):
        inner_hex_data[i, 0] = c[0]
        inner_hex_data[i, 1] = c[1]
        inner_hex_data[i, 2] = 0.0 # Angle 0
    
    outer_hex_data = np.array([0.0, 0.0, np.rad2deg(best_alpha)])
    outer_hex_side_length = float(best_S)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
>>>>>>> REPLACE
