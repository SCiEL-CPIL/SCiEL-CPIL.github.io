import numpy as np

def construct():
    # Lattice vectors for unit hexagons
    u = np.array([np.sqrt(3), 0.0])
    v = np.array([np.sqrt(3)/2.0, 1.5])
    
    # Define 11 hexagon centers in lattice coordinates (i, j)
    # This forms a "Cross" shape: center, 6 neighbors, 4 second-neighbors along axes
    lattice_coords = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        (2, 0), (-2, 0),
        (0, 2), (0, -2)
    ]
    
    # Convert to Cartesian coordinates
    centers = []
    for i, j in lattice_coords:
        p = i * u + j * v
        centers.append(p)
    
    centers = np.array(centers)
    
    # Normals for the outer hexagon (flat-top, side S)
    # The outer hexagon is fixed at rotation 0.
    # Normals are at 0, 60, 120, 180, 240, 300 degrees
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    def calc_side_length(alpha):
        # Rotate centers by alpha
        c = np.cos(alpha)
        s = np.sin(alpha)
        R = np.array([[c, -s], [s, c]])
        rotated_centers = centers @ R.T
        
        # Project onto normals
        # projections[i, k] = dot(rotated_centers[i], normals[k])
        projections = rotated_centers @ normals.T
        abs_projections = np.abs(projections)
        
        # Max projection
        max_proj = np.max(abs_projections)
        
        # Side length S such that apothem A = S * sqrt(3) / 2 >= max_proj
        # S = 2 * max_proj / sqrt(3)
        S = 2.0 * max_proj / np.sqrt(3)
        return S
    
    # Minimize S(alpha) for alpha in [0, pi/3]
    # Use ternary search or dense sampling
    # Since the function is periodic and symmetric, [0, pi/3] is sufficient.
    
    # Initial dense sampling
    alphas = np.linspace(0, np.pi/3, 1000)
    S_vals = [calc_side_length(a) for a in alphas]
    min_idx = np.argmin(S_vals)
    best_alpha = alphas[min_idx]
    
    # Refine with ternary search around best_alpha
    # The function is not necessarily convex, but locally it should be well-behaved
    # Or just use a finer grid
    start = best_alpha - 0.01
    end = best_alpha + 0.01
    if start < 0: start = 0
    if end > np.pi/3: end = np.pi/3
    
    fine_alphas = np.linspace(start, end, 1000)
    fine_S_vals = [calc_side_length(a) for a in fine_alphas]
    fine_min_idx = np.argmin(fine_S_vals)
    final_alpha = fine_alphas[fine_min_idx]
    final_S = calc_side_length(final_alpha)
    
    # Prepare output
    # Inner hexagons: (x, y, angle_degrees)
    # The inner hexagons are unit hexagons. Their orientation is 0 relative to the cluster?
    # The task says "each inner hexagon may translate and rotate".
    # In our model, the cluster is rigid. The individual hexagons have the same orientation.
    # We can set their angle to 0.0.
    c = np.cos(final_alpha)
    s = np.sin(final_alpha)
    R = np.array([[c, -s], [s, c]])
    final_centers = centers @ R.T
    
    inner = []
    for p in final_centers:
        inner.append([p[0], p[1], 0.0])
        
    # Outer hexagon: center (0,0), side length final_S, rotation 0.0
    return inner, (0.0, 0.0), final_S, 0.0
