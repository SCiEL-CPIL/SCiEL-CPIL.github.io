# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length s.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), rows (x, y, angle_deg).
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_deg).
        outer_hex_side_length: float.
    """
    n = 12

    # --- Geometry helpers ---
    def hex_vertices(cx, cy, d, angle_deg):
        a = np.deg2rad(angle_deg)
        angles = a + np.deg2rad(60 * np.arange(6))
        return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

    def edge_normals(vertices):
        """Return unit outward normals for each edge of a convex polygon given in CCW order."""
        edges = np.roll(vertices, -1, axis=0) - vertices
        # For CCW polygon, outward normal is (dy, -dx) normalized? 
        # Actually for CCW, the outward normal of edge (p_i -> p_{i+1}) is (dy, -dx) / |edge|
        # Wait: if polygon is CCW, the interior is to the left. So outward is to the right: (dy, -dx).
        dx = edges[:, 1]
        dy = -edges[:, 0]
        norms = np.sqrt(dx**2 + dy**2)
        norms[norms == 0] = 1e-12
        return np.column_stack([dx / norms, dy / norms])

    def sep_axis_min_gap(A_verts, A_normals, B_verts, B_normals):
        """
        Compute the minimum gap between two convex polygons using SAT.
        Returns the minimum distance (positive if separated, negative if overlapping).
        We compute the maximum over all axes of (min_A - max_B) or (min_B - max_A).
        Actually, we want the minimum over all axes of the gap.
        Gap on axis n = min( min_A_proj - max_B_proj, min_B_proj - max_A_proj ) is not right.
        
        Standard SAT: for each axis, compute projection intervals.
        If intervals overlap on all axes, they intersect.
        The "minimum gap" is the minimum over all axes of the distance between intervals.
        Distance between intervals [a1, a2] and [b1, b2] is:
        if a2 < b1: b1 - a2
        elif b2 < a1: a1 - b2
        else: 0 (overlap)
        """
        min_gap = np.inf
        for normals in [A_normals, B_normals]:
            for n in normals:
                proj_A = A_verts @ n
                proj_B = B_verts @ n
                a_min, a_max = proj_A.min(), proj_A.max()
                b_min, b_max = proj_B.min(), proj_B.max()
                if a_max < b_min:
                    gap = b_min - a_max
                elif b_max < a_min:
                    gap = a_min - b_max
                else:
                    gap = 0.0
                min_gap = min(min_gap, gap)
        return min_gap

    def point_in_hexagon(v, outer_verts):
        """Check if point v is inside or on the boundary of outer hexagon (CCW vertices)."""
        # For each edge (p1, p2), cross(p2-p1, v-p1) should be >= -eps for CCW
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % 6]
            cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
            if cross < -1e-6:
                return False
        return True

    # --- Objective function ---
    def objective(params):
        # params: [s, x1, y1, a1, x2, y2, a2, ..., x12, y12, a12]
        # But we fix angles to 0 for simplicity? The problem allows any angle.
        # However, rotating hexagons might help. Let's allow angles.
        # Actually, for regular hexagons, rotation by 60 degrees is symmetric.
        # Let's fix angles to 0 to reduce dimensionality, as optimal packings of hexagons
        # often align with the outer hexagon or have specific symmetries.
        # But to be safe, let's allow angles.
        
        s = params[0]
        if s < np.sqrt(12) - 1e-6:
            return 1e10
        
        inner_params = params[1:]
        inner_centers = inner_params[0::3]
        inner_angles = inner_params[1::3]  # in degrees
        # inner_params[2::3] are unused if we group differently?
        # Let's restructure: params = [s, (x1, a1), (x2, a2), ...]? No, y is needed.
        # params = [s, x1, y1, a1, x2, y2, a2, ...]
        
        # Recompute
        s = params[0]
        coords = params[1:]
        # coords: 12 * 3 = 36 values
        x = coords[0::3]
        y = coords[1::3]
        a_deg = coords[2::3]
        
        # Outer hexagon
        outer_verts = hex_vertices(0, 0, s, 0)
        outer_normals = edge_normals(outer_verts)
        
        penalty = 0.0
        
        # Check inner-inner collisions
        for i in range(n):
            verts_i = hex_vertices(x[i], y[i], 1.0, a_deg[i])
            normals_i = edge_normals(verts_i)
            
            # Check inner-outer
            for v in verts_i:
                if not point_in_hexagon(v, outer_verts):
                    # Penalty for being outside
                    # Compute distance to boundary
                    min_dist = np.inf
                    for j in range(6):
                        p1 = outer_verts[j]
                        p2 = outer_verts[(j + 1) % 6]
                        # Distance from point to line segment
                        # Simplified: just use the cross product magnitude scaled
                        # Better: compute actual distance to edge
                        edge_vec = p2 - p1
                        edge_len = np.linalg.norm(edge_vec)
                        if edge_len < 1e-12:
                            continue
                        t = np.dot(v - p1, edge_vec) / (edge_len ** 2)
                        t = np.clip(t, 0, 1)
                        proj = p1 + t * edge_vec
                        dist = np.linalg.norm(v - proj)
                        min_dist = min(min_dist, dist)
                    penalty += 100 * min_dist ** 2
        
            # Check inner-inner
            for j in range(i + 1, n):
                verts_j = hex_vertices(x[j], y[j], 1.0, a_deg[j])
                normals_j = edge_normals(verts_j)
                
                gap = sep_axis_min_gap(verts_i, normals_i, verts_j, normals_j)
                if gap < 1e-4:
                    # Penalty for overlap or close contact
                    # We want gap >= 0 (separated)
                    # If gap < 0, they overlap.
                    # Penalty should be large when gap is negative.
                    penalty += 100 * max(0, -gap) ** 2
                    # Also add a small penalty for being too close to encourage separation
                    penalty += 1 * max(0, 1e-4 - gap) ** 2
        
        return s + penalty

    # --- Initial guess ---
    # Use the grid arrangement from the seed as a starting point, but with a smaller s
    # The seed used s=8. Let's try to start with a tighter bound.
    # Theoretical minimum s is sqrt(12) ≈ 3.464.
    # A reasonable packing might have s around 4-5.
    
    # Let's use a hexagonal grid layout for initial centers
    # Center at (0,0)
    # Neighbors at distance 2 (since side 1 hexagons touch at distance 2)
    # Actually, distance between centers of touching hexagons is 2 * apothem? No.
    # Apothem = sqrt(3)/2. Distance between centers = 2 * sqrt(3)/2 * 2? 
    # Let's just use the seed's relative positions but scaled.
    
    initial_centers = np.array([
        [0, 0],
        [-2.5, 0],
        [2.5, 0],
        [-1.25, 2.17],
        [1.25, 2.17],
        [-1.25, -2.17],
        [1.25, -2.17],
        [-3.75, 2.17],
        [3.75, 2.17],
        [-3.75, -2.17],
        [3.75, -2.17],
        [0, -4]
    ])
    
    initial_angles = np.zeros(n)
    
    # Estimate initial s
    # Max distance from origin to any vertex of any inner hexagon
    max_r = 0
    for i in range(n):
        verts = hex_vertices(initial_centers[i, 0], initial_centers[i, 1], 1.0, 0)
        r = np.max(np.linalg.norm(verts, axis=1))
        max_r = max(max_r, r)
    
    initial_s = max_r + 0.5  # Some buffer
    
    # Build initial params
    init_params = np.zeros(1 + 3 * n)
    init_params[0] = initial_s
    for i in range(n):
        init_params[1 + 3*i] = initial_centers[i, 0]
        init_params[2 + 3*i] = initial_centers[i, 1]
        init_params[3 + 3*i] = initial_angles[i]
    
    # Bounds
    # s >= sqrt(12)
    # Centers can be anywhere, but let's bound them to [-10, 10]
    # Angles in [0, 360)
    bounds = [(np.sqrt(12), 100)]
    for i in range(n):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
        bounds.append((0, 360))
    
    # Optimize
    result = minimize(objective, init_params, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12})
    
    # Extract results
    s_opt = result.x[0]
    coords_opt = result.x[1:]
    x_opt = coords_opt[0::3]
    y_opt = coords_opt[1::3]
    a_opt = coords_opt[2::3]
    
    # Construct output arrays
    inner_hex_data = np.column_stack([x_opt, y_opt, a_opt])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s_opt


# EVOLVE-BLOCK-END
