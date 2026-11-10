# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons in a regular hexagon, minimizing the outer side length.
    Returns (inner, outer, s) where inner is (12,3) array of (x,y,angle_deg),
    outer is (3,) array (x,y,angle_deg), and s is the outer side length.
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s limit

    sqrt3 = np.sqrt(3)

    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        return np.array([x, y])

    # Define candidate configurations in axial coordinates
    configs = []

    # Config A: 4x3 rectangular
    configs.append([(q, r) for q in range(4) for r in range(3)])

    # Config B: Hexagonal-ish (1 + 6 + 5)
    configs.append([(0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
                    (2, 0), (2, -1), (1, -2), (0, -2), (-1, -1)])

    # Config C: 6x2 strip
    configs.append([(q, r) for q in range(6) for r in range(2)])

    # Config D: 3x4 (same as A but different order)
    configs.append([(q, r) for q in range(3) for r in range(4)])

    # Config E: More compact - 2-3-3-2-2 pattern
    # Let me think of a better shape
    # A hexagon of side 2 has 1 + 6 + 12 = 19, too many
    # Let's try: a 3-wide, 4-tall with some removals
    # Or just use the ones we have

    def compute_min_containing_hex_side(points, hex_angle_deg=0):
        """
        Compute the minimum side length of a regular hexagon with given angle
        that contains all given points. The hexagon center is optimized.
        
        For a regular hexagon with side s, angle a, center c:
        The 6 edge outward normals are at angles a + 30 + 60i degrees.
        The distance from center to each edge is s * sqrt(3) / 2.
        
        For each normal direction n_i, we need:
            max_j (dot(p_j - c, n_i)) <= s * sqrt(3) / 2
        
        So s >= max_i [max_j (dot(p_j - c, n_i))] / (sqrt(3)/2)
        
        To minimize s, we choose c to minimize the max over i of the max projection.
        This is equivalent to finding the smallest regular hexagon (fixed orientation) 
        containing all points.
        
        The optimal center is such that the max projection in each of the 6 directions
        is balanced as much as possible.
        
        A simple approach: the optimal center is the centroid of the points that are
        "tight" in each direction. But let's just use the centroid as a good approximation,
        and then the side length is determined.
        
        Actually, for a fixed orientation, the minimum enclosing regular hexagon has its
        center at a point that minimizes the maximum of the 6 support functions.
        This is a convex problem. The optimal center satisfies that the max projection
        in each of the 6 normal directions is equal (if possible) or as balanced as possible.
        
        For simplicity and speed, let's compute it as:
        For each of the 6 directions, compute the max and min projection of all points.
        The "width" in that direction is max - min.
        The side length s must satisfy: s * sqrt(3) >= max_width / 2 for each direction? 
        No, that's not quite right either.
        
        Let me think differently. For a hexagon centered at c with side s and angle a:
        For each of the 6 normals n_i (unit vectors at angles a+30+60i):
            The constraint is: for all points p, dot(p - c, n_i) <= s * sqrt(3) / 2
            And: dot(p - c, n_i) >= -s * sqrt(3) / 2  (the opposite edge)
        
        So: |dot(p - c, n_i)| <= s * sqrt(3) / 2 for all p, i.
        
        This means: max_p |dot(p - c, n_i)| <= s * sqrt(3) / 2 for all i.
        
        So s >= max_i [max_p |dot(p - c, n_i)|] / (sqrt(3)/2)
        
        To minimize s, choose c to minimize max_i [max_p |dot(p - c, n_i)|].
        
        This is the minimum enclosing regular hexagon problem for a fixed orientation.
        The optimal c is the center of the minimum enclosing circle of the 6 "extreme" points?
        Not exactly. Let's just use the centroid and accept a slightly larger hexagon,
        or do a few iterations of optimization.
        """
        points = np.array(points)
        hex_angle_rad = np.radians(hex_angle_deg)
        
        # 6 normal directions
        normal_angles = hex_angle_rad + np.radians(30 + 60 * np.arange(6))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)
        
        # For a given center c, the required s is:
        # s = max_i max_p |dot(p - c, n_i)| / (sqrt3/2)
        
        # To find optimal c, we can use the fact that the optimal c minimizes
        # the maximum of the 6 "radii" r_i(c) = max_p |dot(p - c, n_i)|
        
        # A good initial guess is the centroid
        c = np.mean(points, axis=0)
        
        # Iterate: for each direction, find the extreme point, then adjust c
        # Actually, let's just use the centroid for now and compute s
        diffs = points - c  # (n, 2)
        projections = np.abs(diffs @ normals.T)  # (n, 6)
        max_projections = np.max(projections, axis=0)  # (6,)
        s = np.max(max_projections) / (sqrt3 / 2)
        
        return s, c

    best_s = np.inf
    best_result = None

    for config_idx, config in enumerate(configs):
        if time.time() > deadline:
            break

        # Convert axial to cartesian
        centers = np.array([axial_to_cartesian(q, r) for q, r in config])

        # Center at origin
        centers = centers - np.mean(centers, axis=0)

        # Search over rotation angles
        best_angle_local = 0
        best_s_local = np.inf

        # Coarse search
        for angle_deg in np.arange(0, 60, 1.0):
            a_rad = np.radians(angle_deg)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated_centers = centers @ rot_matrix.T
            s, c = compute_min_containing_hex_side(rotated_centers, hex_angle_deg=0)
            if s < best_s_local:
                best_s_local = s
                best_angle_local = angle_deg

        # Fine search
        for angle_deg in np.arange(best_angle_local - 1, best_angle_local + 1, 0.05):
            a_rad = np.radians(angle_deg)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated_centers = centers @ rot_matrix.T
            s, c = compute_min_containing_hex_side(rotated_centers, hex_angle_deg=0)
            if s < best_s_local:
                best_s_local = s
                best_angle_local = angle_deg

        if best_s_local < best_s:
            best_s = best_s_local
            best_result = (config_idx, best_angle_local, centers.copy())

    if best_result is None:
        inner_hex_data = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0],
            [-1.25, -2.17, 0], [1.25, -2.17, 0],
            [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0],
            [0, -4, 0]
        ])
        outer_hex_data = np.array([0, 0, 0])
        return inner_hex_data, outer_hex_data, 8.0

    config_idx, best_angle, base_centers = best_result

    # Nelder-Mead refinement
    def objective_full(params):
        angle_deg = params[0]
        cx = params[1]
        cy = params[2]

        a_rad = np.radians(angle_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_centers = base_centers @ rot_matrix.T + np.array([cx, cy])
        s, c = compute_min_containing_hex_side(rotated_centers, hex_angle_deg=0)
        return s

    x0 = np.array([best_angle, 0.0, 0.0])

    try:
        result = minimize(objective_full, x0, method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14,
                                   'adaptive': True})
        if result.fun < best_s:
            best_angle = result.x[0]
            cx_opt = result.x[1]
            cy_opt = result.x[2]
            best_s = result.fun
    except:
        cx_opt, cy_opt = 0.0, 0.0

    # Final computation
    a_rad = np.radians(best_angle)
    cos_a = np.cos(a_rad)
    sin_a = np.sin(a_rad)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_centers = base_centers @ rot_matrix.T + np.array([cx_opt, cy_opt])

    s_final, outer_center = compute_min_containing_hex_side(rotated_centers, hex_angle_deg=0)

    # Build output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = rotated_centers[i, 0]
        inner_hex_data[i, 1] = rotated_centers[i, 1]
        inner_hex_data[i, 2] = best_angle

    outer_hex_data = np.array([outer_center[0], outer_center[1], 0.0])

    return inner_hex_data, outer_hex_data, s_final


# EVOLVE-BLOCK-END
