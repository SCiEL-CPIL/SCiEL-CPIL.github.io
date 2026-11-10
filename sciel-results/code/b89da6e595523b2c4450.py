import numpy as np
import scipy.optimize as so
import time

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    timeout = 580.0

    def get_hex_vertices(cx, cy, s, a_deg):
        a_rad = np.deg2rad(a_deg)
        angles = a_rad + np.deg2rad(60) * np.arange(6)
        x = cx + s * np.cos(angles)
        y = cy + s * np.sin(angles)
        return np.column_stack((x, y))

    def hex_overlap_penalty(v1, v2):
        c1 = np.mean(v1, axis=0)
        c2 = np.mean(v2, axis=0)
        if np.linalg.norm(c1 - c2) > 2.5:
            return 0.0
        axes = []
        for v in [v1, v2]:
            for i in range(6):
                p1 = v[i]
                p2 = v[(i+1)%6]
                edge = p2 - p1
                n = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(n)
                if norm > 1e-9:
                    n = n / norm
                    axes.append(n)
        min_dist = float('inf')
        for n in axes:
            proj1 = np.dot(v1, n)
            proj2 = np.dot(v2, n)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            if max1 < min2:
                gap = min2 - max1
            elif max2 < min1:
                gap = min1 - max2
            else:
                gap = 0.0
            min_dist = min(min_dist, gap)
        if min_dist < 0:
            return -min_dist * 10.0
        else:
            return 0.0

    def objective(params):
        if time.time() - start_time > timeout:
            return 1e10
        hexes = params.reshape(11, 3)
        normal_angles = np.deg2rad(30 + 60 * np.arange(6))
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        max_proj = 0
        verts = []
        for i in range(11):
            x, y, a = hexes[i]
            v = get_hex_vertices(x, y, 1.0, a)
            verts.append(v)
            projs = np.dot(v, normals.T)
            max_proj = max(max_proj, np.max(np.abs(projs)))
        S = 2.0 / np.sqrt(3.0) * max_proj
        penalty = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                penalty += hex_overlap_penalty(verts[i], verts[j])
        return S + penalty

    # Initial guess: 7 in center, 4 scattered
    r = np.sqrt(3)
    init = np.zeros((11, 3))
    init[0] = [0, 0, 0]
    for i in range(6):
        angle = i * 60
        init[i+1] = [r * np.cos(np.deg2rad(angle)), r * np.sin(np.deg2rad(angle)), 0]
    np.random.seed(42)
    for i in range(7, 11):
        angle = np.random.uniform(0, 360)
        dist = 3.0
        init[i] = [dist * np.cos(np.deg2rad(angle)), dist * np.sin(np.deg2rad(angle)), 0]

    best_params = init
    best_score = objective(init)
    
    options = {'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5}
    
    # Run optimization
    while time.time() - start_time < timeout:
        try:
            res = so.minimize(objective, best_params, method='Nelder-Mead', options=options)
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
            # If we found a valid packing, we can break or continue to refine
            # Check penalty
            hexes = best_params.reshape(11, 3)
            verts = [get_hex_vertices(hexes[i][0], hexes[i][1], 1.0, hexes[i][2]) for i in range(11)]
            penalty = 0
            for i in range(11):
                for j in range(i+1, 11):
                    penalty += hex_overlap_penalty(verts[i], verts[j])
            if penalty < 1e-6:
                # Valid packing found. Try to optimize S only by adjusting positions slightly?
                # Or just accept.
                break
        except Exception:
            pass

    hexes = best_params.reshape(11, 3)
    normal_angles = np.deg2rad(30 + 60 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    max_proj = 0
    for i in range(11):
        x, y, a = hexes[i]
        v = get_hex_vertices(x, y, 1.0, a)
        projs = np.dot(v, normals.T)
        max_proj = max(max_proj, np.max(np.abs(projs)))
    S = 2.0 / np.sqrt(3.0) * max_proj
    
    inner_hex_data = hexes
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = S
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
