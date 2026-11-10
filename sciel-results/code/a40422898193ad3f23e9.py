# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    a_rad = np.deg2rad(a_deg)
    angles = a_rad + np.deg2rad(60 * np.arange(6))
    x = cx + s * np.cos(angles)
    y = cy + s * np.sin(angles)
    return np.column_stack((x, y))

def _check_separation(hex1, hex2):
    def get_normals(vertices):
        normals = np.zeros((6, 2))
        for i in range(6):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            normals[i] = n
        return normals

    n1 = get_normals(hex1)
    n2 = get_normals(hex2)
    all_normals = np.vstack([n1, n2])
    norms = np.linalg.norm(all_normals, axis=1, keepdims=True)
    all_normals = all_normals / norms

    for n in all_normals:
        proj1 = hex1 @ n
        proj2 = hex2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
            return True
    return False

def _required_outer_size(all_vertices, outer_center, outer_angle_deg):
    a_rad = np.deg2rad(outer_angle_deg)
    normal_angles = a_rad + np.deg2rad(30) + np.deg2rad(60 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    rel_vertices = all_vertices - outer_center
    proj = rel_vertices @ normals.T
    max_proj = np.max(proj)
    if max_proj < 0:
        return 0.0
    S = (2.0 / np.sqrt(3.0)) * max_proj
    return S

def _validate(inner_hex_data, outer_hex_data, side):
    """Full validation check. Returns True if valid."""
    # Check all values finite
    if not (np.all(np.isfinite(inner_hex_data)) and np.all(np.isfinite(outer_hex_data)) and np.isfinite(side)):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False

    # Build inner hexagon vertices
    inner_verts = []
    for i in range(11):
        cx, cy, ang = inner_hex_data[i]
        inner_verts.append(_hex_vertices(cx, cy, 1.0, ang))

    # Check pairwise separation
    for i in range(11):
        for j in range(i + 1, 11):
            if not _check_separation(inner_verts[i], inner_verts[j]):
                return False

    # Check outer containment
    ox, oy, oang = outer_hex_data
    outer_verts = _hex_vertices(ox, oy, side, oang)
    # Shift by outer center
    shifted_outer = outer_verts - np.array([ox, oy])
    shifted_inner = []
    for v in inner_verts:
        shifted_inner.append(v - np.array([ox, oy]))

    # For each inner vertex, check it's inside outer hexagon
    for iv in shifted_inner:
        for p in iv:
            for k in range(6):
                p1 = shifted_outer[k]
                p2 = shifted_outer[(k + 1) % 6]
                cross_val = np.cross(p2 - p1, p - p1)
                if cross_val < -1e-6:
                    return False

    # Check edge lengths of shifted inner hexagons
    for sv in shifted_inner:
        for k in range(6):
            d = np.linalg.norm(sv[(k+1)%6] - sv[k])
            if abs(d - 1.0) > 1e-8:
                return False

    return True

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 590.0

    # Fixed rigid cluster: flower (7 hexagons) + 4 satellites
    # Flower: center at origin, 6 neighbors at distance sqrt(3) at angles 0,60,120,180,240,300
    h = np.sqrt(3)

    # Define the 11 hexagon centers in a fixed configuration
    # Use a compact arrangement:
    # Ring 0: 1 center
    # Ring 1: 6 hexagons at distance h
    # Ring 2: 4 hexagons at distance 2*h, placed at angles 30, 150, 210, 330 (between ring-1 hexagons)
    # This creates a non-symmetric pattern that may pack tighter

    centers = np.array([
        [0.0, 0.0],
        [h, 0.0],
        [h*np.cos(np.deg2rad(60)), h*np.sin(np.deg2rad(60))],
        [h*np.cos(np.deg2rad(120)), h*np.sin(np.deg2rad(120))],
        [h*np.cos(np.deg2rad(180)), h*np.sin(np.deg2rad(180))],
        [h*np.cos(np.deg2rad(240)), h*np.sin(np.deg2rad(240))],
        [h*np.cos(np.deg2rad(300)), h*np.sin(np.deg2rad(300))],
        [2*h*np.cos(np.deg2rad(30)), 2*h*np.sin(np.deg2rad(30))],
        [2*h*np.cos(np.deg2rad(150)), 2*h*np.sin(np.deg2rad(150))],
        [2*h*np.cos(np.deg2rad(210)), 2*h*np.sin(np.deg2rad(210))],
        [2*h*np.cos(np.deg2rad(330)), 2*h*np.sin(np.deg2rad(330))],
    ])

    # All inner hexagons have angle 0 (flat-top)
    inner_angles = np.zeros(11)

    # Build all inner vertices (for fast reuse)
    inner_verts = []
    for i in range(11):
        inner_verts.append(_hex_vertices(centers[i, 0], centers[i, 1], 1.0, 0.0))

    all_inner_vertices = np.vstack(inner_verts)

    # Pre-check: verify the fixed configuration is internally valid
    valid_internal = True
    for i in range(11):
        for j in range(i + 1, 11):
            if not _check_separation(inner_verts[i], inner_verts[j]):
                valid_internal = False
                break
        if not valid_internal:
            break

    if not valid_internal:
        # Fallback: use a looser configuration
        # 3-4-4 grid with larger spacing
        sp = 2.05  # slightly more than 2 to ensure separation
        centers = np.array([
            [0, 0], [-sp, 0], [sp, 0],
            [-sp/2, h], [sp/2, h], [-3*sp/2, h], [3*sp/2, h],
            [0, 2*h], [-sp, 2*h], [sp, 2*h],
            [-sp/2, 3*h], [sp/2, 3*h]
        ])
        inner_verts = []
        for i in range(11):
            inner_verts.append(_hex_vertices(centers[i, 0], centers[i, 1], 1.0, 0.0))
        all_inner_vertices = np.vstack(inner_verts)

    # Now optimize: outer_center (2), outer_angle (1), outer_side (1)
    # Objective: minimize outer_side subject to all inner vertices being inside outer hexagon

    def objective(params):
        if time.time() > deadline:
            return 1e9
        ocx, ocy, oang, S = params
        # Compute required S for given center and angle
        required_S = _required_outer_size(all_inner_vertices, np.array([ocx, ocy]), oang)
        # Return the side length (we want to minimize it)
        return required_S

    # Multiple restarts for outer parameters
    best_S = float('inf')
    best_params = None

    # Initial guesses: center at centroid of inner hexagons, angle at multiples of 30
    centroid = np.mean(centers, axis=0)

    angle_guesses = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    center_offsets = [0.0, 0.1, -0.1, 0.2, -0.2, 0.05, -0.05]

    n_restarts = 0
    max_restarts = 50

    for ang in angle_guesses:
        for dx in center_offsets:
            for dy in center_offsets:
                if time.time() > deadline or n_restarts >= max_restarts:
                    break
                x0 = [centroid[0] + dx, centroid[1] + dy, float(ang), 5.0]
                n_restarts += 1

                try:
                    res = minimize(objective, x0, method='Nelder-Mead',
                                   options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                    if res.fun < best_S and res.fun < 1e8:
                        best_S = res.fun
                        best_params = res.x
                except Exception:
                    pass
            if time.time() > deadline or n_restarts >= max_restarts:
                break
        if time.time() > deadline or n_restarts >= max_restarts:
            break

    if best_params is None:
        # Fallback: use centroid and angle 0
        ocx, ocy = centroid
        oang = 0.0
        best_S = _required_outer_size(all_inner_vertices, np.array([ocx, ocy]), oang)
        best_params = [ocx, ocy, oang, best_S]

    # Extract best solution
    ocx, ocy, oang, S = best_params

    # Build return arrays
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i] = [centers[i, 0], centers[i, 1], 0.0]

    outer_hex_data = np.array([ocx, ocy, oang])

    # Validate the result
    if not _validate(inner_hex_data, outer_hex_data, S):
        # If invalid, try with slightly larger S
        S += 1e-5
        outer_hex_data = np.array([ocx, ocy, oang])
        if not _validate(inner_hex_data, outer_hex_data, S):
            # Last resort: very large hexagon
            S = 10.0
            inner_hex_data = np.array([
                [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
                [-1.25, 2.17, 0], [1.25, 2.17, 0],
                [-1.25, -2.17, 0], [1.25, -2.17, 0],
                [-3.75, 2.17, 0], [3.75, 2.17, 0],
                [-3.75, -2.17, 0], [3.75, -2.17, 0]
            ])
            outer_hex_data = np.array([0, 0, 0])

    return inner_hex_data, outer_hex_data, S

# EVOLVE-BLOCK-END
