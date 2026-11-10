# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, angle_deg):
    a = np.deg2rad(angle_deg)
    return np.array([[cx + s * np.cos(a + np.deg2rad(60 * i)),
                      cy + s * np.sin(a + np.deg2rad(60 * i))] for i in range(6)])

def _outer_vertices(cx, cy, s, angle_deg):
    a = np.deg2rad(angle_deg)
    return np.array([[cx + s * np.cos(a + np.deg2rad(60 * i)),
                      cy + s * np.sin(a + np.deg2rad(60 * i))] for i in range(6)])

def _point_in_hex(p, cx, cy, s, angle_deg):
    verts = _hex_vertices(cx, cy, s, angle_deg)
    ok = True
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i + 1) % 6]
        if np.cross(p2 - p1, p - p1) < -1e-9:
            ok = False
            break
    return ok

def _hex_separated(verts1, verts2):
    """Check if two unit hexagons (given as 6x2 arrays) are separated (not overlapping)."""
    n1 = len(verts1)
    n2 = len(verts2)
    for i in range(n1):
        p1 = verts1[i]
        p2 = verts1[(i + 1) % n1]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        normal = normal / (np.linalg.norm(normal) + 1e-15)
        proj1 = np.dot(verts1, normal)
        proj2 = np.dot(verts2, normal)
        if proj1.max() < proj2.min() - 1e-6 or proj2.max() < proj1.min() - 1e-6:
            return True
    return False

def _check_validity(centers, angles_deg, side, outer_angle_deg, tol=1e-6):
    """Full validity check."""
    if not np.all(np.isfinite(centers)):
        return False
    if not np.all(np.isfinite(angles_deg)):
        return False
    if not np.isfinite(side):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False
    if not np.isfinite(outer_angle_deg):
        return False
    outer_verts = _outer_vertices(0, 0, side, outer_angle_deg)
    for i in range(11):
        for j in range(11):
            if i == j:
                continue
            if not _hex_separated(_hex_vertices(centers[i,0], centers[i,1], 1.0, angles_deg[i]),
                                  _hex_vertices(centers[j,0], centers[j,1], 1.0, angles_deg[j])):
                return False
    for i in range(11):
        v = _hex_vertices(centers[i,0], centers[i,1], 1.0, angles_deg[i])
        for p in v:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1) % 6]
                if np.cross(p2 - p1, p - p1) < -tol:
                    return False
    return True

def _compute_min_side(centers, angles_deg, outer_angle_deg, tol=1e-9):
    """Compute the minimum side length of an outer hexagon containing all inner hexagons."""
    outer_verts = _outer_vertices(0, 0, 1.0, outer_angle_deg)
    normals = np.array([[-(outer_verts[(i+1)%6][1] - outer_verts[i][1]),
                         (outer_verts[(i+1)%6][0] - outer_verts[i][0])] for i in range(6)])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / norms
    max_h = 0
    for i in range(11):
        v = _hex_vertices(centers[i,0], centers[i,1], 1.0, angles_deg[i])
        for k in range(6):
            proj = np.dot(v, normals[k])
            h = proj.max()
            if h > max_h:
                max_h = h
    return (2.0 / np.sqrt(3.0)) * max_h

def _objective(params, outer_angle_deg):
    """Objective: minimize side length. params = [x0,y0,a0, x1,y1,a1, ..., x10,y10,a10]."""
    centers = params[0::3]
    angles_deg = params[1::3]
    side = _compute_min_side(centers, angles_deg, outer_angle_deg)
    # Add penalties for overlap
    penalty = 0
    for i in range(11):
        for j in range(i+1, 11):
            if not _hex_separated(_hex_vertices(centers[i], 0, 1.0, angles_deg[i]),
                                  _hex_vertices(centers[j], 0, 1.0, angles_deg[j])):
                # Compute penetration depth
                v1 = _hex_vertices(centers[i], 0, 1.0, angles_deg[i])
                v2 = _hex_vertices(centers[j], 0, 1.0, angles_deg[j])
                # Simple penalty: distance between centers
                d = np.linalg.norm(centers[i] - centers[j])
                penalty += max(0, 2.0 - d) ** 2
    return side + 0.1 * penalty

def hexagon_packing_11():
    start_time = time.time()
    deadline = 630.0
    safety_margin = 60.0

    best_side = float('inf')
    best_params = None
    best_outer_angle = 0

    # Initial guess: honeycomb-like arrangement
    d = np.sqrt(3)
    initial_centers = np.array([
        [0, 0],
        [d, 0],
        [d/2, d * np.sqrt(3)/2],
        [-d/2, d * np.sqrt(3)/2],
        [-d, 0],
        [-d/2, -d * np.sqrt(3)/2],
        [d/2, -d * np.sqrt(3)/2],
        [2*d, 0],
        [3*d/2, d * np.sqrt(3)/2],
        [3*d/2, -d * np.sqrt(3)/2],
        [1.5*d, 0],
    ])
    initial_angles = np.zeros(11)
    initial_params = np.concatenate([initial_centers.flatten(), initial_angles])

    # Try multiple outer angles
    outer_angles = np.linspace(0, 60, 13)  # 0 to 60 degrees in steps

    for oa_deg in outer_angles:
        if time.time() > deadline - safety_margin:
            break

        # Use the initial guess as starting point
        res = minimize(_objective, initial_params, args=(oa_deg,),
                      method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10,
                               'adaptive': True})
        if time.time() > deadline - safety_margin:
            break

        params = res.x
        centers = params[0::3]
        angles_deg = params[1::3]
        side = _compute_min_side(centers, angles_deg, oa_deg)

        if _check_validity(centers, angles_deg, side, oa_deg, tol=1e-6):
            if side < best_side:
                best_side = side
                best_params = params.copy()
                best_outer_angle = oa_deg

    # Try additional random restarts if time permits
    np.random.seed(42)
    n_restarts = 20
    for r in range(n_restarts):
        if time.time() > deadline - safety_margin:
            break

        # Perturb the initial guess
        centers = initial_centers + np.random.randn(11, 2) * 0.5
        angles = np.random.randn(11) * 10
        params = np.concatenate([centers.flatten(), angles])

        oa_deg = np.random.uniform(0, 60)

        res = minimize(_objective, params, args=(oa_deg,),
                      method='Nelder-Mead',
                      options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10,
                               'adaptive': True})
        if time.time() > deadline - safety_margin:
            break

        params = res.x
        centers = params[0::3]
        angles_deg = params[1::3]
        side = _compute_min_side(centers, angles_deg, oa_deg)

        if _check_validity(centers, angles_deg, side, oa_deg, tol=1e-6):
            if side < best_side:
                best_side = side
                best_params = params.copy()
                best_outer_angle = oa_deg

    if best_params is None:
        return _fallback()

    centers = best_params[0::3]
    angles_deg = best_params[1::3]
    inner = np.column_stack([centers, angles_deg])
    outer = np.array([0.0, 0.0, best_outer_angle])
    return inner, outer, best_side

def _fallback():
    d = np.sqrt(3)
    inner = np.array([
        [0, 0, 0],
        [d, 0, 0],
        [d/2, d*np.sqrt(3)/2, 0],
        [-d/2, d*np.sqrt(3)/2, 0],
        [-d, 0, 0],
        [-d/2, -d*np.sqrt(3)/2, 0],
        [d/2, -d*np.sqrt(3)/2, 0],
        [2*d, 0, 0],
        [3*d/2, d*np.sqrt(3)/2, 0],
        [3*d/2, -d*np.sqrt(3)/2, 0],
        [1.5*d, 0, 0],
    ])
    outer = np.array([0.0, 0.0, 0.0])
    # Compute actual side
    side = _compute_min_side(inner[:, 0:2], inner[:, 2], 0.0)
    return inner, outer, side

# EVOLVE-BLOCK-END
