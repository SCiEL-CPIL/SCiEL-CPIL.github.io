# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, side, angle_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

def _compute_min_side(inner_positions, outer_center, outer_angle_deg):
    """
    Compute the minimum side length s of a regular hexagon centered at outer_center
    with the given angle that contains all 12 inner unit hexagons (all with angle 0).
    Vectorized for speed.
    """
    outer_angle = np.deg2rad(outer_angle_deg)
    normal_angles = outer_angle + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6,2)

    unit_angles = np.deg2rad(60 * np.arange(6))
    unit_cos = np.cos(unit_angles)
    unit_sin = np.sin(unit_angles)

    # All 72 vertices of the 12 inner unit hexagons
    all_verts = np.zeros((72, 2))
    all_verts[0::6, 0] = inner_positions[:, 0] + unit_cos[0]
    all_verts[0::6, 1] = inner_positions[:, 1] + unit_sin[0]
    all_verts[1::6, 0] = inner_positions[:, 0] + unit_cos[1]
    all_verts[1::6, 1] = inner_positions[:, 1] + unit_sin[1]
    all_verts[2::6, 0] = inner_positions[:, 0] + unit_cos[2]
    all_verts[2::6, 1] = inner_positions[:, 1] + unit_sin[2]
    all_verts[3::6, 0] = inner_positions[:, 0] + unit_cos[3]
    all_verts[3::6, 1] = inner_positions[:, 1] + unit_sin[3]
    all_verts[4::6, 0] = inner_positions[:, 0] + unit_cos[4]
    all_verts[4::6, 1] = inner_positions[:, 1] + unit_sin[4]
    all_verts[5::6, 0] = inner_positions[:, 0] + unit_cos[5]
    all_verts[5::6, 1] = inner_positions[:, 1] + unit_sin[5]

    rel = all_verts - outer_center
    max_projs = np.max(rel @ normals.T, axis=0)
    max_inradius = np.max(max_projs)
    if max_inradius < 0:
        max_inradius = 0
    s = 2.0 * max_inradius / np.sqrt(3.0)
    return s

def _objective(params, outer_angle_deg):
    inner_pos = params[:24].reshape(12, 2)
    oc = params[24:26]
    return _compute_min_side(inner_pos, oc, outer_angle_deg)

def _check_separation(hex1_verts, hex2_verts, tol=1e-6):
    """Separating axis test: return True if hexagons are separated (no overlap)."""
    axes = []
    for verts in (hex1_verts, hex2_verts):
        for i in range(6):
            edge = verts[(i + 1) % 6] - verts[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                axes.append(n / norm)
    for axis in axes:
        proj1 = hex1_verts @ axis
        proj2 = hex2_verts @ axis
        if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
            return True
    return False

def _vertex_inside_outer(v, outer_verts, tol=1e-6):
    """Check if vertex v is inside or on the boundary of outer hexagon."""
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
        if cross < -tol:
            return False
    return True

def _validate(inner_hex_data, outer_hex_data, s):
    """Validate the packing. Return True if valid."""
    if s < np.sqrt(12) - 1e-8:
        return False
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False
    if not np.isfinite(s):
        return False
    outer_verts = _hex_vertices(outer_hex_data[0], outer_hex_data[1], s, outer_hex_data[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    inner_verts_list = []
    for i in range(12):
        verts = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        inner_verts_list.append(verts)
        for k in range(6):
            edge_len = np.linalg.norm(verts[(k + 1) % 6] - verts[k])
            if abs(edge_len - 1.0) > 1e-8:
                return False
    for i in range(12):
        for j in range(i + 1, 12):
            if not _check_separation(inner_verts_list[i], inner_verts_list[j]):
                return False
    for i in range(12):
        for v in inner_verts_list[i]:
            if not _vertex_inside_outer(v, outer_verts):
                return False
    return True

def _make_hex_lattice(n_rings, spacing):
    """Hexagonal lattice: 1 center + rings. Returns first 12 positions centered at origin."""
    positions = [[0.0, 0.0]]
    for ring in range(1, n_rings + 1):
        for i in range(6 * ring):
            angle = np.deg2rad(60.0 * (i // ring + 1) + (i % ring) * (60.0 / ring))
            positions.append([ring * spacing * np.cos(angle),
                              ring * spacing * np.sin(angle)])
    positions = np.array(positions[:12])
    center = positions.mean(axis=0)
    positions -= center
    return positions

def _make_grid(rows, cols, dx, dy):
    """Rectangular grid of positions, centered at origin."""
    positions = []
    for r in range(rows):
        for c in range(cols):
            x = (c - (cols - 1) / 2.0) * dx
            y = (r - (rows - 1) / 2.0) * dy
            if r % 2 == 1:
                x += dx / 2
            positions.append([x, y])
    positions = np.array(positions[:12])
    center = positions.mean(axis=0)
    positions -= center
    return positions

def _gradient_descent(init_params, outer_angle_deg, lr=0.05, n_steps=1500, deadline=None):
    """Simple finite-difference gradient descent with adaptive learning rate."""
    params = init_params.copy()
    best_params = params.copy()
    best_val = _objective(params, outer_angle_deg)

    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break

        grad = np.zeros_like(params)
        h = 1e-6
        for i in range(len(params)):
            params[i] += h
            f_plus = _objective(params, outer_angle_deg)
            params[i] -= 2 * h
            f_minus = _objective(params, outer_angle_deg)
            params[i] += h
            grad[i] = (f_plus - f_minus) / (2 * h)

        params -= lr * grad
        val = _objective(params, outer_angle_deg)

        if val < best_val:
            best_val = val
            best_params = params.copy()

        if val > best_val * 1.01:
            lr *= 0.5
        else:
            lr = min(lr * 1.05, 0.1)

    return best_params, best_val

def _nelder_mead(init_params, outer_angle_deg, maxiter=3000, deadline=None):
    """Nelder-Mead optimization with deadline guard."""
    def obj(p):
        if deadline is not None and time.time() > deadline:
            return 1e10
        return _objective(p, outer_angle_deg)
    result = minimize(obj, init_params, method='Nelder-Mead',
                      options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-11})
    return result.x, result.fun

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner_hex_data, outer_hex_data, s)
    """
    start_time = time.time()
    deadline = start_time + 580.0  # 50s safety margin before 630s limit

    sqrt3 = np.sqrt(3)

    # Diverse initial configurations
    initial_configs = []

    # Hexagonal lattices with different spacings (match hexagonal symmetry)
    for spacing in [2.3, 2.5, 2.7, 3.0]:
        initial_configs.append(_make_hex_lattice(2, spacing))

    # Grid configurations
    for dx, dy in [(2.3, sqrt3 * 0.9), (2.5, sqrt3), (2.7, sqrt3 * 1.1), (3.0, sqrt3 * 1.2)]:
        initial_configs.append(_make_grid(3, 4, dx, dy))
        initial_configs.append(_make_grid(4, 3, dx, dy))

    # Random perturbations of the base lattice
    rng = np.random.default_rng(42)
    for _ in range(5):
        base = _make_hex_lattice(2, 2.5)
        base = base + rng.normal(0, 0.3, size=(12, 2))
        initial_configs.append(base)

    # Try different outer angles
    outer_angles = [0, 30, 60, 90, 120, 150]

    best_s = np.inf
    best_params = None
    best_angle = 0

    for angle_deg in outer_angles:
        if time.time() > deadline - 60:
            break

        for init_pos in initial_configs:
            if time.time() > deadline - 60:
                break

            init_params = np.concatenate([init_pos.flatten(), np.array([0.0, 0.0])])

            # Phase 1: fast gradient descent to get close to a local minimum
            gd_params, gd_val = _gradient_descent(init_params, angle_deg,
                                                  lr=0.05, n_steps=800,
                                                  deadline=deadline - 60)

            # Phase 2: Nelder-Mead polish
            nm_params, nm_val = _nelder_mead(gd_params, angle_deg,
                                              maxiter=2500, deadline=deadline - 60)

            if nm_val < best_s:
                best_s = nm_val
                best_params = nm_params
                best_angle = angle_deg

    # Final refinement if time remains
    if best_params is not None and time.time() < deadline - 30:
        result = minimize(lambda p: _objective(p, best_angle), best_params,
                          method='Nelder-Mead',
                          options={'maxiter': 6000, 'xatol': 1e-10, 'fatol': 1e-12})
        if result.fun < best_s:
            best_s = result.fun
            best_params = result.x

    # Extract best solution
    inner_positions = best_params[:24].reshape(12, 2)
    outer_center = best_params[24:26]

    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = inner_positions[i, 0]
        inner_hex_data[i, 1] = inner_positions[i, 1]
        inner_hex_data[i, 2] = 0.0

    outer_hex_data = np.array([outer_center[0], outer_center[1], float(best_angle)])
    s_final = best_s

    # Validate; fall back to a known valid configuration if needed
    if not _validate(inner_hex_data, outer_hex_data, s_final):
        positions_fb = _make_hex_lattice(2, 3.0)
        inner_hex_data = np.zeros((12, 3))
        for i in range(12):
            inner_hex_data[i, 0] = positions_fb[i, 0]
            inner_hex_data[i, 1] = positions_fb[i, 1]
            inner_hex_data[i, 2] = 0.0
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        s_final = _compute_min_side(positions_fb, np.array([0.0, 0.0]), 0.0) + 0.2

    return inner_hex_data, outer_hex_data, s_final
# EVOLVE-BLOCK-END
