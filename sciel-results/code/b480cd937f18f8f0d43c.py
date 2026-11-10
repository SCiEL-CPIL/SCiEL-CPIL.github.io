# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 630.0
_SAFETY_MARGIN = 30.0


def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _min_containing_hex_side(centers):
    """
    Compute the minimum side length s of a regular hexagon centered at (0,0), angle 0,
    that contains all unit hexagons with given centers.
    """
    n = len(centers)
    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)

    # All vertices for all hexagons: (n, 6, 2)
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        verts[i, :, 0] = centers[i, 0] + base_cos
        verts[i, :, 1] = centers[i, 1] + base_sin

    # Outward normals at angles 30, 90, 150, 210, 270, 330
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)

    verts_flat = verts.reshape(-1, 2)  # (36, 2)
    projections = normals @ verts_flat.T  # (6, 36)
    max_proj_per_normal = np.max(projections, axis=1)  # (6,)
    max_proj = np.max(max_proj_per_normal)  # scalar

    s = (2.0 / np.sqrt(3.0)) * max_proj
    return s


def _separation_violation(centers):
    """
    Compute total separation violation (how much pairs overlap).
    Returns 0 if all pairs are separated, positive value otherwise.
    """
    n = len(centers)
    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)

    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)

    all_verts = np.zeros((n, 6, 2))
    for i in range(n):
        all_verts[i, :, 0] = centers[i, 0] + base_cos
        all_verts[i, :, 1] = centers[i, 1] + base_sin

    total_violation = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]
            vj = all_verts[j]

            proj_i = vi @ normals.T
            proj_j = vj @ normals.T

            min_i = np.min(proj_i, axis=1)
            max_i = np.max(proj_i, axis=1)
            min_j = np.min(proj_j, axis=1)
            max_j = np.max(proj_j, axis=1)

            overlaps = np.minimum(max_i, max_j) - np.maximum(min_i, min_j)
            max_overlap = np.max(overlaps)

            if max_overlap > 0:
                total_violation += max_overlap ** 2

    return total_violation


def _check_separation(centers, tol=1e-6):
    """Check if all pairs of unit hexagons are separated using SAT."""
    n = len(centers)
    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)

    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])

    all_verts = np.zeros((n, 6, 2))
    for i in range(n):
        all_verts[i, :, 0] = centers[i, 0] + base_cos
        all_verts[i, :, 1] = centers[i, 1] + base_sin

    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]
            vj = all_verts[j]

            proj_i = vi @ normals.T
            proj_j = vj @ normals.T

            min_i = np.min(proj_i, axis=1)
            max_i = np.max(proj_i, axis=1)
            min_j = np.min(proj_j, axis=1)
            max_j = np.max(proj_j, axis=1)

            separated = np.any(max_i < min_j - tol) or np.any(max_j < min_i - tol)
            if not separated:
                return False
    return True


def _make_hex_lattice_12():
    """Create a hexagonal lattice arrangement of 12 hexagons (non-overlapping by construction)."""
    sqrt3 = np.sqrt(3)
    return np.array([
        [0.0, 0.0],
        [1.5, 0.866025],
        [1.5, -0.866025],
        [0.0, 1.732051],
        [0.0, -1.732051],
        [-1.5, 0.866025],
        [-1.5, -0.866025],
        [3.0, 0.0],
        [2.25, 2.598076],
        [0.75, 3.464102],
        [-1.5, 2.598076],
        [-3.0, 0.0],
    ])


def _make_compact_initial():
    """A more compact initial arrangement (hexagonal lattice, 12 cells)."""
    sqrt3 = np.sqrt(3)
    return np.array([
        [0.0, 0.0],
        [sqrt3, 0.0],
        [sqrt3/2, 1.5],
        [-sqrt3/2, 1.5],
        [-sqrt3, 0.0],
        [-sqrt3/2, -1.5],
        [sqrt3/2, -1.5],
        [1.5*sqrt3, 1.5],
        [1.5*sqrt3, -1.5],
        [0.0, 3.0],
        [-1.5*sqrt3, 1.5],
        [-1.5*sqrt3, -1.5],
    ])


def _make_spread_initial():
    """A spread-out initial arrangement."""
    return np.array([
        [0.0, 0.0],
        [-2.5, 0.0],
        [2.5, 0.0],
        [-1.25, 2.165],
        [1.25, 2.165],
        [-1.25, -2.165],
        [1.25, -2.165],
        [-3.75, 2.165],
        [3.75, 2.165],
        [-3.75, -2.165],
        [3.75, -2.165],
        [0.0, -4.33],
    ])


def _force_directed_relax(centers, n_iters=500, dt=0.05, repulsion_strength=1.0):
    """
    Force-directed relaxation: push overlapping hexagons apart using a simple
    repulsive force model based on separation violation gradients.
    """
    centers = centers.copy().astype(float)
    n = len(centers)

    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)

    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])

    for it in range(n_iters):
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - _SAFETY_MARGIN:
            break

        forces = np.zeros_like(centers)

        for i in range(n):
            for j in range(i + 1, n):
                vi = centers[i] + np.column_stack([base_cos, base_sin])
                vj = centers[j] + np.column_stack([base_cos, base_sin])

                proj_i = vi @ normals.T
                proj_j = vj @ normals.T

                min_i = np.min(proj_i, axis=1)
                max_i = np.max(proj_i, axis=1)
                min_j = np.min(proj_j, axis=1)
                max_j = np.max(proj_j, axis=1)

                # For each normal axis, compute overlap and force direction
                for k in range(6):
                    overlap = min(max_i[k], max_j[k]) - max(min_i[k], min_j[k])
                    if overlap > 0:
                        # Push apart along this normal
                        n_dir = normals[k]
                        # Determine which hex is "on top"
                        if max_i[k] > max_j[k]:
                            forces[i] += overlap * n_dir * repulsion_strength
                            forces[j] -= overlap * n_dir * repulsion_strength
                        else:
                            forces[j] += overlap * n_dir * repulsion_strength
                            forces[i] -= overlap * n_dir * repulsion_strength

        centers += dt * forces
        # Decay step size
        dt *= 0.995

    return centers


def _barycentric_scale(centers, alpha):
    """Scale centers toward their centroid by factor alpha."""
    centroid = np.mean(centers, axis=0)
    return centroid + alpha * (centers - centroid)


def _objective(params):
    """Objective: minimize the containing hexagon side length with penalty for overlaps."""
    centers = params.reshape(12, 2)
    s = _min_containing_hex_side(centers)
    violation = _separation_violation(centers)
    penalty = 100.0 * violation
    return s + penalty


def _affine_transform(base_centers, theta_deg, scale, tx, ty):
    """
    Apply affine transformation: rotate by theta_deg, scale by 'scale', translate by (tx, ty).
    base_centers: (12, 2) array of base lattice centers.
    Returns transformed (12, 2) centers.
    """
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    # Scale then rotate then translate
    scaled = base_centers * scale
    rotated = scaled @ R.T
    return rotated + np.array([tx, ty])


def _affine_objective(params, base_centers):
    """
    Objective for affine transformation: minimize containing hex side length.
    params = [theta_deg, scale, tx, ty]
    Non-overlap is guaranteed by construction if scale >= 1.
    """
    theta_deg, scale, tx, ty = params
    if scale < 0.999999:
        return 1e6  # penalty for shrinking (would cause overlaps)
    centers = _affine_transform(base_centers, theta_deg, scale, tx, ty)
    s = _min_containing_hex_side(centers)
    # Also add small penalty for any separation violation (shouldn't happen for scale >= 1)
    viol = _separation_violation(centers)
    return s + 100.0 * viol


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), (x, y, angle_deg)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_deg)
        outer_hex_side_length: float
    """
    best_centers = None
    best_s = np.inf

    # Base configurations: hexagonal lattice arrangements (non-overlapping by construction)
    base_configs = [
        _make_hex_lattice_12(),
        _make_compact_initial(),
    ]

    for base_centers in base_configs:
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - _SAFETY_MARGIN:
            break

        # Phase 1: Coarse grid search over theta (0-60 deg, exploiting 6-fold symmetry)
        # and scale (0.8 to 1.2)
        theta_values = np.arange(0.0, 60.0, 2.0)  # 30 values
        scale_values = np.array([1.0])  # Start with scale = 1 (unit hexagons, no scaling)

        for theta_deg in theta_values:
            for scale in scale_values:
                # For each theta and scale, find optimal translation
                # The optimal translation centers the configuration in the containing hex
                # Compute the bounding box of the rotated/scaled configuration
                rotated = _affine_transform(base_centers, theta_deg, scale, 0.0, 0.0)
                # The optimal translation centers the shape: subtract the mean
                centroid = np.mean(rotated, axis=0)
                tx, ty = -centroid[0], -centroid[1]

                centers = _affine_transform(base_centers, theta_deg, scale, tx, ty)
                s = _min_containing_hex_side(centers)

                if _check_separation(centers):
                    if s < best_s:
                        best_s = s
                        best_centers = centers.copy()

        # Phase 2: Nelder-Mead refinement on the 4D affine parameters
        if best_centers is not None:
            elapsed = time.time() - _START_TIME
            if elapsed < _DEADLINE - _SAFETY_MARGIN:
                # Start from the best theta found, scale=1, centered
                x0 = np.array([0.0, 1.0, 0.0, 0.0])
                try:
                    result = minimize(
                        _affine_objective,
                        x0,
                        args=(base_centers,),
                        method='Nelder-Mead',
                        options={
                            'maxiter': 20000,
                            'maxfev': 200000,
                            'xatol': 1e-10,
                            'fatol': 1e-12,
                            'adaptive': True,
                        }
                    )
                except Exception:
                    result = None

                if result is not None:
                    theta_deg, scale, tx, ty = result.x
                    if scale >= 0.999999:
                        candidate = _affine_transform(base_centers, theta_deg, scale, tx, ty)
                        if _check_separation(candidate):
                            s_cand = _min_containing_hex_side(candidate)
                            if s_cand < best_s:
                                best_s = s_cand
                                best_centers = candidate.copy()

        # Phase 3: Multi-start Nelder-Mead from different initial thetas
        for theta_init in [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]:
            elapsed = time.time() - _START_TIME
            if elapsed > _DEADLINE - _SAFETY_MARGIN:
                break

            x0 = np.array([theta_init, 1.0, 0.0, 0.0])
            try:
                result = minimize(
                    _affine_objective,
                    x0,
                    args=(base_centers,),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 15000,
                        'maxfev': 150000,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    }
                )
            except Exception:
                result = None

            if result is not None:
                theta_deg, scale, tx, ty = result.x
                if scale >= 0.999999:
                    candidate = _affine_transform(base_centers, theta_deg, scale, tx, ty)
                    if _check_separation(candidate):
                        s_cand = _min_containing_hex_side(candidate)
                        if s_cand < best_s:
                            best_s = s_cand
                            best_centers = candidate.copy()

        # Phase 4: Try scale slightly above 1 with different rotations
        for scale_try in [1.0, 1.02, 1.05]:
            elapsed = time.time() - _START_TIME
            if elapsed > _DEADLINE - _SAFETY_MARGIN:
                break

            for theta_init in [0.0, 15.0, 30.0, 45.0]:
                x0 = np.array([theta_init, scale_try, 0.0, 0.0])
                try:
                    result = minimize(
                        _affine_objective,
                        x0,
                        args=(base_centers,),
                        method='Nelder-Mead',
                        options={
                            'maxiter': 10000,
                            'maxfev': 100000,
                            'xatol': 1e-10,
                            'fatol': 1e-12,
                            'adaptive': True,
                        }
                    )
                except Exception:
                    result = None

                if result is not None:
                    theta_deg, scale, tx, ty = result.x
                    if scale >= 0.999999:
                        candidate = _affine_transform(base_centers, theta_deg, scale, tx, ty)
                        if _check_separation(candidate):
                            s_cand = _min_containing_hex_side(candidate)
                            if s_cand < best_s:
                                best_s = s_cand
                                best_centers = candidate.copy()

    # Fallback
    if best_centers is None:
        best_centers = _make_spread_initial().copy()
        best_s = _min_containing_hex_side(best_centers)

    # Final Nelder-Mead polish on full 24D positions (if time permits)
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - _SAFETY_MARGIN:
        x0_final = best_centers.flatten()
        try:
            result_final = minimize(
                _objective,
                x0_final,
                method='Nelder-Mead',
                options={
                    'maxiter': 12000,
                    'maxfev': 120000,
                    'xatol': 1e-11,
                    'fatol': 1e-13,
                    'adaptive': True,
                }
            )
        except Exception:
            result_final = None

        if result_final is not None:
            candidate = result_final.x.reshape(12, 2)
            if _check_separation(candidate):
                s_cand = _min_containing_hex_side(candidate)
                if s_cand < best_s:
                    best_s = s_cand
                    best_centers = candidate.copy()

    # Build output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_centers[i, 0]
        inner_hex_data[i, 1] = best_centers[i, 1]
        inner_hex_data[i, 2] = 0.0

    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = best_s

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
