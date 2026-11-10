# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

_START_TIME = time.time()
_DEADLINE = 630.0
_SAFETY_MARGIN = 40.0


def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _min_containing_hex_side(centers):
    """
    Minimum side length s of a regular hexagon centered at (0,0), angle 0,
    that contains all unit hexagons with given centers.
    Uses the fact that for a regular hexagon of side s, the inradius is s*sqrt(3)/2,
    and a point (x,y) is inside iff for all 6 edge normals n_k, dot(n_k, p) <= s*sqrt(3)/2.
    So s >= max_k max_i dot(n_k, p_i) / (sqrt(3)/2) = (2/sqrt(3)) * max_k max_i dot(n_k, p_i).
    """
    n = len(centers)
    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)

    verts = np.zeros((n, 6, 2))
    verts[:, :, 0] = centers[:, 0:1] + base_cos[None, :]
    verts[:, :, 1] = centers[:, 1:2] + base_sin[None, :]

    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])

    verts_flat = verts.reshape(-1, 2)
    projections = normals @ verts_flat.T
    max_proj = np.max(projections)

    s = (2.0 / np.sqrt(3.0)) * max_proj
    return s


def _separation_violation_smooth(centers):
    """
    Smooth separation violation using softplus approximation of max overlap.
    For each pair, compute the maximum overlap along any of the 6 SAT axes.
    Use a smooth approximation: softplus(max_overlap) instead of max(max_overlap, 0).
    """
    n = len(centers)
    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)

    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])

    all_verts = np.zeros((n, 6, 2))
    all_verts[:, :, 0] = centers[:, 0:1] + base_cos[None, :]
    all_verts[:, :, 1] = centers[:, 1:2] + base_sin[None, :]

    total_violation = 0.0
    beta = 50.0  # sharpness of softplus

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

            # Smooth: softplus(max_overlap) ≈ max(max_overlap, 0) for large beta
            if max_overlap > 0.01:
                total_violation += max_overlap ** 2
            elif max_overlap > -0.01:
                # In the transition zone, use softplus for smoothness
                total_violation += np.log(1.0 + np.exp(beta * max_overlap)) / beta
            # else: negligible

    return total_violation


def _separation_violation(centers):
    """Hard separation violation (for validation)."""
    n = len(centers)
    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)

    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])

    all_verts = np.zeros((n, 6, 2))
    all_verts[:, :, 0] = centers[:, 0:1] + base_cos[None, :]
    all_verts[:, :, 1] = centers[:, 1:2] + base_sin[None, :]

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
    all_verts[:, :, 0] = centers[:, 0:1] + base_cos[None, :]
    all_verts[:, :, 1] = centers[:, 1:2] + base_sin[None, :]

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
    """Compact hexagonal lattice arrangement of 12 hexagons."""
    sqrt3 = np.sqrt(3)
    return np.array([
        [0.0, 0.0],
        [sqrt3, 0.0],
        [sqrt3/2, 1.5],
        [-sqrt3/2, 1.5],
        [-sqrt3, 0.0],
        [-sqrt3/2, -1.5],
        [sqrt3/2, -1.5],
        [3*sqrt3/2, 1.5],
        [3*sqrt3/2, -1.5],
        [0.0, 3.0],
        [-3*sqrt3/2, 1.5],
        [-3*sqrt3/2, -1.5],
    ])


def _make_spread_initial():
    """A more spread-out initial arrangement."""
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


def _objective_de(params):
    """
    Objective for differential evolution: minimize containing hex side + penalty.
    Uses smooth separation violation.
    """
    centers = params.reshape(12, 2)
    s = _min_containing_hex_side(centers)
    violation = _separation_violation_smooth(centers)
    penalty = 50.0 * violation
    return s + penalty


def _objective_nm(params):
    """
    Objective for Nelder-Mead refinement: minimize containing hex side + penalty.
    Uses hard separation violation (Nelder-Mead can handle non-smoothness better with adaptive).
    """
    centers = params.reshape(12, 2)
    s = _min_containing_hex_side(centers)
    violation = _separation_violation(centers)
    penalty = 100.0 * violation
    return s + penalty


def _refine_with_nm(centers, maxiter=8000, maxfev=80000):
    """Refine a valid configuration with Nelder-Mead."""
    x0 = centers.flatten()
    try:
        result = minimize(
            _objective_nm,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': maxiter,
                'maxfev': maxfev,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True,
            }
        )
        if result is not None:
            candidate = result.x.reshape(12, 2)
            if _check_separation(candidate):
                return candidate
    except Exception:
        pass
    return None


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

    # Initial configurations
    init_configs = [
        _make_hex_lattice_12(),
        _make_spread_initial(),
    ]

    # Add scaled versions
    for scale in [0.95, 1.05]:
        init_configs.append(_make_hex_lattice_12() * scale)

    rng = np.random.default_rng(123)

    # Add random perturbations of the lattice
    for r in range(3):
        noise = 0.3 * (1.0 + r)
        init_configs.append(_make_hex_lattice_12() + rng.normal(0, noise, size=(12, 2)))

    # Evaluate initial configs
    for init in init_configs:
        if _check_separation(init):
            s_init = _min_containing_hex_side(init)
            if s_init < best_s:
                best_s = s_init
                best_centers = init.copy()

    # Phase 1: Differential Evolution (global search)
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - _SAFETY_MARGIN:
        # Define bounds: centers should be within a reasonable region
        bounds = [(-6.0, 6.0)] * 24

        try:
            de_result = differential_evolution(
                _objective_de,
                bounds,
                seed=42,
                maxiter=300,
                popsize=20,
                tol=1e-8,
                mutation=(0.3, 1.0),
                recombination=0.7,
                polish=True,
                init='soi',  # scale out of initial
                callback=lambda xk, convergence: (
                    time.time() - _START_TIME < _DEADLINE - _SAFETY_MARGIN
                ),
            )

            if de_result is not None:
                candidate = de_result.x.reshape(12, 2)
                if _check_separation(candidate):
                    s_cand = _min_containing_hex_side(candidate)
                    if s_cand < best_s:
                        best_s = s_cand
                        best_centers = candidate.copy()

                # Also try without polish (raw DE result)
                # The polish step may have introduced issues
        except Exception:
            pass

    # Phase 2: Nelder-Mead refinement from best solution
    elapsed = time.time() - _START_TIME
    if best_centers is not None and elapsed < _DEADLINE - _SAFETY_MARGIN:
        refined = _refine_with_nm(best_centers, maxiter=12000, maxfev=120000)
        if refined is not None:
            s_cand = _min_containing_hex_side(refined)
            if s_cand < best_s:
                best_s = s_cand
                best_centers = refined.copy()

    # Phase 3: Additional DE runs with different seeds if time permits
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - _SAFETY_MARGIN:
        bounds = [(-6.0, 6.0)] * 24
        for seed in [7, 99, 256]:
            elapsed = time.time() - _START_TIME
            if elapsed > _DEADLINE - _SAFETY_MARGIN:
                break
            try:
                de_result2 = differential_evolution(
                    _objective_de,
                    bounds,
                    seed=seed,
                    maxiter=200,
                    popsize=15,
                    tol=1e-8,
                    mutation=(0.5, 1.0),
                    recombination=0.8,
                    polish=False,
                    init='latinhypercube',
                )
                if de_result2 is not None:
                    candidate = de_result2.x.reshape(12, 2)
                    if _check_separation(candidate):
                        s_cand = _min_containing_hex_side(candidate)
                        if s_cand < best_s:
                            best_s = s_cand
                            best_centers = candidate.copy()
            except Exception:
                pass

    # Phase 4: Final Nelder-Mead polish
    elapsed = time.time() - _START_TIME
    if best_centers is not None and elapsed < _DEADLINE - _SAFETY_MARGIN:
        refined2 = _refine_with_nm(best_centers, maxiter=15000, maxfev=150000)
        if refined2 is not None:
            s_cand = _min_containing_hex_side(refined2)
            if s_cand < best_s:
                best_s = s_cand
                best_centers = refined2.copy()

    # Fallback if no valid solution found
    if best_centers is None:
        best_centers = _make_hex_lattice_12()
        best_s = _min_containing_hex_side(best_centers)

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
