# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _cluster_vertices(params):
    """params: 33 floats = 11*(x, y, angle_deg). Returns (11,6,2) vertices."""
    verts = np.empty((11, 6, 2))
    for i in range(11):
        cx, cy, ad = params[3 * i], params[3 * i + 1], params[3 * i + 2]
        verts[i] = _hex_vertices(cx, cy, 1.0, ad)
    return verts


def _face_normals(a_deg):
    """6 face normals (outward) of a unit regular hexagon with given angle."""
    a = np.deg2rad(a_deg)
    mid_angles = a + np.deg2rad(30 + 60 * np.arange(6))
    return np.column_stack([np.cos(mid_angles), np.sin(mid_angles)])


def _pair_gap_fast(verts_a, verts_b, normals_a, normals_b):
    """Check separation using precomputed normals. Returns max gap over all 12 normals."""
    best_gap = -np.inf
    for n in np.vstack([normals_a, normals_b]):
        proj_a = verts_a @ n
        proj_b = verts_b @ n
        gap = max(proj_b.min() - proj_a.max(), proj_a.min() - proj_b.max())
        if gap > best_gap:
            best_gap = gap
    return best_gap


def _objective(params):
    """Objective to minimize: penalty for overlaps + outer hexagon side length."""
    if not np.all(np.isfinite(params)):
        return 1e10
    
    verts = _cluster_vertices(params)
    
    # Precompute normals for each hexagon
    normals_list = []
    for i in range(11):
        cx, cy, ad = params[3 * i], params[3 * i + 1], params[3 * i + 2]
        normals_list.append(_face_normals(ad))
    
    # Check pairwise separation
    min_gap = np.inf
    for i in range(11):
        for j in range(i + 1, 11):
            g = _pair_gap_fast(verts[i], verts[j], normals_list[i], normals_list[j])
            if g < min_gap:
                min_gap = g
    
    if min_gap < 1e-6:
        return 1e6 * (1e-6 - min_gap) + 100.0
    
    # Compute minimal enclosing regular hexagon side length
    all_verts = verts.reshape(-1, 2)
    centroid = all_verts.mean(axis=0)
    shifted = all_verts - centroid
    
    # Optimize orientation a in [0, 60) with finer grid
    best_side = np.inf
    for a_deg in np.linspace(0, 60, 121):
        normals = _face_normals(a_deg)
        dots = shifted @ normals.T
        r = dots.max()
        side = 2.0 * r / np.sqrt(3.0)
        if side < best_side:
            best_side = side
    
    return best_side


def _refine_orientation(shifted):
    """Find optimal orientation using golden section search."""
    best_side = np.inf
    best_a = 0.0
    
    # Coarse search
    for a_deg in np.linspace(0, 60, 121):
        normals = _face_normals(a_deg)
        dots = shifted @ normals.T
        r = dots.max()
        side = 2.0 * r / np.sqrt(3.0)
        if side < best_side:
            best_side = side
            best_a = a_deg
    
    # Fine search around best
    for a_deg in np.linspace(best_a - 1, best_a + 1, 21):
        if a_deg < 0:
            a_deg += 60
        if a_deg >= 60:
            a_deg -= 60
        normals = _face_normals(a_deg)
        dots = shifted @ normals.T
        r = dots.max()
        side = 2.0 * r / np.sqrt(3.0)
        if side < best_side:
            best_side = side
            best_a = a_deg % 60
    
    return best_side, best_a


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner, outer, side).
    """
    # Initial guess: compact hexagonal cluster
    # Center + 6 neighbors + 4 in second ring
    init = []
    init.append([0.0, 0.0, 0.0])  # center
    for k in range(6):
        ang = np.deg2rad(60 * k)
        init.append([2.0 * np.cos(ang), 2.0 * np.sin(ang), 0.0])
    for k in [0, 2, 4, 5]:
        ang = np.deg2rad(60 * k + 30)
        init.append([4.0 * np.cos(ang), 4.0 * np.sin(ang), 0.0])
    
    init_params = np.array(init).flatten()
    
    # Bounds
    bounds = []
    for i in range(11):
        bounds.append((-6.0, 6.0))
        bounds.append((-6.0, 6.0))
        bounds.append((-90.0, 90.0))
    
    # Run differential evolution with more iterations
    result = differential_evolution(
        _objective,
        bounds,
        x0=init_params,
        maxiter=300,
        popsize=20,
        tol=1e-10,
        seed=42,
        polish=False,
        init='sobol',
        workers=-1,
    )
    
    best_params = result.x
    if not np.all(np.isfinite(best_params)):
        best_params = init_params
    
    # Local refinement with L-BFGS-B
    try:
        refined = minimize(
            _objective,
            best_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if refined.fun < result.fun:
            best_params = refined.x
    except:
        pass
    
    # Build inner hex data
    inner_hex_data = np.empty((11, 3))
    for i in range(11):
        inner_hex_data[i] = best_params[3 * i:3 * i + 3]
    
    # Compute outer hexagon
    verts = _cluster_vertices(best_params)
    all_verts = verts.reshape(-1, 2)
    centroid = all_verts.mean(axis=0)
    shifted = all_verts - centroid
    
    best_side, best_a = _refine_orientation(shifted)
    
    # Shift so outer hexagon is centered at origin
    inner_hex_data[:, 0] -= centroid[0]
    inner_hex_data[:, 1] -= centroid[1]
    
    outer_hex_data = np.array([0.0, 0.0, best_a])
    outer_hex_side_length = float(best_side)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
