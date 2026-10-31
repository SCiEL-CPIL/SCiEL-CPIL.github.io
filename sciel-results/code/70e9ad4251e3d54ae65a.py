import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Get vertices of a regular hexagon with center (cx,cy), rotation angle_deg, side length."""
    a = np.deg2rad(angle_deg)
    angles = a + np.pi / 6 + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _max_projection(points, directions):
    """Max projection of points onto given directions (columns of directions)."""
    # points: (n, 2), directions: (k, 2)
    # For each direction, compute max dot product
    dots = points @ directions.T  # (n, k)
    return dots.max(axis=0)  # (k,)

def _min_projection(points, directions):
    dots = points @ directions.T
    return dots.min(axis=0)

def _compute_max_extent(inner_positions, inner_angles, outer_center, outer_side, outer_angle):
    """
    Compute the required outer side length for all inner hexagons to fit.
    Uses projection onto 6 directions of the outer hexagon.
    """
    # Outer hexagon directions (6 normals to edges)
    oa = np.deg2rad(outer_angle)
    outer_dirs = np.column_stack([
        np.cos(oa + np.pi / 6 + i * np.pi / 3),
        np.sin(oa + np.pi / 6 + i * np.pi / 3)
    ])  # (6, 2)

    # For each inner hexagon, get its vertices
    all_verts = []
    for i in range(N):
        cx, cy = inner_positions[i]
        ang = inner_angles[i]
        verts = _hex_vertices(cx, cy, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)  # (6N, 2)

    # Translate relative to outer center
    all_verts = all_verts - np.array(outer_center)

    # The outer hexagon with side s and center at origin, rotation oa:
    # A point p is inside if for all 6 directions d_j: |p . d_j| <= s * sqrt(3) / 2
    # Actually, for a regular hexagon with side s, the apothem (distance from center to edge) is s * sqrt(3)/2
    # The constraint is: for each of the 6 edge normals, the projection must be within [-apothem, apothem]
    
    apothem_factor = np.sqrt(3) / 2  # apothem = side * apothem_factor

    # For each direction, compute max and min projection
    max_proj = _max_projection(all_verts, outer_dirs)  # (6,)
    min_proj = _min_projection(all_verts, outer_dirs)  # (6,)

    # Required side: max over all directions of max(|max_proj|, |min_proj|) / apothem_factor
    required = np.maximum(np.abs(max_proj), np.abs(min_proj)) / apothem_factor
    return required.max()

def _objective(params):
    """
    Objective: minimize the outer side length.
    params: [x0, y0, x1, y1, ..., x10, y10, a0, a1, ..., a10, ox, oy, oa, side]
    We optimize positions, angles, and outer hexagon parameters.
    Actually, let's fix outer center at origin and outer angle at 0 for symmetry,
    and just optimize inner positions and angles + outer side.
    
    Better: optimize inner positions and angles, and compute required side.
    params: 2*N positions + N angles = 3N parameters
    """
    pos = params[:2 * N].reshape(N, 2)
    angles = params[2 * N:3 * N]
    
    # Compute required outer side with outer center at (0,0), angle 0
    required_side = _compute_max_extent(pos, angles, (0.0, 0.0), 1.0, 0.0)
    return required_side

def _logsumexp_objective(params, beta=50.0):
    """
    Smooth surrogate using log-sum-exp of the 6 direction constraints.
    """
    pos = params[:2 * N].reshape(N, 2)
    angles = params[2 * N:3 * N]
    
    # Get all vertices
    all_verts = []
    for i in range(N):
        cx, cy = pos[i]
        ang = angles[i]
        verts = _hex_vertices(cx, cy, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)  # (6N, 2)

    # Outer hexagon directions (angle 0)
    oa = 0.0
    outer_dirs = np.column_stack([
        np.cos(np.pi / 6 + i * np.pi / 3),
        np.sin(np.pi / 6 + i * np.pi / 3)
    ])  # (6, 2)

    apothem_factor = np.sqrt(3) / 2

    dots = all_verts @ outer_dirs.T  # (6N, 6)
    
    # For each direction, we need max(|proj|) / apothem_factor
    # Use log-sum-exp for smooth max
    # For direction j: log(sum(exp(beta * |dots[:,j]|))) / beta approximates max
    abs_dots = np.abs(dots)  # (6N, 6)
    
    # Smooth max via log-sum-exp
    # To avoid overflow, subtract max first
    smooth_max = np.zeros(6)
    for j in range(6):
        col = abs_dots[:, j]
        m = col.max()
        smooth_max[j] = m + np.log(np.exp(beta * (col - m)).sum()) / beta
    
    required = smooth_max / apothem_factor
    return required.max()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial arrangement: hexagonal lattice
    # For 11 hexagons, use a hexagonal packing
    # Hexagonal lattice spacing: 2 * sqrt(3) for side-by-side, but for hexagons it's different
    # Two unit hexagons side by side: distance between centers = 2 * apothem = sqrt(3)
    # Actually, for regular hexagons of side 1, the distance between centers when touching is 2 * sqrt(3)/2 * 2 = ... 
    # Let me think: a regular hexagon has "width" (flat-to-flat) = sqrt(3), and "height" (vertex-to-vertex) = 2.
    # When two hexagons touch edge-to-edge, center distance = sqrt(3).
    
    # Hexagonal lattice: rows offset
    # Row 0: 4 hexagons
    # Row 1: 3 hexagons (offset)
    # Row 2: 4 hexagons
    # Total: 11
    
    # Lattice vectors for hexagonal packing of hexagons:
    # a1 = (sqrt(3), 0)  -- horizontal spacing
    # a2 = (sqrt(3)/2, 3/2)  -- diagonal spacing
    
    sqrt3 = np.sqrt(3)
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3 / 2, 1.5])
    
    positions = []
    # Row 0: 4 hexagons
    for k in range(4):
        positions.append(k * a1)
    # Row 1: 3 hexagons, offset by a2
    for k in range(3):
        positions.append(a2 + k * a1)
    # Row 2: 4 hexagons, offset by 2*a2
    for k in range(4):
        positions.append(2 * a2 + k * a1)
    
    positions = np.array(positions)  # (11, 2)
    angles = np.zeros(N)
    
    # Center the arrangement
    centroid = positions.mean(axis=0)
    positions -= centroid
    
    # Initial parameters
    x0 = np.concatenate([positions.flatten(), angles])
    
    best_params = x0.copy()
    best_side = _compute_max_extent(positions, angles, (0.0, 0.0), 1.0, 0.0)
    
    # Phase 1: L-BFGS-B optimization with smooth objective
    if time.time() < deadline - 60:
        try:
            res = minimize(
                _logsumexp_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.success or res.fun < best_side:
                # Verify with exact computation
                pos = res.x[:2 * N].reshape(N, 2)
                ang = res.x[2 * N:3 * N]
                side = _compute_max_extent(pos, ang, (0.0, 0.0), 1.0, 0.0)
                if side < best_side:
                    best_side = side
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Polish with exact objective using L-BFGS-B
    if time.time() < deadline - 30:
        try:
            res2 = minimize(
                _objective,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res2.fun < best_side:
                pos = res2.x[:2 * N].reshape(N, 2)
                ang = res2.x[2 * N:3 * N]
                side = _compute_max_extent(pos, ang, (0.0, 0.0), 1.0, 0.0)
                if side < best_side:
                    best_side = side
                    best_params = res2.x.copy()
        except Exception:
            pass
    
    # Phase 3: Try different initial angles
    if time.time() < deadline - 20:
        for trial_angle in [0, 30, 60]:
            trial_angles = np.full(N, trial_angle * np.pi / 180)
            trial_pos = positions.copy()
            trial_x0 = np.concatenate([trial_pos.flatten(), trial_angles])
            try:
                res3 = minimize(
                    _logsumexp_objective,
                    trial_x0,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res3.fun < best_side:
                    pos = res3.x[:2 * N].reshape(N, 2)
                    ang = res3.x[2 * N:3 * N]
                    side = _compute_max_extent(pos, ang, (0.0, 0.0), 1.0, 0.0)
                    if side < best_side:
                        best_side = side
                        best_params = res3.x.copy()
            except Exception:
                pass
            if time.time() > deadline - 20:
                break
    
    # Phase 4: Final polish
    if time.time() < deadline - 10:
        try:
            res4 = minimize(
                _objective,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res4.fun < best_side:
                pos = res4.x[:2 * N].reshape(N, 2)
                ang = res4.x[2 * N:3 * N]
                side = _compute_max_extent(pos, ang, (0.0, 0.0), 1.0, 0.0)
                if side < best_side:
                    best_side = side
                    best_params = res4.x.copy()
        except Exception:
            pass
    
    # Extract final result
    pos = best_params[:2 * N].reshape(N, 2)
    ang = best_params[2 * N:3 * N]
    
    # Convert angles to degrees
    inner = []
    for i in range(N):
        inner.append([float(pos[i, 0]), float(pos[i, 1]), float(np.rad2deg(ang[i]))])
    
    # Outer hexagon: center at (0,0), angle 0, side = best_side
    # But we can also optimize the outer hexagon's rotation and center
    # For now, keep center at (0,0) and angle at 0
    
    # Try to optimize outer center and angle for a small improvement
    if time.time() < deadline - 5:
        def outer_obj(params):
            ox, oy, oa_deg = params
            oa_rad = np.deg2rad(oa_deg)
            return _compute_max_extent(pos, ang, (ox, oy), 1.0, oa_rad)
        
        try:
            res5 = minimize(
                outer_obj,
                [0.0, 0.0, 0.0],
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if res5.fun < best_side:
                ox, oy, oa_deg = res5.x
                side = _compute_max_extent(pos, ang, (ox, oy), 1.0, oa_deg)
                if side < best_side:
                    best_side = side
                    # Adjust inner positions relative to new center
                    pos = pos - np.array([ox, oy])
                    # Re-extract
                    inner = []
                    for i in range(N):
                        inner.append([float(pos[i, 0]), float(pos[i, 1]), float(np.rad2deg(ang[i]))])
                    outer_center = (float(ox), float(oy))
                    outer_angle = float(oa_deg)
                else:
                    outer_center = (0.0, 0.0)
                    outer_angle = 0.0
            else:
                outer_center = (0.0, 0.0)
                outer_angle = 0.0
        except Exception:
            outer_center = (0.0, 0.0)
            outer_angle = 0.0
    else:
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
    
    return inner, outer_center, float(best_side), float(outer_angle)

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
