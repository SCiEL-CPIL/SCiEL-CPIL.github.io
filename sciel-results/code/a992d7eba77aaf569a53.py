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


def _lattice_positions():
    """
    12 unit hexagons on a hexagonal lattice (non-overlapping by construction).
    Lattice vectors: a1 = (3, 0), a2 = (1.5, 3*sqrt(3)/2)
    Centers: (i*a1 + j*a2) for (i,j) in a 12-point cluster.
    """
    a1 = np.array([3.0, 0.0])
    a2 = np.array([1.5, 3.0 * np.sqrt(3) / 2.0])
    
    # 12 points forming a compact cluster
    # Center (0,0) + 6 neighbors + 5 more
    indices = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-2, 0), (-1, -1),
    ]
    
    centers = np.array([i * a1 + j * a2 for i, j in indices])
    return centers


def _min_enclosing_hex_side(transformed_centers):
    """
    Given centers of unit hexagons (angle 0), find the minimum side length s
    of a regular hexagon (centered at origin, angle 0) that contains all vertices.
    """
    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)
    
    # Vertices of all unit hexagons
    verts_x = transformed_centers[:, 0:1] + base_cos[np.newaxis, :]
    verts_y = transformed_centers[:, 1:2] + base_sin[np.newaxis, :]
    
    verts_flat = np.column_stack([verts_x.ravel(), verts_y.ravel()])  # (72, 2)
    
    # Outward normals at 30, 90, 150, 210, 270, 330 degrees
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    projections = normals @ verts_flat.T  # (6, 72)
    max_proj_per_normal = np.max(projections, axis=1)  # (6,)
    max_proj = np.max(max_proj_per_normal)
    
    # For a regular hexagon with side s, the apothem (distance from center to edge) is s*sqrt(3)/2
    # The projection of a vertex onto the outward normal at angle 30+60k equals s*sqrt(3)/2 * cos(0) = s*sqrt(3)/2
    # Wait, let me reconsider. The normal to edge k is at angle 30+60k degrees.
    # The distance from center to that edge is the apothem = s * sqrt(3) / 2.
    # So max_proj should equal s * sqrt(3) / 2 for the tightest fit.
    s = 2.0 * max_proj / np.sqrt(3.0)
    return s


def _apply_transform(base_centers, theta, sx, sy, tx, ty):
    """Apply rotation + scaling + translation to base centers."""
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    S = np.array([[sx, 0], [0, sy]])
    # Transform: T = R @ S @ x + t
    transformed = (R @ S @ base_centers.T).T + np.array([tx, ty])
    return transformed


def _objective(params):
    """
    Objective: minimize the enclosing hexagon side length.
    params = [theta, sx, sy, tx, ty]
    """
    theta, sx, sy, tx, ty = params
    
    # Constraints: sx, sy must be positive (scale factors)
    if sx <= 0.01 or sy <= 0.01:
        return 1e6
    
    transformed = _apply_transform(_LATTICE_BASE, theta, sx, sy, tx, ty)
    s = _min_enclosing_hex_side(transformed)
    return s


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


def _check_containment(centers, s, tol=1e-6):
    """Check all inner vertices are inside the outer hexagon of side s centered at origin."""
    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)
    
    # Outer hexagon vertices
    outer_verts = np.column_stack([s * np.cos(base_angles), s * np.sin(base_angles)])
    
    # For each inner vertex, check it's inside the outer hexagon
    for i in range(len(centers)):
        vx = centers[i, 0] + base_cos
        vy = centers[i, 1] + base_sin
        for k in range(6):
            v1 = outer_verts[k]
            v2 = outer_verts[(k + 1) % 6]
            edge = v2 - v1
            # cross(edge, point - v1) should be >= -tol for CCW polygon
            cross_val = edge[0] * (vy[k] - v1[1]) - edge[1] * (vx[k] - v1[0])
            if cross_val < -tol:
                return False
    return True


# Precompute lattice base positions
_LATTICE_BASE = _lattice_positions()


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    """
    best_s = np.inf
    best_params = None
    best_centers = None
    
    # The lattice configuration has fixed relative positions.
    # We optimize: rotation angle theta, and we need to find the best enclosing hex.
    # Key insight: for a fixed set of hexagons, we can rotate the whole configuration
    # and find the minimal enclosing regular hexagon.
    #
    # Actually, let's think more carefully. The 12 hexagons are on a fixed lattice.
    # We can rotate the entire configuration by angle theta.
    # Then we need to find the minimal enclosing regular hexagon.
    #
    # The minimal enclosing regular hexagon for a set of points (with unit hexagon padding)
    # depends on the orientation. We optimize theta.
    #
    # But we can also translate. The optimal translation centers the configuration
    # within the enclosing hexagon.
    
    # For a given rotation theta, the optimal translation is determined by
    # centering the bounding box in the hexagon.
    
    def _min_s_for_rotation(theta):
        """Find minimum enclosing hex side for a given rotation angle."""
        c = np.cos(theta)
        s_ = np.sin(theta)
        R = np.array([[c, -s_], [s_, c]])
        
        rotated = (R @ _LATTICE_BASE.T).T
        
        # Find the optimal translation to minimize enclosing hex side
        # The enclosing hexagon is centered at origin with angle 0.
        # We want to translate the rotated points so that the enclosing hex is minimized.
        
        # For a regular hexagon centered at origin, the constraint is:
        # For each of 6 directions (normals at 30+60k), the max projection <= s*sqrt(3)/2
        # and min projection >= -s*sqrt(3)/2
        
        # If we translate by t, projections become (n . (p + t)) = n.p + n.t
        # We want to minimize max_k (max_i (n_k . p_i + n_k . t)) - min_i (n_k . p_i + n_k . t)
        # Actually we want all points in [-s*sqrt(3)/2, s*sqrt(3)/2] for each normal.
        
        # Let's use a different approach: for each normal direction, compute the range
        # of projections. The translation can shift all projections.
        
        # For normal k: projections of rotated points are n_k . rotated[i]
        # After translation by t: n_k . rotated[i] + n_k . t
        # We need: max_i (n_k . rotated[i] + n_k . t) <= s*sqrt(3)/2
        #           min_i (n_k . rotated[i] + n_k . t) >= -s*sqrt(3)/2
        # So: s*sqrt(3)/2 >= max_i(n_k . rotated[i]) + n_k . t
        #     s*sqrt(3)/2 >= -min_i(n_k . rotated[i]) - n_k . t
        # Combined: s*sqrt(3)/2 >= max(max_i(n_k.rotated) + n_k.t, -min_i(n_k.rotated) - n_k.t)
        #          = max((max_i + min_i)/2 + (max_i - min_i)/2 + n_k.t, 
        #                -(max_i + min_i)/2 + (max_i - min_i)/2 - n_k.t)
        
        # This is getting complex. Let's just optimize theta and t = (tx, ty) together.
        # 3 parameters: theta, tx, ty
        
        def _obj_3d(params3):
            th, tx, ty = params3
            ct = np.cos(th)
            st = np.sin(th)
            R3 = np.array([[ct, -st], [st, ct]])
            rot = (R3 @ _LATTICE_BASE.T).T + np.array([tx, ty])
            return _min_enclosing_hex_side(rot)
        
        return _obj_3d
    
    # Optimize over theta, tx, ty
    # Use multiple initial guesses for theta
    
    best_result = None
    best_obj = np.inf
    
    initial_thetas = np.linspace(0, np.pi / 3, 13)  # 13 values over 0 to 60 degrees
    
    for th0 in initial_thetas:
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - _SAFETY_MARGIN:
            break
        
        # Initial translation: center of mass to origin
        c = np.cos(th0)
        s_ = np.sin(th0)
        R0 = np.array([[c, -s_], [s_, c]])
        rotated0 = (R0 @ _LATTICE_BASE.T).T
        com = np.mean(rotated0, axis=0)
        tx0 = -com[0]
        ty0 = -com[1]
        
        x0 = np.array([th0, tx0, ty0])
        
        try:
            result = minimize(
                _min_s_for_rotation(th0) if False else _obj_full,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 20000,
                    'maxfev': 50000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True,
                }
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result.x.copy()
        except Exception:
            pass
    
    if best_result is not None:
        theta, tx, ty = best_result
        c = np.cos(theta)
        s_ = np.sin(theta)
        R = np.array([[c, -s_], [s_, c]])
        best_centers = (R @ _LATTICE_BASE.T).T + np.array([tx, ty])
        best_s = _min_enclosing_hex_side(best_centers)
    
    # Verify separation (should be guaranteed by lattice construction, but check)
    if best_centers is not None:
        if not _check_separation(best_centers):
            # Fallback: use the original lattice without rotation
            best_centers = _LATTICE_BASE.copy()
            best_s = _min_enclosing_hex_side(best_centers)
    
    # If we still don't have a valid result, use a simple spread arrangement
    if best_centers is None:
        best_centers = _LATTICE_BASE.copy()
        best_s = _min_enclosing_hex_side(best_centers)
    
    # Build output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_centers[i, 0]
        inner_hex_data[i, 1] = best_centers[i, 1]
        inner_hex_data[i, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = best_s
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


def _obj_full(params):
    """Full objective: theta, tx, ty -> min enclosing hex side."""
    theta, tx, ty = params
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    transformed = (R @ _LATTICE_BASE.T).T + np.array([tx, ty])
    return _min_enclosing_hex_side(transformed)


# EVOLVE-BLOCK-END
