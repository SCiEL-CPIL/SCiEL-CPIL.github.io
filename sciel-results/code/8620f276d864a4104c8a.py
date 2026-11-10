# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack((cx + d * np.cos(angles), cy + d * np.sin(angles)))


def _min_containing_hex_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum side length s of a regular hexagon (centered at outer_center,
    with the given angle) that contains all inner unit hexagons.
    
    Returns (s_min, valid) where valid is True if the configuration is feasible
    (no overlaps and all inner vertices can fit).
    """
    ov_cx, ov_cy = outer_center[0], outer_center[1]
    ov_a = np.deg2rad(outer_angle_deg)
    
    # The 6 outward normals to the sides of the outer hexagon
    # For a hexagon with vertices at angles (a + 60*i), the side normals point at (a + 30 + 60*i)
    normal_angles = ov_a + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    # For each inner hexagon, find the maximum projection onto each normal
    max_projections = np.zeros(6)
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        # Translate to outer center frame
        hv_rel = hv - np.array([ov_cx, ov_cy])
        projections = hv_rel @ normals.T  # shape (6, 6)
        max_projections = np.maximum(max_projections, np.max(projections, axis=0))
    
    # The distance from center to each side is s * sqrt(3)/2
    # So we need s * sqrt(3)/2 >= max_projection for all 6 directions
    s_min = np.max(max_projections) / (np.sqrt(3.0) / 2.0)
    
    return s_min


def _check_separation(inner):
    """
    Check if all pairs of inner hexagons are separated (non-overlapping) using SAT.
    Returns True if all pairs are separated.
    """
    n = len(inner)
    inner_verts = [_hex_vertices(row[0], row[1], 1.0, row[2]) for row in inner]
    
    for i in range(n):
        for j in range(i + 1, n):
            v1 = inner_verts[i]
            v2 = inner_verts[j]
            
            # Get all edge normals from both hexagons
            edges = []
            for k in range(6):
                edges.append(v1[(k + 1) % 6] - v1[k])
                edges.append(v2[(k + 1) % 6] - v2[k])
            
            separated = False
            for edge in edges:
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    continue
                normal /= norm
                
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                gap1 = min2 - max1
                gap2 = min1 - max2
                if gap1 > 1e-6 or gap2 > 1e-6:
                    separated = True
                    break
            
            if not separated:
                return False
    
    return True


def _objective(params, outer_center, outer_angle_deg):
    """
    Objective function: minimize the containing hexagon side length.
    params: 24 values = 12 * (x, y) for the 12 inner hexagons (angles fixed at 0)
    """
    inner = np.zeros((12, 3))
    inner[:, 0] = params[0::2]
    inner[:, 1] = params[1::2]
    inner[:, 2] = 0.0  # Fixed angles
    
    # Check separation
    if not _check_separation(inner):
        # Penalize overlaps heavily
        return 1000.0
    
    s = _min_containing_hex_side(inner, outer_center, outer_angle_deg)
    return s


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin before 630s limit
    
    outer_center = np.array([0.0, 0.0])
    outer_angle = 0.0
    
    # Initial configuration: symmetric hexagonal arrangement
    d = np.sqrt(3.0)  # Center-to-center distance for adjacent unit hexagons
    
    # 12 hexagons: center + ring of 6 + ring of 5 more (partial second ring)
    # Let's use a known good starting configuration
    initial_params = np.array([
        0.0, 0.0,          # center
        d, 0.0,            # right
        -d, 0.0,           # left
        d/2, d*1.5,        # top-right
        -d/2, d*1.5,       # top-left
        d/2, -d*1.5,       # bottom-right
        -d/2, -d*1.5,      # bottom-left
        2*d, 0.0,          # far right
        -2*d, 0.0,         # far left
        3*d/2, d*1.5,      # far top-right
        -3*d/2, d*1.5,     # far top-left
        0.0, -2*d*1.5,     # far bottom
    ])
    
    # Verify initial configuration is valid
    inner_init = np.zeros((12, 3))
    inner_init[:, 0] = initial_params[0::2]
    inner_init[:, 1] = initial_params[1::2]
    
    if not _check_separation(inner_init):
        # Fallback to a simpler grid
        initial_params = np.array([
            0.0, 0.0,
            2.5, 0.0,
            -2.5, 0.0,
            1.25, 2.165,
            -1.25, 2.165,
            1.25, -2.165,
            -1.25, -2.165,
            3.75, 2.165,
            -3.75, 2.165,
            3.75, -2.165,
            -3.75, -2.165,
            0.0, -4.330,
        ])
    
    best_params = initial_params.copy()
    best_s = _min_containing_hex_side(inner_init, outer_center, outer_angle)
    
    # Nelder-Mead optimization
    try:
        result = minimize(
            _objective,
            initial_params,
            args=(outer_center, outer_angle),
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'xatol': 1e-10,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        
        if result.fun < best_s:
            best_params = result.x
            best_s = result.fun
        
    except Exception as e:
        pass
    
    # Check time and potentially do a second optimization pass with different initial conditions
    elapsed = time.time() - start_time
    if elapsed < deadline - 100.0:
        # Try a second optimization with perturbed initial conditions
        try:
            np.random.seed(42)
            for trial in range(3):
                if time.time() - start_time > deadline - 60.0:
                    break
                
                perturbed = initial_params + np.random.normal(0, 0.5, size=initial_params.shape)
                result2 = minimize(
                    _objective,
                    perturbed,
                    args=(outer_center, outer_angle),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 3000,
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                
                if result2.fun < best_s:
                    best_params = result2.x
                    best_s = result2.fun
                    
        except Exception as e:
            pass
    
    # Construct final inner hexagons
    inner = np.zeros((12, 3))
    inner[:, 0] = best_params[0::2]
    inner[:, 1] = best_params[1::2]
    inner[:, 2] = 0.0
    
    # Compute final s with a small margin
    s_final = _min_containing_hex_side(inner, outer_center, outer_angle) + 1e-4
    
    # Verify validity one more time
    if not _check_separation(inner):
        # Fallback to initial configuration
        inner = np.zeros((12, 3))
        inner[:, 0] = initial_params[0::2]
        inner[:, 1] = initial_params[1::2]
        inner[:, 2] = 0.0
        s_final = _min_containing_hex_side(inner, outer_center, outer_angle) + 1e-4
    
    outer = np.array([0.0, 0.0, 0.0])
    
    return inner, outer, s_final


# EVOLVE-BLOCK-END
