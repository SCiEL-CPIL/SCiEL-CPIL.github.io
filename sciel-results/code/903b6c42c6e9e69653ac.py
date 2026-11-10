# EVOLVE-BLOCK-START
import time
import numpy as np


def _hex_vertices(cx, cy, d, a_rad):
    """Vertices of a regular hexagon: center (cx,cy), side d, angle a_rad."""
    angles = a_rad + np.arange(6) * (np.pi / 3.0)
    return np.column_stack((cx + d * np.cos(angles), cy + d * np.sin(angles)))


def _min_containing_hex_side(inner_centers, d=1.0, outer_angle=0.0):
    """
    Minimum side length of a regular hexagon (center at origin, fixed angle)
    that contains all unit hexagons at inner_centers.
    Returns (s, is_finite).
    """
    # For a regular hexagon centered at origin with side s and angle a,
    # a point (x,y) is inside iff for each of the 6 outward normals n_k,
    # n_k . (x,y) <= s * |n_k| * (apothem factor).
    # Apothem of regular hexagon side s = s * sqrt(3)/2.
    # The 6 outward normals are at angles a + 60*k + 30 deg? Let's derive.
    #
    # A regular hexagon with vertices at angles a, a+60, ..., a+300 (from center)
    # has edges perpendicular to directions a+30, a+90, ..., a+330.
    # The outward normal of the edge between vertices at angles a+60k and a+60(k+1)
    # points in direction a + 60k + 30 deg.
    # Distance from center to each edge (apothem) = s * sqrt(3)/2.
    # So constraint: for each k, n_k . p <= s * sqrt(3)/2, where n_k is unit normal.
    #
    # For a unit hexagon at center c, the farthest vertex in direction n_k is
    # c + d * (projection of vertex direction onto n_k). The max over vertices of
    # n_k . v = c + d * max_i n_k . u_i where u_i are unit vertex directions.
    # For a unit hexagon (d=1), vertices are at angles a_i = 60*i deg.
    # max_i n_k . (cos(60i), sin(60i)) = max of cos(angle_diff) over 6 directions.
    # This equals cos(min angular distance from n_k to any vertex direction).
    #
    # Precompute the 6 vertex direction vectors for a unit hexagon.
    vdirs = np.array([[np.cos(np.pi/3.0 * i), np.sin(np.pi/3.0 * i)] for i in range(6)])
    # 6 outward normals for outer hexagon
    normals = np.array([[np.cos(outer_angle + np.pi/3.0 * k + np.pi/6.0),
                         np.sin(outer_angle + np.pi/3.0 * k + np.pi/6.0)] for k in range(6)])
    # max projection of unit hex vertices onto each normal
    # proj_max[k] = max_i normals[k] . vdirs[i]
    proj_max = np.max(normals @ vdirs.T, axis=1)  # shape (6,)
    # For each inner center c_j and each normal k:
    # constraint: normals[k] . c_j + d * proj_max[k] <= s * sqrt(3)/2
    # So s >= (normals[k] . c_j + d * proj_max[k]) / (sqrt(3)/2)
    sqrt3_2 = np.sqrt(3.0) / 2.0
    # inner_centers shape (12,2)
    # projections: (12, 6) = inner_centers @ normals.T
    proj = inner_centers @ normals.T  # (12, 6)
    # required_s[j,k] = (proj[j,k] + d * proj_max[k]) / sqrt3_2
    required = (proj + d * proj_max[None, :]) / sqrt3_2
    s = np.max(required)
    if not np.isfinite(s):
        return s, False
    return s, True


def _objective(params):
    """
    Objective to minimize: outer hexagon side length s.
    params: array of length 12*2 + 1 = 25.
    First 24: x,y for 12 inner hexagons. Last: outer angle (radians).
    Returns s (or large penalty).
    """
    n = 12
    centers = params[:2 * n].reshape(n, 2)
    outer_angle = params[2 * n]
    s, ok = _min_containing_hex_side(centers, d=1.0, outer_angle=outer_angle)
    if not ok:
        return 1e10
    return s


def _check_separation(centers, d=1.0, tol=1e-6):
    """
    Check that no two unit hexagons overlap using separating axis test.
    For two regular hexagons of same size, the SAT axes are the 6 edge normals
    of each hexagon (which are the same set if same orientation).
    Two unit hexagons with same orientation are separated iff for all 6 normals n_k,
    |n_k . (c1 - c2)| >= 2 * apothem = sqrt(3) * d.
    Actually: each hexagon projects onto an interval of length 2 * apothem = d*sqrt(3).
    For separation, the intervals must not overlap: |n_k . (c1-c2)| >= d * sqrt(3).
    """
    sqrt3 = np.sqrt(3.0)
    normals = np.array([[np.cos(np.pi/3.0 * k), np.sin(np.pi/3.0 * k)] for k in range(6)])
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            diff = centers[i] - centers[j]
            projs = normals @ diff
            if np.any(np.abs(projs) < sqrt3 * d - tol):
                return False
    return True


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    """
    n = 12
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 630s kill

    # Initial guess: hexagonal lattice arrangement
    # Centers of 12 hexagons in a honeycomb-like pattern
    # Using axial coordinates for hex grid:
    # Let's place them in a roughly circular/hexagonal cluster.
    # 7 in center hex ring + 5 more? Or 12 in a 4-3-4-1 pattern?
    # Simpler: use a rectangular-ish grid and let optimizer fix it.
    # Start with a hexagonal lattice:
    # Row 0: 4 hexagons
    # Row 1: 3 hexagons (offset)
    # Row 2: 3 hexagons (offset)
    # Row 3: 2 hexagons (offset)
    # Total: 12
    
    # Hex lattice: horizontal spacing = 3, vertical spacing = 2*sqrt(3)
    # For flat-top hexagons (angle 0), vertices at 0, 60, 120, 180, 240, 300 deg.
    # Edge length 1, so width = 2, height = sqrt(3).
    # Horizontal distance between centers in same row: 3 (2*1.5? no, for flat-top: 3*side/2 * 2 = 3)
    # Actually for flat-top hexagons with side 1:
    #   width = 2 (vertex to vertex horizontally)
    #   height = sqrt(3) (flat to flat vertically)
    #   Horizontal neighbor distance: 3 * cos(30) = 3*sqrt(3)/2? 
    # Let me just use a reasonable initial spread.
    
    # Use a simple grid: 4 columns x 3 rows, with offsets
    centers = np.zeros((n, 2))
    # Place in a hexagonal pattern
    # Ring 0: center (1 hex)
    # Ring 1: 6 hexes
    # Ring 2: 5 more to make 12
    # For hex lattice with flat-top orientation:
    # Axial to pixel: x = 3/2 * q, y = sqrt(3)/2 * q + sqrt(3) * r
    # Let's use cube coordinates or just hardcode a nice starting position.
    
    # Starting positions (roughly hexagonal cluster):
    init = [
        [0, 0],           # 0: center
        [3, 0],           # 1: right
        [-3, 0],          # 2: left
        [1.5, 2.598],     # 3: top-right (sqrt(3)*1.5 = 2.598)
        [-1.5, 2.598],    # 4: top-left
        [1.5, -2.598],    # 5: bottom-right
        [-1.5, -2.598],   # 6: bottom-left
        [4.5, 2.598],     # 7: far top-right
        [-4.5, 2.598],    # 8: far top-left
        [4.5, -2.598],    # 9: far bottom-right
        [-4.5, -2.598],   # 10: far bottom-left
        [0, 5.196],       # 11: top center
    ]
    centers = np.array(init, dtype=float)
    
    # Build initial params: centers + outer angle
    params = np.zeros(2 * n + 1)
    params[:2 * n] = centers.flatten()
    params[2 * n] = 0.0  # outer angle start
    
    # Evaluate initial
    best_params = params.copy()
    best_s = _objective(best_params)
    
    # Nelder-Mead optimization
    from scipy.optimize import minimize
    
    # Use Nelder-Mead with adaptive step
    result = None
    try:
        # Scale the angle variable appropriately
        # Initial simplex: perturb each variable by a small amount
        x0 = best_params.copy()
        # Perturbations
        step = 0.1
        simplex = [x0.copy()]
        for i in range(len(x0)):
            p = x0.copy()
            p[i] += step
            simplex.append(p)
        
        res = minimize(
            _objective,
            x0,
            method='Nelder-Mead',
            options={
                'xatol': 1e-8,
                'fatol': 1e-10,
                'maxiter': 5000,
                'maxfev': 20000,
                'adaptive': True,
            },
        )
        if res.success or res.fun < best_s:
            best_params = res.x
            best_s = res.fun
    except Exception:
        pass
    
    # Check time and possibly do a second round with tighter tolerance
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:
        try:
            x0 = best_params.copy()
            step = 0.05
            simplex2 = [x0.copy()]
            for i in range(len(x0)):
                p = x0.copy()
                p[i] += step
                simplex2.append(p)
            
            res2 = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'maxiter': 10000,
                    'maxfev': 50000,
                    'adaptive': True,
                },
            )
            if res2.fun < best_s:
                best_params = res2.x
                best_s = res2.fun
        except Exception:
            pass
    
    # Extract results
    final_centers = best_params[:2 * n].reshape(n, 2)
    final_outer_angle = best_params[2 * n]
    
    # Verify separation; if violated, fall back to a known-valid spread
    if not _check_separation(final_centers, d=1.0):
        # Spread out more to ensure separation
        # Scale centers until separated
        scale = 1.0
        for _ in range(100):
            test_centers = final_centers * scale
            if _check_separation(test_centers, d=1.0):
                final_centers = test_centers
                break
            scale *= 1.05
    
    # Compute final outer hexagon side length
    s, ok = _min_containing_hex_side(final_centers, d=1.0, outer_angle=final_outer_angle)
    if not ok or s < np.sqrt(12):
        s = max(s if ok else 8.0, np.sqrt(12) + 0.1)
    
    # Build output arrays
    inner_hex_data = np.zeros((n, 3))
    inner_hex_data[:, 0] = final_centers[:, 0]
    inner_hex_data[:, 1] = final_centers[:, 1]
    inner_hex_data[:, 2] = 0.0  # all inner hexagons at angle 0
    
    outer_hex_data = np.array([0.0, 0.0, np.degrees(final_outer_angle)])
    
    return inner_hex_data, outer_hex_data, float(s)


# EVOLVE-BLOCK-END
