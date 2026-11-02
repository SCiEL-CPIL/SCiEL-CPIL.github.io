"""Boundary Tracing: outer hexagon as 6 adjustable support planes,
inner hexagons optimized to maximize min clearance via smooth surrogate."""
import math
import time
import numpy as np

N = 12
T0 = time.time()
DEADLINE = 150.0
MARGIN = 10.0

# Unit regular hexagon vertices (flat-top orientation, circumradius = side = 1)
# Vertices at angles 0,60,120,180,240,300 degrees
_HEX_ANGS = np.array([0, 60, 120, 180, 240, 300]) * (math.pi / 180.0)
_HEX_VERTS = np.stack([np.cos(_HEX_ANGS), np.sin(_HEX_ANGS)], axis=1)  # (6,2)
# Edge outward normals: perpendicular to each edge, pointing outward
_EDGE_MIDS = (_HEX_VERTS + np.roll(_HEX_VERTS, -1, axis=0)) / 2.0
_EDGE_DIRS = np.roll(_HEX_VERTS, -1, axis=0) - _HEX_VERTS
_EDGE_DIRS = _EDGE_DIRS / np.linalg.norm(_EDGE_DIRS, axis=1, keepdims=True)
# Outward normals (rotate edge dir by -90 deg)
_EDGE_NORMALS = np.stack([_EDGE_DIRS[:, 1], -_EDGE_DIRS[:, 0]], axis=1)
# Make sure normals point outward (dot with mid > 0)
for i in range(6):
    if np.dot(_EDGE_NORMALS[i], _EDGE_MIDS[i]) < 0:
        _EDGE_NORMALS[i] = -_EDGE_NORMALS[i]
        _EDGE_DIRS[i] = -_EDGE_DIRS[i]
# Support distance (apothem) for unit hex = cos(30deg) = sqrt(3)/2
_APOTHEM = math.sqrt(3.0) / 2.0

def _hex_support_dist(center, angle_deg):
    """For a unit hexagon at given center and rotation, return (6,) support
    distances along the 6 canonical outward directions of a flat-top hexagon.
    These are the distances from center to each supporting plane of the hexagon."""
    # The support distance in a direction u is max over vertices of dot(v, u)
    # where v are vertices relative to center.
    # Vertices of rotated hexagon: R(angle) @ _HEX_VERTS
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    R = np.array([[ca, -sa], [sa, ca]])
    verts = R @ _HEX_VERTS  # (6,2)
    # We want support distances along the 6 canonical directions (outward normals of a flat-top hex)
    # canonical directions: normals of a flat-top hexagon = directions at 30,90,150,210,270,330 deg
    # Actually the support planes of the hexagon are perpendicular to the edge normals.
    # For containment in outer hex, we need: for each of the 6 outer edge directions,
    # the max projection of inner hex vertices onto that direction + center projection <= outer support.
    # Let's use the 6 directions = the 6 edge normals of a flat-top unit hex (pointing outward).
    # For a flat-top hex, edges are at angles 30,90,150,210,270,330 (normals)
    # Actually let me just use the 6 vertex directions of a pointy-top hex as the 6 "corner" directions,
    # or better: use 6 directions evenly spaced. The outer hexagon has 6 edges; each edge has an outward normal.
    # For a regular hexagon centered at origin with side s, the outward normals of edges are at angles
    # 30, 90, 150, 210, 270, 330 degrees (for flat-top) or 0,60,120,180,240,300 (for pointy-top).
    # Let's use pointy-top outer: normals at 0,60,120,180,240,300.
    # Support distance of inner hex in direction d = max_i dot(verts_i, d)
    # (verts are relative to center; center handled separately)
    # 6 directions for outer hexagon edges (pointy-top):
    dirs = _HEX_ANGS  # 0,60,...,300
    cos_d = np.cos(dirs)
    sin_d = np.sin(dirs)
    # Projection of each vertex onto each direction
    # verts: (6,2), dirs: (6,2) -> (6,6)
    projections = verts @ np.stack([cos_d, sin_d], axis=1).T  # (6,6)
    support = projections.max(axis=0)  # (6,)
    return support

def _min_clearance(inner_pts, inner_angles, outer_center, outer_side, outer_angle):
    """Compute minimum clearance between inner hexagons and outer hexagon boundary.
    Positive means all inner hexagons are strictly inside with margin.
    Returns the minimum over all inner hexagons and all 6 outer edge directions."""
    oc = np.array(outer_center)
    oa = math.radians(outer_angle)
    # Outer hexagon: 6 edge outward normals at angles outer_angle + 0,60,...,300
    outer_dirs_ang = oa + _HEX_ANGS
    outer_cos = np.cos(outer_dirs_ang)
    outer_sin = np.sin(outer_dirs_ang)
    # Outer support distance (apothem) = outer_side * sqrt(3)/2
    outer_apothem = outer_side * math.sqrt(3.0) / 2.0
    # For each inner hexagon, compute max projection of its vertices onto each outer direction
    min_clr = float('inf')
    for (cx, cy), ang in zip(inner_pts, inner_angles):
        c = np.array([cx, cy]) - oc
        a = math.radians(ang)
        ca, sa = math.cos(a), math.sin(a)
        R = np.array([[ca, -sa], [sa, ca]])
        verts = R @ _HEX_VERTS  # (6,2)
        # Project vertices onto outer directions
        # For direction d, the constraint is: dot(c, d) + max_i dot(verts_i, d) <= outer_apothem
        cos_d = outer_cos
        sin_d = outer_sin
        proj = verts @ np.stack([cos_d, sin_d], axis=1).T  # (6,6)
        max_proj = proj.max(axis=0)  # (6,)
        clr = outer_apothem - (np.dot(c, outer_cos) + np.dot(c, outer_sin) * 0.0 + (np.array([np.dot(c, np.array([math.cos(da), math.sin(da)])) for da in outer_dirs_ang])))
        # Simpler:
        center_projs = np.array([np.dot(c, np.array([math.cos(da), math.sin(da)])) for da in outer_dirs_ang])
        clr = outer_apothem - (center_projs + max_proj)
        min_clr = min(min_clr, clr.min())
    return min_clr

def _objective(params, outer_center, outer_side, outer_angle):
    """Negative min clearance (we maximize clearance)."""
    # params: (N*3,) = [x1,y1,a1, x2,y2,a2, ...]
    inner_pts = [(params[i*3], params[i*3+1]) for i in range(N)]
    inner_angles = [params[i*3+2] for i in range(N)]
    clr = _min_clearance(inner_pts, inner_angles, outer_center, outer_side, outer_angle)
    return -clr

def _check_validity(inner, outer_center, outer_side, outer_angle):
    """Check all inner hexagons are inside outer hexagon (non-negative clearance)."""
    inner_pts = [(r[0], r[1]) for r in inner]
    inner_angles = [r[2] for r in inner]
    clr = _min_clearance(inner_pts, inner_angles, outer_center, outer_side, outer_angle)
    return clr >= -1e-9

def construct():
    global T0
    t_start = time.time()
    
    # Strategy: 
    # 1. Start with a reasonable initial packing
    # 2. Use scipy optimization to maximize min clearance
    # 3. Compute the minimal outer side that contains all inner hexagons
    # 4. The score is the outer side length (lower is better)
    
    # Initial configuration: place hexagons in a grid-like pattern
    # For 12 hexagons, try a 3x4 or 2x6 arrangement
    # Unit hexagons have width 2 (flat-to-flat for pointy-top) or sqrt(3) (flat-top)
    # Let's use flat-top hexagons (vertices at 0,60,...)
    # Width (vertex to vertex) = 2, height (flat to flat) = sqrt(3)
    
    # Initial: 3 columns x 4 rows, offset every other row
    # Flat-top hex: width=2, height=sqrt(3)
    # Horizontal spacing = 1.5 (3/4 of width for offset rows)
    # Vertical spacing = sqrt(3)
    
    init_params = np.zeros(N * 3)
    cols, rows = 3, 4
    for i in range(N):
        col = i % cols
        row = i // cols
        x = col * 1.5 * 2.0  # 3.0 spacing
        y = row * math.sqrt(3.0)
        if row % 2 == 1:
            x += 1.0  # offset
        init_params[i*3] = x
        init_params[i*3+1] = y
        init_params[i*3+2] = 0.0  # angle in degrees
    
    # Center the configuration
    xs = init_params[0::3]
    ys = init_params[1::3]
    cx = (xs.max() + xs.min()) / 2.0
    cy = (ys.max() + ys.min()) / 2.0
    init_params[0::3] -= cx
    init_params[1::3] -= cy
    
    best_params = init_params.copy()
    best_outer_side = None
    best_score = float('inf')
    
    # Try to optimize using scipy
    try:
        from scipy.optimize import minimize, differential_evolution
    except ImportError:
        from scipy.optimize import minimize
    
    # For optimization, we fix outer hexagon at origin with some initial side,
    # and maximize min clearance. Then compute actual required side.
    
    # The key insight: for a given inner configuration, the minimal outer hexagon
    # side is determined by the maximum over all 6 directions of:
    #   max over inner hexagons of (|projection of center| + support_dist_in_that_dir)
    # Then outer_side = max_val / (sqrt(3)/2)
    
    def _compute_min_outer_side(inner_pts, inner_angles, outer_angle=0.0):
        """Compute the minimum side length of a regular hexagon (at given angle) 
        centered at origin that contains all inner hexagons."""
        oa = math.radians(outer_angle)
        outer_dirs_ang = oa + _HEX_ANGS
        max_proj = 0.0
        for (cx, cy), ang in zip(inner_pts, inner_angles):
            c = np.array([cx, cy])
            a = math.radians(ang)
            ca, sa = math.cos(a), math.sin(a)
            R = np.array([[ca, -sa], [sa, ca]])
            verts = R @ _HEX_VERTS
            for da in outer_dirs_ang:
                d = np.array([math.cos(da), math.sin(da)])
                proj = np.dot(c, d) + (verts @ d).max()
                max_proj = max(max_proj, abs(proj))
        # outer_apothem = max_proj, outer_side = outer_apothem / (sqrt(3)/2)
        return max_proj / (math.sqrt(3.0) / 2.0)
    
    # Optimize: minimize the computed outer side length
    def _outer_side_objective(params):
        if time.time() - t_start > DEADLINE - MARGIN:
            return best_score  # signal timeout
        inner_pts = [(params[i*3], params[i*3+1]) for i in range(N)]
        inner_angles = [params[i*3+2] for i in range(N)]
        return _compute_min_outer_side(inner_pts, inner_angles)
    
    # Try differential evolution for global search
    bounds = [(-5, 5)] * (2*N) + [(-180, 180)] * N
    
    t_de_start = time.time()
    try:
        result = differential_evolution(
            _outer_side_objective, bounds,
            seed=42, maxiter=200, tol=1e-6,
            popsize=15, mutation=(0.5, 1.5), recombination=0.8,
            polish=True, init='latinhypercube',
            updating='deferred', seed=42
        )
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x.copy()
    except Exception:
        pass
    
    # Polish with L-BFGS-B
    t_polish = time.time()
    try:
        res2 = minimize(
            _outer_side_objective, best_params,
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if res2.fun < best_score:
            best_score = res2.fun
            best_params = res2.x.copy()
    except Exception:
        pass
    
    # Try a few random restarts if time permits
    rng = np.random.RandomState(123)
    for restart in range(5):
        if time.time() - t_start > DEADLINE - MARGIN:
            break
        # Perturb from best
        perturb = best_params + rng.randn(N*3) * 0.5
        try:
            res3 = minimize(
                _outer_side_objective, perturb,
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res3.fun < best_score:
                best_score = res3.fun
                best_params = res3.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() - t_start < DEADLINE - MARGIN:
        try:
            res4 = minimize(
                _outer_side_objective, best_params,
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res4.fun < best_score:
                best_score = res4.fun
                best_params = res4.x.copy()
        except Exception:
            pass
    
    # Build final result
    inner = []
    for i in range(N):
        inner.append([float(best_params[i*3]), float(best_params[i*3+1]), float(best_params[i*3+2])])
    
    outer_center = (0.0, 0.0)
    outer_side = float(best_score)
    outer_angle = 0.0
    
    # Safety check: ensure validity
    if not _check_validity(inner, outer_center, outer_side, outer_angle):
        # Add small buffer
        outer_side *= 1.001
    
    return inner, outer_center, outer_side, outer_angle
