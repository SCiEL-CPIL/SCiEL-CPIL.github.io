import time
import math
import numpy as np
from scipy.optimize import minimize

N = 11
T0 = time.time()
DEADLINE = 148.0
SQRT3 = math.sqrt(3.0)
TWO_PI = 2.0 * math.pi


def _hex_verts(theta):
    """Vertices of unit hexagon (circumradius=1) rotated by theta."""
    ang = np.arange(6) * (math.pi / 3.0) + theta
    return np.column_stack([np.cos(ang), np.sin(ang)])


def _outer_normals(theta_o):
    """Outward normals of 6 sides of outer hexagon."""
    ang = theta_o + np.arange(6) * (math.pi / 3.0) + math.pi / 6.0
    return np.column_stack([np.cos(ang), np.sin(ang)])


def _min_side(theta, centers, theta_o):
    """Minimum outer side length to contain all inner hexagons."""
    verts_local = _hex_verts(theta)
    verts = (centers[:, None, :] + verts_local[None, :, :]).reshape(-1, 2)
    normals = _outer_normals(theta_o)
    proj = verts @ normals.T
    apothem = proj.max()
    return apothem * 2.0 / SQRT3


def _all_constraints(params, theta_o):
    """Return list of constraint functions for SLSQP."""
    centers = params[:N * 2].reshape(N, 2)
    theta = params[N * 2]
    verts_local = _hex_verts(theta)
    verts = (centers[:, None, :] + verts_local[None, :, :]).reshape(-1, 2)
    normals = _outer_normals(theta_o)
    
    constraints = []
    
    # Non-overlap: distance between centers >= 2 (for non-rotated hexagons)
    # For rotated hexagons, we need to check vertex-to-vertex distances
    # Simplified: use center distance >= 2 as necessary condition
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: p[i*2] - p[j*2],  # x_i - x_j
                'jac': None
            })
    
    # Actually, let's use a simpler approach with penalty in objective
    # and just check validity at the end
    
    return []


def _objective(params, theta_o):
    """Objective: minimize outer side length + penalty for overlaps."""
    centers = params[:N * 2].reshape(N, 2)
    theta = params[N * 2]
    
    s = _min_side(theta, centers, theta_o)
    
    # Penalty for overlapping hexagons
    # For two unit hexagons, they overlap if center distance < 2
    # (this is conservative; actual minimum distance depends on rotation)
    pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            d = math.sqrt(dx * dx + dy * dy)
            if d < 2.0:
                pen += (2.0 - d) ** 2
    
    return s + 100.0 * pen


def _is_valid(centers, theta, theta_o, tol=1e-6):
    """Check if configuration is valid (no overlaps, contained)."""
    # Check non-overlap: all pairs of hexagons must not intersect
    # Two unit hexagons don't overlap iff distance between centers >= 2
    # (for the same orientation; for different orientations, we need more care)
    # Conservative check: center distance >= 2 - small_tol
    for i in range(N):
        for j in range(i + 1, N):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            d = math.sqrt(dx * dx + dy * dy)
            if d < 2.0 - tol:
                return False
    
    # Check containment: all vertices inside outer hexagon
    verts_local = _hex_verts(theta)
    verts = (centers[:, None, :] + verts_local[None, :, :]).reshape(-1, 2)
    normals = _outer_normals(theta_o)
    s = _min_side(theta, centers, theta_o)
    apothem = s * SQRT3 / 2.0
    proj = verts @ normals.T
    if proj.max() > apothem + tol:
        return False
    
    return True


def _make_hex_grid_init():
    """Hexagonal lattice: center + 6 ring + 4 outer."""
    centers = np.zeros((N, 2))
    centers[0] = [0, 0]
    for i in range(6):
        ang = i * math.pi / 3.0
        centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
    for i in range(4):
        ang = (i + 0.5) * math.pi / 3.0
        centers[i + 7] = [3.5 * math.cos(ang), 3.5 * math.sin(ang)]
    return centers


def _make_offset_init():
    """Offset hexagonal lattice."""
    centers = np.zeros((N, 2))
    centers[0] = [0, 0]
    for i in range(6):
        ang = i * math.pi / 3.0 + math.pi / 6.0
        centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
    for i in range(4):
        ang = (i + 0.5) * math.pi / 3.0
        centers[i + 7] = [3.4 * math.cos(ang), 3.4 * math.sin(ang)]
    return centers


def _make_compact_init():
    """Compact cluster arrangement."""
    centers = np.zeros((N, 2))
    positions = [
        (0, 0),
        (2, 0), (-1, 1.732), (1, 1.732),
        (-1, -1.732), (1, -1.732),
        (3, 0), (-2, 1.732), (2, 1.732),
        (-2, -1.732), (2, -1.732)
    ]
    for i in range(N):
        centers[i] = positions[i]
    return centers


def _make_random_init(rng):
    """Random initialization with jitter around hex grid."""
    base = _make_hex_grid_init()
    jitter = rng.uniform(-0.3, 0.3, size=(N, 2))
    centers = base + jitter
    return centers


def _optimize_single(init_params, theta_o, maxiter=200, tol=1e-10):
    """Optimize using L-BFGS-B."""
    try:
        res = minimize(_objective, init_params, args=(theta_o,),
                       method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': tol, 'gtol': 1e-8})
        return res.x
    except Exception:
        return init_params


def _polish(params, theta_o):
    """High-precision polish."""
    best_params = params
    best_obj = _objective(params, theta_o)
    
    # SLSQP polish
    try:
        res = minimize(_objective, params, args=(theta_o,),
                       method='SLSQP',
                       options={'maxiter': 300, 'ftol': 1e-14})
        if res.fun < best_obj - 1e-10:
            best_params = res.x
            best_obj = res.fun
    except Exception:
        pass
    
    # Tight L-BFGS-B
    try:
        res = minimize(_objective, best_params, args=(theta_o,),
                       method='L-BFGS-B',
                       options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
        if res.fun < best_obj - 1e-10:
            best_params = res.x
            best_obj = res.fun
    except Exception:
        pass
    
    return best_params


def construct():
    global T0
    
    best_score = float('inf')
    best_result = None
    
    # Multiple outer rotation angles to try
    outer_rotations = [0.0, math.pi/6, math.pi/3, math.pi/4, math.pi/2]
    
    # Different initial configurations
    init_modes = [
        'hex_grid', 'offset', 'compact', 'random'
    ]
    
    rng = np.random.RandomState(42)
    
    for theta_o in outer_rotations:
        if time.time() - T0 > DEADLINE - 15:
            break
        
        for mode in init_modes:
            if time.time() - T0 > DEADLINE - 15:
                break
            
            if mode == 'hex_grid':
                centers = _make_hex_grid_init()
            elif mode == 'offset':
                centers = _make_offset_init()
            elif mode == 'compact':
                centers = _make_compact_init()
            else:
                centers = _make_random_init(rng)
            
            theta_init = rng.uniform(-math.pi/6, math.pi/6)
            init_params = np.concatenate([centers.reshape(-1), [theta_init]])
            
            # Optimize
            params = _optimize_single(init_params, theta_o, maxiter=300)
            
            centers_p = params[:N * 2].reshape(N, 2)
            theta_p = params[N * 2]
            
            if not _is_valid(centers_p, theta_p, theta_o):
                continue
            
            s = _min_side(theta_p, centers_p, theta_o)
            if s < best_score:
                best_score = s
                best_result = (centers_p.copy(), theta_p, theta_o)
        
        # Try a few more random inits for this rotation
        for _ in range(3):
            if time.time() - T0 > DEADLINE - 15:
                break
            
            centers = _make_random_init(rng)
            theta_init = rng.uniform(-math.pi/6, math.pi/6)
            init_params = np.concatenate([centers.reshape(-1), [theta_init]])
            
            params = _optimize_single(init_params, theta_o, maxiter=200)
            
            centers_p = params[:N * 2].reshape(N, 2)
            theta_p = params[N * 2]
            
            if not _is_valid(centers_p, theta_p, theta_o):
                continue
            
            s = _min_side(theta_p, centers_p, theta_o)
            if s < best_score:
                best_score = s
                best_result = (centers_p.copy(), theta_p, theta_o)
    
    # Polish the best solution
    if best_result is not None and time.time() - T0 < DEADLINE - 2:
        centers_b, theta_b, theta_o_b = best_result
        params_b = np.concatenate([centers_b.reshape(-1), [theta_b]])
        
        polished = _polish(params_b, theta_o_b)
        centers_pol = polished[:N * 2].reshape(N, 2)
        theta_pol = polished[N * 2]
        
        if _is_valid(centers_pol, theta_pol, theta_o_b):
            s_pol = _min_side(theta_pol, centers_pol, theta_o_b)
            if s_pol < best_score:
                best_score = s_pol
                best_result = (centers_pol.copy(), theta_pol, theta_o_b)
    
    if best_result is None:
        # Fallback: return something (may be invalid)
        centers = np.zeros((N, 2))
        return (list(centers[:, 0].tolist()), list(centers[:, 1].tolist())), (0.0, 0.0), 40.0, 0.0
    
    centers, theta, theta_o = best_result
    s = _min_side(theta, centers, theta_o)
    inner = [[float(centers[i, 0]), float(centers[i, 1]), float(math.degrees(theta))] for i in range(N)]
    return inner, (0.0, 0.0), float(s), float(math.degrees(theta_o))
