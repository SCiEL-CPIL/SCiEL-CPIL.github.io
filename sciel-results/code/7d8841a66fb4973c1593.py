import time
import math
import numpy as np
from scipy.optimize import minimize

N = 11
T0 = time.time()
DEADLINE = 145.0
SQRT3 = math.sqrt(3.0)


def _hex_vertices_local(theta):
    """Vertices of unit regular hexagon (circumradius 1) rotated by theta."""
    ang = np.arange(6) * (math.pi / 3.0) + theta
    vx = np.cos(ang)
    vy = np.sin(ang)
    return np.stack([vx, vy], axis=1)  # (6,2)


def _outer_hex_side_length(centers, theta, theta_outer):
    """
    Given centers of N unit hexagons all rotated by theta,
    and outer hexagon rotated by theta_outer (centered at origin),
    return the minimum side length s of the outer hexagon such that
    all inner hexagons are contained.
    
    The outer hexagon has center (0,0), side length s, rotation theta_outer.
    Its apothem (distance from center to side midpoint) is s * sqrt(3) / 2.
    A point (x,y) is inside if for all 6 side normals n_i, dot(point, n_i) <= apothem.
    """
    # All inner hexagons have the same rotation theta
    verts_local = _hex_vertices_local(theta)  # (6,2)
    # All vertices: (N*6, 2)
    verts = centers[:, None, :] + verts_local[None, :, :]  # (N,6,2)
    verts = verts.reshape(-1, 2)  # (N*6, 2)
    
    # Outward normals of outer hexagon sides
    # Side i has outward normal at angle theta_outer + i*pi/3 + pi/6
    normals_ang = theta_outer + np.arange(6) * (math.pi / 3.0) + math.pi / 6.0
    normals = np.stack([np.cos(normals_ang), np.sin(normals_ang)], axis=1)  # (6,2)
    
    # Projection of each vertex onto each normal
    proj = verts @ normals.T  # (N*6, 6)
    max_proj = proj.max()  # maximum projection over all vertices and all sides
    
    # apothem = max_proj, so s = 2 * apothem / sqrt(3)
    s = 2.0 * max_proj / SQRT3
    return s


def _pairwise_distance_penalty(centers):
    """
    Penalty for pairs of hexagons closer than 2 (which would overlap).
    Uses smooth penalty: sum over pairs of max(0, 2 - dist)^2
    """
    diff = centers[:, None, :] - centers[None, :, :]  # (N,N,2)
    d = np.linalg.norm(diff, axis=2)  # (N,N)
    iu = np.triu_indices(N, k=1)
    dpair = d[iu]
    viol = np.maximum(0.0, 2.0 - dpair)
    return np.sum(viol * viol)


def _objective(params, theta_outer):
    """
    params: [x0, y0, ..., xN-1, yN-1, theta]
    theta is the common rotation of all inner hexagons.
    """
    centers = params[:N * 2].reshape(N, 2)
    theta = params[N * 2]
    s = _outer_hex_side_length(centers, theta, theta_outer)
    pen = _pairwise_distance_penalty(centers)
    return s + 500.0 * pen


def _is_valid(centers, theta, theta_outer):
    """Check if the configuration is valid: no overlaps, all contained."""
    # Check pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    if d[iu].min() < 2.0 - 1e-6:
        return False
    
    # Check containment
    s = _outer_hex_side_length(centers, theta, theta_outer)
    verts_local = _hex_vertices_local(theta)
    verts = (centers[:, None, :] + verts_local[None, :, :]).reshape(-1, 2)
    normals_ang = theta_outer + np.arange(6) * (math.pi / 3.0) + math.pi / 6.0
    normals = np.stack([np.cos(normals_ang), np.sin(normals_ang)], axis=1)
    apothem = s * SQRT3 / 2.0
    proj = verts @ normals.T
    if proj.max() > apothem + 1e-6:
        return False
    return True


def _make_hex_lattice_init(seed):
    """
    Create initial positions based on hexagonal lattice.
    For N=11: center + 6 ring + 4 outer.
    """
    rng = np.random.RandomState(seed)
    mode = seed % 5
    
    if mode == 0:
        # Standard hex lattice: center, 6 at distance 2, 4 at distance ~3.46
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * math.pi / 3.0
            centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
        # 4 outer hexagons at positions between ring hexagons
        outer_positions = [
            (3.0 * math.cos(math.pi/6), 3.0 * math.sin(math.pi/6)),
            (3.0 * math.cos(5*math.pi/6), 3.0 * math.sin(5*math.pi/6)),
            (3.0 * math.cos(7*math.pi/6), 3.0 * math.sin(7*math.pi/6)),
            (3.0 * math.cos(11*math.pi/6), 3.0 * math.sin(11*math.pi/6)),
        ]
        for i in range(4):
            centers[i + 7] = outer_positions[i]
    elif mode == 1:
        # Slightly rotated lattice
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * math.pi / 3.0 + 0.1
            centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * math.pi / 3.0 + 0.1
            centers[i + 7] = [3.3 * math.cos(ang), 3.3 * math.sin(ang)]
    elif mode == 2:
        # Dense packing: center, 6 close, 4 slightly further
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * math.pi / 3.0
            centers[i + 1] = [1.95 * math.cos(ang), 1.95 * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * math.pi / 3.0
            centers[i + 7] = [3.2 * math.cos(ang), 3.2 * math.sin(ang)]
    elif mode == 3:
        # Random perturbation of lattice
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * math.pi / 3.0
            centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * math.pi / 3.0
            centers[i + 7] = [3.4 * math.cos(ang), 3.4 * math.sin(ang)]
        centers += rng.normal(0, 0.1, size=(N, 2))
    else:
        # Random
        centers = rng.uniform(-3, 3, size=(N, 2))
    
    theta = rng.uniform(-math.pi / 6, math.pi / 6)
    params = np.concatenate([centers.reshape(-1), [theta]])
    return params


def _optimize_theta_outer(theta_outer, init_params, maxiter=500):
    """Optimize inner positions and common rotation for fixed outer rotation."""
    try:
        res = minimize(_objective, init_params, args=(theta_outer,),
                       method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12})
        return res.x
    except Exception:
        return init_params


def _polish(params, theta_outer):
    """High-precision polish."""
    try:
        res = minimize(_objective, params, args=(theta_outer,),
                       method='SLSQP', options={'maxiter': 300, 'ftol': 1e-15})
        if res.success or res.fun < _objective(params, theta_outer):
            return res.x
    except Exception:
        pass
    try:
        res = minimize(_objective, params, args=(theta_outer,),
                       method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13})
        if res.success or res.fun < _objective(params, theta_outer):
            return res.x
    except Exception:
        pass
    return params


def _optimize_all(params, theta_outer, maxiter=1000):
    """
    Optimize all parameters including theta_outer.
    params: [x0, y0, ..., xN-1, yN-1, theta, theta_outer]
    """
    def obj_full(p):
        centers = p[:N * 2].reshape(N, 2)
        theta = p[N * 2]
        theta_o = p[N * 2 + 1]
        return _objective(np.concatenate([centers.reshape(-1), [theta]]), theta_o)
    
    try:
        res = minimize(obj_full, params,
                       method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12})
        return res.x
    except Exception:
        return params


def construct():
    global T0
    
    best_score = float('inf')
    best_result = None
    
    # Try multiple outer rotations and initial configurations
    outer_rotations = [0.0, math.pi / 12.0, math.pi / 6.0, math.pi / 4.0, math.pi / 3.0, 5*math.pi/12.0]
    
    seed = 0
    max_attempts = 30
    
    for theta_o in outer_rotations:
        for attempt in range(max_attempts):
            if time.time() - T0 > DEADLINE - 15:
                break
            
            seed += 1
            init_params = _make_hex_lattice_init(seed)
            
            # Optimize inner positions and common theta
            params = _optimize_theta_outer(theta_o, init_params, maxiter=300)
            
            centers = params[:N * 2].reshape(N, 2)
            theta = params[N * 2]
            
            if not _is_valid(centers, theta, theta_o):
                continue
            
            s = _outer_hex_side_length(centers, theta, theta_o)
            
            if s < best_score:
                best_score = s
                best_result = (centers.copy(), theta, theta_o)
            
            # Polish
            if time.time() - T0 > DEADLINE - 10:
                break
            
            polished = _polish(params, theta_o)
            centers_p = polished[:N * 2].reshape(N, 2)
            theta_p = polished[N * 2]
            
            if _is_valid(centers_p, theta_p, theta_o):
                s_p = _outer_hex_side_length(centers_p, theta_p, theta_o)
                if s_p < best_score:
                    best_score = s_p
                    best_result = (centers_p.copy(), theta_p, theta_o)
        
        if time.time() - T0 > DEADLINE - 15:
            break
    
    # If we found a good result, try optimizing theta_outer as well
    if best_result is not None and time.time() - T0 < DEADLINE - 5:
        centers, theta, theta_o = best_result
        full_params = np.concatenate([centers.reshape(-1), [theta], [theta_o]])
        full_optimized = _optimize_all(full_params, theta_o, maxiter=200)
        
        centers_f = full_optimized[:N * 2].reshape(N, 2)
        theta_f = full_optimized[N * 2]
        theta_o_f = full_optimized[N * 2 + 1]
        
        if _is_valid(centers_f, theta_f, theta_o_f):
            s_f = _outer_hex_side_length(centers_f, theta_f, theta_o_f)
            if s_f < best_score:
                best_score = s_f
                best_result = (centers_f.copy(), theta_f, theta_o_f)
    
    if best_result is None:
        # Fallback: simple hex lattice
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * math.pi / 3.0
            centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * math.pi / 3.0
            centers[i + 7] = [3.4 * math.cos(ang), 3.4 * math.sin(ang)]
        theta = 0.0
        s = _outer_hex_side_length(centers, theta, 0.0)
        inner = [[float(centers[i, 0]), float(centers[i, 1]), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), float(s), 0.0
    
    centers, theta, theta_o = best_result
    s = _outer_hex_side_length(centers, theta, theta_o)
    inner = [[float(centers[i, 0]), float(centers[i, 1]), float(math.degrees(theta))] for i in range(N)]
    return inner, (0.0, 0.0), float(s), float(math.degrees(theta_o))
