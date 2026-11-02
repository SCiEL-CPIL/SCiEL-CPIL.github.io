import time
import math
import numpy as np
from scipy.optimize import minimize

N = 11
T0 = time.time()
DEADLINE = 145.0  # safety margin before 150s kill
TOL = 1e-7
SQRT3 = math.sqrt(3.0)


def _hex_vertices_local(theta):
    # unit hexagon vertices (circumradius 1) rotated by theta
    ang = np.arange(6) * (math.pi / 3.0) + theta
    vx = np.cos(ang)
    vy = np.sin(ang)
    return np.stack([vx, vy], axis=1)  # (6,2)


def _outer_side_normals(theta_outer):
    # outward normals of the 6 sides of the outer hexagon
    ang = theta_outer + np.arange(6) * (math.pi / 3.0) + math.pi / 6.0
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)  # (6,2)


def _outer_side(theta, centers, theta_outer):
    # centers: (N,2); returns minimal outer side length containing all vertices
    verts_local = _hex_vertices_local(theta)  # (6,2)
    verts = centers[:, None, :] + verts_local[None, :, :]  # (N,6,2)
    verts = verts.reshape(-1, 2)  # (N*6,2)
    normals = _outer_side_normals(theta_outer)  # (6,2)
    proj = verts @ normals.T  # (N*6, 6)
    apothem = proj.max()
    s = apothem * 2.0 / SQRT3
    return s


def _pairwise_penalty(centers):
    # centers (N,2); penalty for pairs closer than 2
    diff = centers[:, None, :] - centers[None, :, :]  # (N,N,2)
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    dpair = d[iu]
    viol = np.maximum(0.0, 2.0 - dpair)
    return np.sum(viol * viol)


def _objective(params, theta_outer):
    # params: (N*2 + 1) = [x0,y0,...,xN-1,yN-1, theta]
    centers = params[:N * 2].reshape(N, 2)
    theta = params[N * 2]
    s = _outer_side(theta, centers, theta_outer)
    pen = _pairwise_penalty(centers)
    return s + 1000.0 * pen


def _is_valid(centers, theta, theta_outer):
    # check non-overlap
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    if d[iu].min() < 2.0 - 1e-5:
        return False
    # check containment: all vertices inside outer hexagon
    s = _outer_side(theta, centers, theta_outer)
    verts_local = _hex_vertices_local(theta)
    verts = (centers[:, None, :] + verts_local[None, :, :]).reshape(-1, 2)
    normals = _outer_side_normals(theta_outer)
    apothem = s * SQRT3 / 2.0
    proj = verts @ normals.T  # (N*6, 6)
    if proj.max() > apothem + 1e-5:
        return False
    return True


def _make_init(seed):
    rng = np.random.RandomState(seed)
    mode = seed % 4
    if mode == 0:
        # Hexagonal lattice: center + 6 ring + 4 more
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * math.pi / 3.0
            centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * math.pi / 3.0
            centers[i + 7] = [3.5 * math.cos(ang), 3.5 * math.sin(ang)]
    elif mode == 1:
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * math.pi / 3.0 + math.pi / 6.0
            centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * math.pi / 3.0
            centers[i + 7] = [3.4 * math.cos(ang), 3.4 * math.sin(ang)]
    elif mode == 2:
        centers = np.zeros((N, 2))
        positions = [(0,0), (2,0), (-1, SQRT3), (1, SQRT3), (-1, -SQRT3), (1, -SQRT3),
                     (3, 0), (-2, SQRT3), (2, SQRT3), (-2, -SQRT3), (2, -SQRT3)]
        for i in range(N):
            centers[i] = positions[i]
    else:
        centers = rng.uniform(-3, 3, size=(N, 2))
    theta = rng.uniform(-math.pi / 6, math.pi / 6)
    params = np.concatenate([centers.reshape(-1), [theta]])
    return params


def _optimize_one(init_params, theta_outer, maxiter):
    res = minimize(_objective, init_params, args=(theta_outer,),
                   method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
    return res


def _polish(params, theta_outer):
    """High-precision polish using SLSQP with tighter tolerances."""
    try:
        res = minimize(_objective, params, args=(theta_outer,),
                       method='SLSQP', options={'maxiter': 200, 'ftol': 1e-14})
        if res.success or res.fun < _objective(params, theta_outer):
            return res.x
    except Exception:
        pass
    try:
        res = minimize(_objective, params, args=(theta_outer,),
                       method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
        if res.success or res.fun < _objective(params, theta_outer):
            return res.x
    except Exception:
        pass
    return params


def _contact_graph_params_to_centers(params):
    """
    params layout:
      [0]: theta (inner hexagon rotation)
      [1..6]: gap_i for ring hexagons (i=0..5), distance from center = 2 + gap_i
      [7..12]: angle_offset_i for ring hexagons (small rotation of direction)
      [13..16]: gap_j for outer 4 hexagons, distance from their ring parent
      [17..20]: angle_offset_j for outer 4 hexagons
    Returns centers (11,2)
    """
    theta = params[0]
    ring_gaps = params[1:7]
    ring_ang_off = params[7:13]
    outer_gaps = params[13:17]
    outer_ang_off = params[17:21]
    
    centers = np.zeros((N, 2))
    # Center hexagon at origin
    centers[0] = [0.0, 0.0]
    
    # Ring of 6
    for i in range(6):
        base_ang = i * math.pi / 3.0
        ang = base_ang + ring_ang_off[i]
        dist = 2.0 + ring_gaps[i]
        centers[i + 1] = [dist * math.cos(ang), dist * math.sin(ang)]
    
    # 4 outer hexagons attached to ring positions 0, 2, 4, 5 (or 1,3,4,5)
    # Let's attach to ring positions 0, 2, 4, 5 for a compact shape
    outer_ring_indices = [0, 2, 4, 5]
    for j in range(4):
        ri = outer_ring_indices[j]
        ring_center = centers[ri + 1]
        base_ang = ri * math.pi / 3.0
        ang = base_ang + outer_ang_off[j]
        dist = 2.0 + outer_gaps[j]
        centers[7 + j] = ring_center + dist * np.array([math.cos(ang), math.sin(ang)])
    
    return centers


def _contact_graph_objective(params, theta_outer):
    centers = _contact_graph_params_to_centers(params)
    theta = params[0]
    s = _outer_side(theta, centers, theta_outer)
    # Penalty for overlaps
    pen = _pairwise_penalty(centers)
    return s + 500.0 * pen


def _contact_graph_optimize(theta_outer, maxiter, seed):
    """Optimize the contact graph parameters."""
    rng = np.random.RandomState(seed)
    # Initial params: theta=0, gaps=0, offsets=0
    init = np.zeros(21)
    init[0] = rng.uniform(-0.1, 0.1)  # small theta
    init[1:7] = rng.uniform(0, 0.1, size=6)  # small gaps
    init[7:13] = rng.uniform(-0.05, 0.05, size=6)  # small angle offsets
    init[13:17] = rng.uniform(0, 0.1, size=4)
    init[17:21] = rng.uniform(-0.05, 0.05, size=4)
    
    try:
        res = minimize(_contact_graph_objective, init, args=(theta_outer,),
                       method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12})
        return res.x
    except Exception:
        return init


def construct():
    global T0

    best = None
    best_score = float('inf')
    best_params = None

    # Try a few outer rotations
    outer_rotations = [0.0, math.pi / 6.0, math.pi / 3.0]

    seed = 0
    maxiter = 500

    # Phase 1: Contact graph optimization
    for theta_o in outer_rotations:
        while time.time() - T0 < DEADLINE - 20:
            cg_params = _contact_graph_optimize(theta_o, maxiter, seed)
            seed += 1
            centers = _contact_graph_params_to_centers(cg_params)
            theta = cg_params[0]
            if not _is_valid(centers, theta, theta_o):
                continue
            s = _outer_side(theta, centers, theta_o)
            if s < best_score:
                best_score = s
                best = (centers.copy(), theta, theta_o)
                # Convert to standard params for polish
                best_params = np.concatenate([centers.reshape(-1), [theta]])
            if time.time() - T0 > DEADLINE - 25:
                break

    # Phase 2: General optimization from best contact graph result
    if best_params is not None and time.time() - T0 < DEADLINE - 10:
        theta_o = best[2]
        # Run general L-BFGS-B from the contact graph result
        try:
            res = minimize(_objective, best_params, args=(theta_o,),
                           method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            params = res.x
            centers = params[:N * 2].reshape(N, 2)
            theta = params[N * 2]
            if _is_valid(centers, theta, theta_o):
                s = _outer_side(theta, centers, theta_o)
                if s < best_score:
                    best_score = s
                    best = (centers.copy(), theta, theta_o)
                    best_params = params.copy()
        except Exception:
            pass

    # Phase 3: Polish
    if best_params is not None and time.time() - T0 < DEADLINE:
        theta_o = best[2]
        polished = _polish(best_params, theta_o)
        centers_p = polished[:N * 2].reshape(N, 2)
        theta_p = polished[N * 2]
        if _is_valid(centers_p, theta_p, theta_o):
            s_p = _outer_side(theta_p, centers_p, theta_o)
            if s_p < best_score:
                best_score = s_p
                best = (centers_p.copy(), theta_p, theta_o)

    if best is None:
        # Fallback: use contact graph with zero gaps
        cg_params = np.zeros(21)
        centers = _contact_graph_params_to_centers(cg_params)
        theta = 0.0
        s = _outer_side(theta, centers, 0.0)
        inner = [[float(centers[i, 0]), float(centers[i, 1]), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), float(s), 0.0

    centers, theta, theta_o = best
    s = _outer_side(theta, centers, theta_o)
    inner = [[float(centers[i, 0]), float(centers[i, 1]), float(math.degrees(theta))] for i in range(N)]
    return inner, (0.0, 0.0), float(s), float(math.degrees(theta_o))
