import time
import math
import numpy as np
from scipy.optimize import linprog

N = 11
T0 = time.time()
DEADLINE = 145.0  # safety margin before 150s kill
TOL = 1e-7


def _hex_vertices_local(theta):
    # unit hexagon vertices (circumradius 1) rotated by theta
    ang = np.arange(6) * (math.pi / 3.0) + theta
    vx = np.cos(ang)
    vy = np.sin(ang)
    return np.stack([vx, vy], axis=1)  # (6,2)


def _outer_side_normals(theta_outer):
    # outward normals of the 6 sides of the outer hexagon
    # side i has outward normal at angle theta_outer + i*pi/3 + pi/6
    ang = theta_outer + np.arange(6) * (math.pi / 3.0) + math.pi / 6.0
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)  # (6,2)


def _outer_side(theta, centers, theta_outer):
    # centers: (N,2); returns minimal outer side length containing all vertices
    verts_local = _hex_vertices_local(theta)  # (6,2)
    # all world vertices: (N*6, 2)
    verts = centers[:, None, :] + verts_local[None, :, :]  # (N,6,2)
    verts = verts.reshape(-1, 2)  # (N*6,2)
    normals = _outer_side_normals(theta_outer)  # (6,2)
    proj = verts @ normals.T  # (N*6, 6)
    apothem = proj.max()
    s = apothem * 2.0 / math.sqrt(3.0)
    return s


def _min_enclosing_hex_lp(verts):
    """
    Find the minimum-side-length regular hexagon (any center, any rotation)
    enclosing all vertices. Returns (side_length, center, rotation_angle).
    
    Uses LP: minimize t subject to for all 6 sides of outer hexagon:
    n_i(theta) . (v_j - c) <= t*sqrt(3)/2 for all j
    where n_i(theta) are outward normals of outer hexagon with rotation theta.
    
    We search over theta in fine steps, and for each theta solve an LP for (c, t).
    Variables: c_x, c_y, t (3 vars)
    Constraints: for each vertex v_j and each side i:
    n_ix * c_x + n_iy * c_y + t*sqrt(3)/2 >= n_ix * v_jx + n_iy * v_jy
    i.e., -(n_ix*c_x + n_iy*c_y) - t*sqrt(3)/2 <= -(n_ix*v_jx + n_iy*v_jy)
    
    Also need t >= 0.
    """
    best_s = float('inf')
    best_c = None
    best_theta = None
    
    n_verts = len(verts)
    # Search over outer rotation angles
    # Due to 6-fold symmetry, only need 0 to pi/3
    for theta_o in np.linspace(0, math.pi/3, 181):
        normals = _outer_side_normals(theta_o)  # (6,2)
        # LP variables: [cx, cy, t]
        # Objective: minimize t
        c_obj = np.array([0.0, 0.0, 1.0])
        
        A_ub = np.zeros((n_verts * 6, 3))
        b_ub = np.zeros(n_verts * 6)
        
        sqrt3_half = math.sqrt(3.0) / 2.0
        for i in range(6):
            nx, ny = normals[i]
            for j in range(n_verts):
                idx = i * n_verts + j
                # Constraint: nx*cx + ny*cy + t*sqrt3_half >= nx*vx + ny*vy
                # => -nx*cx - ny*cy - sqrt3_half*t <= -(nx*vx + ny*vy)
                A_ub[idx, 0] = -nx
                A_ub[idx, 1] = -ny
                A_ub[idx, 2] = -sqrt3_half
                b_ub[idx] = -(nx * verts[j, 0] + ny * verts[j, 1])
        
        # t >= 0 => -t <= 0
        A_ub = np.vstack([A_ub, [[0, 0, -1.0]]])
        b_ub = np.append(b_ub, 0.0)
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, method='highs',
                          options={'presolve': True})
            if res.status == 0:
                t_val = res.x[2]
                s_val = t_val / sqrt3_half
                if s_val < best_s:
                    best_s = s_val
                    best_c = np.array([res.x[0], res.x[1]])
                    best_theta = theta_o
        except Exception:
            continue
    
    return best_s, best_c, best_theta


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


def _is_valid(centers, theta, theta_outer, outer_center=None):
    """Check validity given a specific outer hexagon (center and rotation)."""
    # check non-overlap
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    if d[iu].min() < 2.0 - 1e-5:
        return False
    # check containment: all vertices inside outer hexagon
    verts_local = _hex_vertices_local(theta)
    verts = (centers[:, None, :] + verts_local[None, :, :]).reshape(-1, 2)
    normals = _outer_side_normals(theta_outer)
    
    if outer_center is not None:
        # Compute required side length with this specific outer hexagon
        rel_verts = verts - outer_center
        proj = rel_verts @ normals.T
        apothem_needed = proj.max()
        s = apothem_needed * 2.0 / math.sqrt(3.0)
        if apothem_needed < 0:
            # All vertices on negative side, shouldn't happen for enclosing
            return True
        # The outer hexagon with side s centered at outer_center encloses them
        # We just need to verify it's a valid enclosing hexagon (which it is by construction)
        return True
    else:
        # Legacy: assume outer center at origin
        proj = verts @ normals.T
        apothem = proj.max()
        if apothem < 0:
            return True
        return True


def _make_init(seed):
    rng = np.random.RandomState(seed)
    # Use a structured initial guess: hexagonal lattice packing
    # Place hexagons in a hexagonal arrangement around center
    # For N=11: 1 center + 6 in first ring + 4 in second ring (partial)
    # Or just use a grid that's likely to be feasible
    centers = np.zeros((N, 2))
    
    # Try a few different structured patterns based on seed
    pattern = seed % 3
    
    if pattern == 0:
        # Hexagonal lattice: center at origin, rings of 6, 12, ...
        # For 11 hexagons: 1 + 6 + 4 (from second ring)
        # Hex lattice spacing: 2*sqrt(3) for non-overlapping (distance between centers >= 2)
        # Actually for hexagons of side 1, distance between centers must be >= 2
        # Hex lattice with spacing 2: neighbors at distance 2
        positions = []
        # Center
        positions.append((0, 0))
        # First ring: 6 hexagons at distance 2, angles 0, 60, 120, ...
        for k in range(6):
            ang = k * math.pi / 3
            positions.append((2 * math.cos(ang), 2 * math.sin(ang)))
        # Second ring: pick 4 from the 12 available
        # Second ring positions at distance 4 and 2*sqrt(3)
        # Let's just pick 4 that are far apart
        second_ring = []
        for k in range(12):
            ang = k * math.pi / 6
            # Alternate between distance 4 (aligned with first ring) and 2*sqrt(3)
            if k % 2 == 0:
                r = 4.0
            else:
                r = 2 * math.sqrt(3)
            second_ring.append((r * math.cos(ang), r * math.sin(ang)))
        # Pick 4 evenly spaced from second ring
        for k in range(0, 12, 3):  # 0, 3, 6, 9
            positions.append(second_ring[k])
        
        centers = np.array(positions[:N])
        
    elif pattern == 1:
        # Grid pattern: 3x3 + 2 extra
        # Hexagonal grid offset
        for i in range(3):
            for j in range(3):
                x = j * 3.5  # horizontal spacing
                y = i * 2 * math.sqrt(3)  # vertical spacing
                if j % 2 == 1:
                    y += math.sqrt(3)  # offset odd rows
                centers[i * 3 + j] = (x - 3.5, y - math.sqrt(3))
        # Add 2 more
        centers[9] = (0, 0)
        centers[10] = (3.5, 2 * math.sqrt(3) + math.sqrt(3))
        
    else:
        # Random but with some structure
        centers = rng.uniform(-3, 3, size=(N, 2))
    
    theta = rng.uniform(-math.pi / 6, math.pi / 6)
    params = np.concatenate([centers.reshape(-1), [theta]])
    return params


def _optimize_one(init_params, theta_outer, maxiter):
    from scipy.optimize import minimize
    res = minimize(_objective, init_params, args=(theta_outer,),
                   method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12})
    return res


def _lp_refine(params, theta_outer):
    """
    Given inner hexagon positions (centers, theta), find the optimal outer hexagon
    (center + rotation) via LP search. Returns (side_length, outer_center, outer_rotation).
    """
    centers = params[:N * 2].reshape(N, 2)
    theta = params[N * 2]
    
    verts_local = _hex_vertices_local(theta)
    verts = (centers[:, None, :] + verts_local[None, :, :]).reshape(-1, 2)
    
    s, c, rot = _min_enclosing_hex_lp(verts)
    return s, c, rot


def _polish(params, theta_outer, maxiter):
    """High-precision polish pass with tighter tolerances"""
    from scipy.optimize import minimize
    res = minimize(_objective, params, args=(theta_outer,),
                   method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-14})
    return res


def construct():
    global T0
    from scipy.optimize import minimize

    best = None
    best_score = float('inf')
    best_outer_center = None
    best_outer_rot = None

    seed = 0
    maxiter = 400

    while time.time() - T0 < DEADLINE - 15:
        init_params = _make_init(seed)
        seed += 1
        try:
            res = _optimize_one(init_params, 0.0, maxiter)
        except Exception:
            continue
        params = res.x
        centers = params[:N * 2].reshape(N, 2)
        theta = params[N * 2]
        
        # Check non-overlap
        diff = centers[:, None, :] - centers[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        if d[iu].min() < 2.0 - 1e-5:
            continue
        
        # Use LP to find optimal outer hexagon
        if time.time() - T0 < DEADLINE - 10:
            try:
                s_lp, c_lp, rot_lp = _lp_refine(params, 0.0)
                if s_lp < best_score:
                    best_score = s_lp
                    best = (centers.copy(), theta, params.copy())
                    best_outer_center = c_lp.copy()
                    best_outer_rot = rot_lp
            except Exception:
                # Fallback to fixed-origin outer
                s = _outer_side(theta, centers, 0.0)
                if s < best_score:
                    best_score = s
                    best = (centers.copy(), theta, params.copy())
                    best_outer_center = np.array([0.0, 0.0])
                    best_outer_rot = 0.0
        else:
            s = _outer_side(theta, centers, 0.0)
            if s < best_score:
                best_score = s
                best = (centers.copy(), theta, params.copy())
                best_outer_center = np.array([0.0, 0.0])
                best_outer_rot = 0.0

        # Polish pass if we have time
        if time.time() - T0 < DEADLINE - 15 and best is not None and best[2] is not None:
            try:
                res_polish = _polish(best[2], 0.0, 150)
                params_p = res_polish.x
                centers_p = params_p[:N * 2].reshape(N, 2)
                theta_p = params_p[N * 2]
                # Check non-overlap
                diff_p = centers_p[:, None, :] - centers_p[None, :, :]
                d_p = np.linalg.norm(diff_p, axis=2)
                if d_p[iu].min() >= 2.0 - 1e-5:
                    s_lp_p, c_lp_p, rot_lp_p = _lp_refine(params_p, 0.0)
                    if s_lp_p < best_score:
                        best_score = s_lp_p
                        best = (centers_p.copy(), theta_p, params_p.copy())
                        best_outer_center = c_lp_p.copy()
                        best_outer_rot = rot_lp_p
            except Exception:
                pass

    if best is None:
        # fallback: return something (may be invalid -> score 0)
        centers = np.zeros((N, 2))
        return (list(centers[:, 0].tolist()), list(centers[:, 1].tolist())), (0.0, 0.0), 40.0, 0.0

    centers, theta, _ = best
    if best_outer_center is None:
        best_outer_center = np.array([0.0, 0.0])
    if best_outer_rot is None:
        best_outer_rot = 0.0
    
    # Recompute side length for the final configuration
    verts_local = _hex_vertices_local(theta)
    verts = (centers[:, None, :] + verts_local[None, :, :]).reshape(-1, 2)
    normals = _outer_side_normals(best_outer_rot)
    rel_verts = verts - best_outer_center
    proj = rel_verts @ normals.T
    apothem = proj.max()
    s = apothem * 2.0 / math.sqrt(3.0)
    
    inner = [[float(centers[i, 0]), float(centers[i, 1]), float(math.degrees(theta))] for i in range(N)]
    return inner, (float(best_outer_center[0]), float(best_outer_center[1])), float(s), float(math.degrees(best_outer_rot))
