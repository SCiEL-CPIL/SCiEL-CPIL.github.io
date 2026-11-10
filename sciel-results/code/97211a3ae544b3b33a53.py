# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _outer_side_length_from_positions(positions, angles_deg):
    n = len(positions)
    normal_angles = np.radians([30, 90, 150, 210, 270, 330])
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    apothem_factor = np.sqrt(3) / 2
    max_proj = 0.0
    for i in range(n):
        verts = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles_deg[i])
        projs = verts @ normals.T
        max_proj = max(max_proj, float(np.max(projs)))
    s = max_proj / apothem_factor
    return max(s, np.sqrt(12) - 1e-12)

def _sat_penetration(verts_i, verts_j):
    """Smooth (soft) penetration depth between two convex hexagons via SAT.
    Returns (penetration, axis_vector) where axis points from i toward j."""
    best = 0.0
    best_axis = np.zeros(2)
    for verts in (verts_i, verts_j):
        for k in range(6):
            e = verts[(k + 1) % 6] - verts[k]
            nrm = np.array([-e[1], e[0]])
            ln = np.linalg.norm(nrm)
            if ln < 1e-12:
                continue
            nrm = nrm / ln
            pi = verts_i @ nrm
            pj = verts_j @ nrm
            mi, xi = np.max(pi), np.min(pi)
            mj, xj = np.max(pj), np.min(pj)
            # signed gap (positive = separated)
            gap = min(xi, xj) - max(mi, mj)
            pen = -gap
            if pen > best + 1e-9:
                best = pen
                # axis from i to j
                c = 0.5 * (xj + xi) - 0.5 * (mi + mj)
                best_axis = c * nrm
    # smooth the max with a small softplus so gradient exists at contact
    pen = np.logaddexp(0.0, best / 1e-3) * 1e-3
    return pen, best_axis

def _smooth_energy(params, wall=3.0):
    """Total energy: sum of squared penetration depths (non-overlap) plus a soft
    outer-wall confinement term. Returns (energy, gradient) w.r.t. params (36)."""
    n = 12
    pos = np.column_stack([params[0::3], params[1::3]])
    ang = params[2::3]
    normal_angles = np.radians([30, 90, 150, 210, 270, 330])
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    apothem_factor = np.sqrt(3) / 2

    verts = np.array([_hex_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i]) for i in range(n)])
    grad = np.zeros(36)

    # --- non-overlap: pairwise penetration, smooth energy 0.5*pen^2 ---
    for i in range(n):
        for j in range(i + 1, n):
            pen, axis = _sat_penetration(verts[i], verts[j])
            if pen <= 1e-9:
                continue
            # d(0.5 pen^2)/dpos_i = -pen * axis ; /dpos_j = +pen * axis
            grad[3 * i]     -= pen * axis[0]
            grad[3 * i + 1] -= pen * axis[1]
            grad[3 * j]     += pen * axis[0]
            grad[3 * j + 1] += pen * axis[1]
            # angle gradient: vertex_k of i moves by r_i * (-sin, cos) w.r.t. angle
            # dpen/dangle_i = pen * ( (dvertex_i/dang_i) . axis )
            ai = np.radians(ang[i])
            for k in range(6):
                t = ai + k * np.pi / 3
                dvi = np.array([-np.sin(t), np.cos(t)])
                t2 = ai + 0 * 1  # placeholder
                # vertex position = pos_i + (cos t, sin t); dvertex/dangle = (-sin t, cos t)
                grad[3 * i + 2] += pen * (dvi @ axis)
            aj = np.radians(ang[j])
            for k in range(6):
                t = aj + k * np.pi / 3
                dvj = np.array([-np.sin(t), np.cos(t)])
                grad[3 * j + 2] -= pen * (dvj @ axis)

    # --- outer wall: soft confinement, energy = sum softplus(proj - apothem) ---
    for i in range(n):
        a = np.radians(ang[i])
        for m in range(6):
            nv = normals[m]
            # max projection of hex i on normal m
            # proj_k = (pos_i + (cos(a+k*pi/3), sin(...))) . nv
            thetas = a + np.arange(6) * np.pi / 3
            offs = np.column_stack([np.cos(thetas), np.sin(thetas)])
            projs = pos[i] @ nv + offs @ nv
            kmax = int(np.argmax(projs))
            pmax = projs[kmax]
            apothem = apothem_factor * wall
            over = pmax - apothem
            if over > -1e-3:
                soft = np.logaddexp(0.0, over / 1e-3) * 1e-3
                # dsoft/dover = sigmoid(over/1e-3)
                sig = 1.0 / (1.0 + np.exp(-over / 1e-3))
                dpos = sig * nv
                grad[3 * i]     += dpos[0]
                grad[3 * i + 1] += dpos[1]
                t = thetas[kmax]
                dvi = np.array([-np.sin(t), np.cos(t)])
                grad[3 * i + 2] += sig * (dvi @ nv)

    energy = float(np.sum(grad**2) * 0.0)  # energy value computed separately below
    # recompute a scalar energy for the optimizer
    e = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            pen, _ = _sat_penetration(verts[i], verts[j])
            e += 0.5 * pen * pen
    for i in range(n):
        a = np.radians(ang[i])
        thetas = a + np.arange(6) * np.pi / 3
        offs = np.column_stack([np.cos(thetas), np.sin(thetas)])
        for m in range(6):
            nv = normals[m]
            projs = pos[i] @ nv + offs @ nv
            pmax = np.max(projs)
            over = pmax - apothem_factor * wall
            if over > -1e-3:
                e += np.logaddexp(0.0, over / 1e-3) * 1e-3
    return e, grad

def _check_validity(inner_hex_data, outer_hex_data, s, tol=1e-6):
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    outer_verts = _hex_vertices(outer_hex_data[0], outer_hex_data[1], s, outer_hex_data[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    for i in range(12):
        verts = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        for k in range(6):
            if abs(np.linalg.norm(verts[(k + 1) % 6] - verts[k]) - 1.0) > 1e-8:
                return False
    for i in range(12):
        for j in range(i + 1, 12):
            verts_i = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
            verts_j = _hex_vertices(inner_hex_data[j, 0], inner_hex_data[j, 1], 1.0, inner_hex_data[j, 2])
            separated = False
            for verts in (verts_i, verts_j):
                for k in range(6):
                    edge = verts[(k + 1) % 6] - verts[k]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm > 1e-12:
                        normal = normal / norm
                        proj_i = verts_i @ normal
                        proj_j = verts_j @ normal
                        if np.max(proj_i) + tol < np.min(proj_j) or np.max(proj_j) + tol < np.min(proj_i):
                            separated = True
                            break
                if separated:
                    break
            if not separated:
                return False
    for i in range(12):
        verts = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        for v in verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -tol:
                    return False
    return True

def _exact_objective(params):
    pos = np.column_stack([params[0::3], params[1::3]])
    ang = params[2::3]
    return _outer_side_length_from_positions(pos, ang)

def _make_config(r, seed):
    """Build a diverse starting config."""
    np.random.seed(seed)
    if r % 5 == 0:
        # honeycomb ring + center
        pos = np.array([[0, 0], [3, 0], [1.5, np.sqrt(3)], [-1.5, np.sqrt(3)],
                        [-3, 0], [-1.5, -np.sqrt(3)], [1.5, -np.sqrt(3)],
                        [4.5, np.sqrt(3)], [3, 2 * np.sqrt(3)], [0, 2 * np.sqrt(3)],
                        [-3, 2 * np.sqrt(3)], [-4.5, np.sqrt(3)]])
        ang = np.zeros(12)
    elif r % 5 == 1:
        # compact hexagon of 12 (center + 6 + 5 of second ring)
        pos = np.array([[0, 0], [3, 0], [1.5, np.sqrt(3)], [-1.5, np.sqrt(3)],
                        [-3, 0], [-1.5, -np.sqrt(3)], [1.5, -np.sqrt(3)],
                        [4.5, np.sqrt(3)], [4.5, -np.sqrt(3)], [3, 2 * np.sqrt(3)],
                        [0, 2 * np.sqrt(3)], [-3, 2 * np.sqrt(3)]])
        ang = np.zeros(12)
    elif r % 5 == 2:
        # two rows
        pos = np.array([[0, 0], [3, 0], [6, 0], [9, 0],
                        [1.5, np.sqrt(3)], [4.5, np.sqrt(3)], [7.5, np.sqrt(3)],
                        [-1.5, -np.sqrt(3)], [1.5, -np.sqrt(3)], [4.5, -np.sqrt(3)],
                        [7.5, -np.sqrt(3)], [10.5, -np.sqrt(3)]])
        ang = np.zeros(12)
    elif r % 5 == 3:
        # random cluster
        pos = np.random.randn(12, 2) * 3.0
        ang = np.random.uniform(0, 60, 12)
    else:
        # ring
        theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        pos = np.column_stack([3.5 * np.cos(theta), 3.5 * np.sin(theta)])
        ang = np.degrees(theta)
    # small random perturbation
    pos = pos + np.random.randn(12, 2) * 0.05
    p = np.zeros(36)
    for i in range(12):
        p[3 * i] = pos[i, 0]
        p[3 * i + 1] = pos[i, 1]
        p[3 * i + 2] = ang[i]
    return p

def hexagon_packing_12():
    start = time.time()
    deadline = 600.0
    best_params = None
    best_s = float('inf')

    def consider(p):
        nonlocal best_params, best_s
        s = _outer_side_length_from_positions(
            np.column_stack([p[0::3], p[1::3]]), p[2::3])
        if s < best_s:
            best_s = s
            best_params = p.copy()

    n_starts = 40
    for r in range(n_starts):
        if time.time() > deadline - 120:
            break
        p = _make_config(r, 1000 + r)
        consider(p)
        # smooth energy minimization to drive non-overlap + compactness
        if time.time() < deadline - 100:
            try:
                res = minimize(_smooth_energy, p, method='L-BFGS-B',
                               options={'maxiter': 400, 'gtol': 1e-8, 'ftol': 1e-12})
                if res.x is not None:
                    consider(res.x)
            except Exception:
                pass
        # exact polish on the true objective
        if time.time() < deadline - 60:
            try:
                res2 = minimize(_exact_objective, p, method='Powell',
                                options={'maxiter': 800, 'xtol': 1e-11, 'ftol': 1e-13})
                if res2.x is not None:
                    consider(res2.x)
            except Exception:
                pass

    if best_params is None:
        best_params = _make_config(0, 0)
        best_s = _outer_side_length_from_positions(
            np.column_stack([best_params[0::3], best_params[1::3]]), best_params[2::3])

    # final high-precision exact polish
    if time.time() < deadline - 20:
        try:
            resf = minimize(_exact_objective, best_params, method='Nelder-Mead',
                            options={'maxiter': 6000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if resf.x is not None:
                consider(resf.x)
        except Exception:
            pass

    inner = np.zeros((12, 3))
    for i in range(12):
        inner[i, 0] = best_params[3 * i]
        inner[i, 1] = best_params[3 * i + 1]
        inner[i, 2] = best_params[3 * i + 2]
    outer = np.array([0.0, 0.0, 0.0])
    s = _outer_side_length_from_positions(
        np.column_stack([best_params[0::3], best_params[1::3]]), best_params[2::3])

    if not _check_validity(inner, outer, s):
        inner = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0], [-1.25, -2.17, 0],
            [1.25, -2.17, 0], [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0], [0, -4, 0]
        ])
        outer = np.array([0, 0, 0])
        s = 8.0
    return inner, outer, s

# EVOLVE-BLOCK-END
