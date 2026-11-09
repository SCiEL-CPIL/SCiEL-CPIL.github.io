"""Combined hexagon packing: lattice init + force-directed + coordinate descent + Nelder-Mead polish."""
import numpy as np
import math
import time

N = 11
_START = time.time()
_DEADLINE = _START + 135.0  # safety margin before 150 s limit

sqrt3 = math.sqrt(3)

# Precomputed unit hexagon vertices (pointy-top, circumradius=1)
_BASE_VERTS = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
# Edge normals for enclosing hexagon width (3 unique directions for a regular hexagon)
_NORM_DIRS = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in (30.0, 90.0, 150.0)])


def _rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _hex_verts(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    theta = math.radians(angle_deg)
    R = _rot(theta)
    return np.array([cx, cy]) + side * _BASE_VERTS @ R.T


def _enclosing_side(verts_flat):
    """Compute minimum enclosing regular hexagon side length for a set of points."""
    # Project onto 3 normal directions; width in each direction
    widths = []
    for n in _NORM_DIRS:
        proj = verts_flat @ n
        widths.append(proj.max() - proj.min())
    return max(widths) / sqrt3


def _edge_min_dist(V, W):
    """Minimum distance between boundaries of two hexagons (0 if overlapping/touching)."""
    best = 1e18
    for i in range(6):
        a0, a1 = V[i], V[(i + 1) % 6]
        for j in range(6):
            b0, b1 = W[j], W[(j + 1) % 6]
            # Segment-segment distance
            d1 = a1 - a0
            d2 = b1 - b0
            r = a0 - b0
            a = float(d1 @ d1)
            e = float(d2 @ d2)
            f = float(d2 @ r)
            EPS = 1e-12
            if a <= EPS and e <= EPS:
                d2v = float(r @ r)
            elif a <= EPS:
                t = f / e
                t = 0.0 if t < 0 else (1.0 if t > 1 else t)
                d2v = float((b0 + t * d2 - a0) @ (b0 + t * d2 - a0))
            elif e <= EPS:
                s = float(d1 @ r) / a
                s = 0.0 if s < 0 else (1.0 if s > 1 else s)
                d2v = float((a0 + s * d1 - b0) @ (a0 + s * d1 - b0))
            else:
                b = float(d1 @ d2)
                denom = a * e - b * b
                s = (b * f - e * float(d1 @ r)) / denom if abs(denom) > EPS else 0.0
                s = 0.0 if s < 0 else (1.0 if s > 1 else s)
                t = (b * s + f) / e
                t = 0.0 if t < 0 else (1.0 if t > 1 else t)
                d2v = float((a0 + s * d1 - (b0 + t * d2)) @ (a0 + s * d1 - (b0 + t * d2)))
            if d2v < best:
                best = d2v
                if best < 1e-12:
                    return 0.0
    return math.sqrt(best)


def _eval(params):
    """
    params: array of length 3*N + 1
      params[3*i]   = x_i
      params[3*i+1] = y_i
      params[3*i+2] = angle_i (radians, local)
      params[3*N]   = global rotation (radians)
    Returns (side, penalty) where penalty > 0 means collisions.
    """
    n = N
    theta_g = params[3 * N]
    R_g = _rot(theta_g)
    verts = np.empty((n, 6, 2))
    for i in range(n):
        cx, cy = params[3 * i], params[3 * i + 1]
        ang = params[3 * i + 2] + theta_g
        center = np.array([cx, cy])
        verts[i] = center + _BASE_VERTS @ _rot(ang).T
    all_verts = verts.reshape(-1, 2)
    side = _enclosing_side(all_verts)
    # Collision penalty
    pen = 0.0
    margin = 0.0005
    for i in range(n):
        for j in range(i + 1, j + 1):
            pass
    for i in range(n):
        for j in range(i + 1, n):
            d = _edge_min_dist(verts[i], verts[j])
            if d < margin:
                pen += (margin - d) ** 2
    return side, pen


def _hex_ring_positions(ring):
    if ring == 0:
        return [(0, 0)]
    positions = []
    q, r = ring, 0
    for side_idx in range(6):
        for step in range(ring):
            positions.append((q, r))
            dq = [1, 0, -1, -1, 0, 1][side_idx]
            dr = [0, -1, -1, 0, 1, 1][side_idx]
            q += dq
            r += dr
    return positions


def _axial_to_cartesian(q, r):
    x = 2.0 * q + r
    y = sqrt3 * r
    return x, y


def _make_lattice_params(rng):
    """Generate initial params from hexagonal lattice with random global rotation."""
    all_positions = [(0, 0)]
    ring = 1
    while len(all_positions) < N:
        all_positions.extend(_hex_ring_positions(ring))
        ring += 1
    selected = all_positions[:N]
    
    theta_g = float(rng.uniform(0.0, math.pi / 3.0))
    R_g = _rot(theta_g)
    
    centers = []
    for q, r in selected:
        x, y = _axial_to_cartesian(q, r)
        centers.append(np.array([x, y]))
    
    centers = np.array(centers)
    # Center the configuration
    center_mean = centers.mean(axis=0)
    centers = centers - center_mean
    # Apply global rotation
    centers = centers @ R_g.T
    
    params = np.empty(3 * N + 1)
    for i in range(N):
        params[3 * i] = centers[i][0]
        params[3 * i + 1] = centers[i][1]
        params[3 * i + 2] = 0.0  # local angle
    params[3 * N] = theta_g
    # Small random perturbation
    params[:3 * N] += rng.normal(0.0, 0.01, size=3 * N)
    return params


def _force_directed(params, max_iters=2000, deadline=_DEADLINE):
    """Force-directed relaxation to resolve collisions and compact."""
    n = N
    pos = np.array([[params[3 * i], params[3 * i + 1]] for i in range(n)], dtype=float)
    angles = np.array([params[3 * i + 2] for i in range(n)], dtype=float)
    theta_g = params[3 * N]
    
    lr = 0.06
    decay = 0.9997
    for it in range(max_iters):
        if time.time() > deadline - 15:
            break
        forces = np.zeros_like(pos)
        for i in range(n):
            for j in range(i + 1, n):
                vi = pos[i] + _BASE_VERTS @ _rot(angles[i] + theta_g).T
                vj = pos[j] + _BASE_VERTS @ _rot(angles[j] + theta_g).T
                d = _edge_min_dist(vi, vj)
                if d < 0.01:
                    diff = pos[i] - pos[j]
                    dist = float(np.linalg.norm(diff))
                    if dist < 1e-6:
                        diff = np.array([0.01, 0.0])
                        dist = 0.01
                    overlap = 0.01 - d
                    forces[i] += (overlap * 3.0) * diff / dist
                    forces[j] -= (overlap * 3.0) * diff / dist
        # Inward attraction
        for i in range(n):
            to_center = -pos[i]
            d = float(np.linalg.norm(to_center))
            if d > 1e-6:
                forces[i] += 0.12 * to_center / d
        pos += lr * forces
        lr *= decay
        if np.max(np.abs(lr * forces)) < 1e-5:
            break
    
    # Recenter
    mean = pos.mean(axis=0)
    pos = pos - mean
    
    new_params = params.copy()
    for i in range(n):
        new_params[3 * i] = pos[i][0]
        new_params[3 * i + 1] = pos[i][1]
    return new_params


def _coordinate_descent(params, max_iters=8000, deadline=_DEADLINE):
    """Coordinate descent with random coordinate selection."""
    rng = np.random.default_rng(42)
    cur_side, cur_pen = _eval(params)
    cur_cost = cur_side + 50.0 * cur_pen
    best_side, best_pen = cur_side, cur_pen
    best_params = params.copy()
    step = 0.12
    no_improve = 0
    
    for it in range(max_iters):
        if time.time() > deadline - 10:
            break
        idx = int(rng.integers(0, params.shape[0]))
        old = params[idx]
        params[idx] = old + float(rng.normal(0.0, step))
        if idx == 3 * N:
            params[idx] = params[idx] % (math.pi / 3.0)
        
        ns, npen = _eval(params)
        ncost = ns + 50.0 * npen
        if ncost < cur_cost:
            cur_cost, cur_side, cur_pen = ncost, ns, npen
            no_improve = 0
            step = min(0.4, step * 1.03)
            if npen <= 1e-6 and ns < best_side:
                best_side, best_pen = ns, npen
                best_params = params.copy()
        else:
            params[idx] = old
            no_improve += 1
            step = max(0.008, step * 0.95)
        
        if no_improve > 500:
            break
    
    return best_params, best_side, best_pen


def _nelder_mead(params, max_iters=800, deadline=_DEADLINE):
    """Nelder-Mead on the 3N+1 dimensional parameter space."""
    n_dim = 3 * N + 1
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    
    simplex = [params.copy()]
    f0, p0 = _eval(params)
    vals = [f0 + 50.0 * p0]
    
    # Build initial simplex
    for d in range(n_dim):
        v = params.copy()
        v[d] += 0.03
        simplex.append(v)
        fs, ps = _eval(v)
        vals.append(fs + 50.0 * ps)
    
    for it in range(max_iters):
        if time.time() > deadline - 8:
            break
        order = sorted(range(len(simplex)), key=lambda i: vals[i])
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        
        if abs(vals[0] - vals[-1]) < 1e-7:
            break
        
        # Centroid of all but worst
        centroid = np.zeros(n_dim)
        for i in range(len(simplex) - 1):
            centroid += simplex[i]
        centroid /= (len(simplex) - 1)
        
        worst = simplex[-1]
        xr = centroid + alpha * (centroid - worst)
        if xr[3 * N] < 0:
            xr[3 * N] %= (math.pi / 3.0)
        fr, pr = _eval(xr)
        fr_cost = fr + 50.0 * pr
        
        if fr_cost < vals[0]:
            xe = centroid + gamma * (centroid - worst)
            if xe[3 * N] < 0:
                xe[3 * N] %= (math.pi / 3.0)
            fe, pe = _eval(xe)
            fe_cost = fe + 50.0 * pe
            if fe_cost < fr_cost:
                simplex[-1], vals[-1] = xe, fe_cost
            else:
                simplex[-1], vals[-1] = xr, fr_cost
        elif fr_cost < vals[-2]:
            simplex[-1], vals[-1] = xr, fr_cost
        else:
            xc = centroid + rho * (worst - centroid)
            if xc[3 * N] < 0:
                xc[3 * N] %= (math.pi / 3.0)
            fc, pc = _eval(xc)
            fc_cost = fc + 50.0 * pc
            if fc_cost < vals[-1]:
                simplex[-1], vals[-1] = xc, fc_cost
            else:
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    if simplex[i][3 * N] < 0:
                        simplex[i][3 * N] %= (math.pi / 3.0)
                    fi, pi_ = _eval(simplex[i])
                    vals[i] = fi + 50.0 * pi_
    
    best_i = min(range(len(simplex)), key=lambda i: vals[i])
    return simplex[best_i], vals[best_i]


def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    rng = np.random.default_rng(12345)
    best_side = float('inf')
    best_params = None
    
    def consider(params):
        nonlocal best_side, best_params
        side, pen = _eval(params)
        if pen <= 1e-5 and side < best_side:
            best_side = side
            best_params = params.copy()
    
    # Stage 1: Lattice initialization + force-directed relaxation
    for trial in range(3):
        if time.time() > deadline - 50:
            break
        params = _make_lattice_params(rng)
        params = _force_directed(params, max_iters=1500, deadline=deadline)
        consider(params)
    
    # Stage 2: Coordinate descent refinement
    if time.time() < deadline - 40:
        base = best_params if best_params is not None else _make_lattice_params(rng)
        refined, _, _ = _coordinate_descent(base.copy(), max_iters=5000, deadline=deadline)
        consider(refined)
    
    # Stage 3: Nelder-Mead polish
    if time.time() < deadline - 20:
        base = best_params if best_params is not None else _make_lattice_params(rng)
        polished, _ = _nelder_mead(base.copy(), max_iters=600, deadline=deadline)
        consider(polished)
    
    # Stage 4: Final coordinate descent on best
    if time.time() < deadline - 10 and best_params is not None:
        final, _, _ = _coordinate_descent(best_params.copy(), max_iters=3000, deadline=deadline)
        consider(final)
    
    # Build result
    if best_params is None:
        # Fallback: spread out
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    theta_g = float(best_params[3 * N]) % (math.pi / 3.0)
    inner = []
    for i in range(N):
        cx = float(best_params[3 * i])
        cy = float(best_params[3 * i + 1])
        ang_deg = math.degrees(float(best_params[3 * i + 2]) + theta_g) % 360.0
        inner.append([cx, cy, ang_deg])
    
    return inner, (0.0, 0.0), float(best_side), float(math.degrees(theta_g)) % 360.0
