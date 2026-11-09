"""Hexagon packing: place 11 unit regular hexagons to minimize enclosing hexagon side."""
import math
import time
import numpy as np

N = 11
_START = time.time()
_DEADLINE = _START + 135.0  # safety margin before 150s kill

# Unit regular hexagon vertices (flat-top), circumradius = 1.
_BASE = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

# Three edge-normal directions (30, 90, 150 deg) for the enclosing-hexagon width.
_NORM = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in (30.0, 90.0, 150.0)])


def _rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _hex_vertices(cx, cy, angle):
    """Return 6 vertices of a unit hexagon centered at (cx, cy) rotated by angle radians."""
    R = _rot(angle)
    return np.array([[cx, cy]] * 6) + _BASE @ R.T


def _sat_collision(V, W):
    """
    Check if two convex hexagons overlap using SAT.
    Returns (penetration_depth, min_axis) or (0, None) if separated.
    penetration_depth > 0 means overlap; the min_axis is the direction of minimum separation.
    """
    min_pen = 1e18
    min_axis = None

    def proj_range(verts, dir_vec):
        dots = verts @ dir_vec
        return dots.min(), dots.max()

    # Get all 12 edge normals (6 from each hexagon)
    for poly in (V, W):
        for i in range(6):
            e = poly[(i + 1) % 6] - poly[i]
            n = np.array([-e[1], e[0]])
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            n = n / n_norm

            r1_min, r1_max = proj_range(V, n)
            r2_min, r2_max = proj_range(W, n)

            overlap = min(r1_max, r2_max) - max(r1_min, r2_min)
            if overlap < 0:
                gap = -overlap
                if gap < min_pen:
                    min_pen = gap
                    min_axis = n
            else:
                if overlap < min_pen:
                    min_pen = overlap
                    min_axis = n

    if min_pen > 0:
        return 0.0, None

    return abs(min_pen), min_axis


def _width(v):
    """Width of a point set along the 3 enclosing-hexagon normal directions."""
    w = []
    for n in _NORM:
        p = np.dot(v, n)
        w.append(p.max() - p.min())
    return w


def _enclosing_side(v):
    return max(_width(v)) / math.sqrt(3.0)


def _eval(params):
    n = N
    verts = np.empty((n, 6, 2))
    for i in range(n):
        cx = params[3 * i]
        cy = params[3 * i + 1]
        angle = params[3 * i + 2]
        verts[i] = _hex_vertices(cx, cy, angle)
    side = _enclosing_side(verts.reshape(-1, 2))
    pen = 0.0
    for i in range(n):
        vi = verts[i]
        for j in range(i + 1, n):
            vj = verts[j]
            depth, _ = _sat_collision(vi, vj)
            if depth > 0:
                pen += depth ** 2
    return side, pen


def _initial_config(seed=7):
    """Generate an initial hexagonal cluster configuration."""
    rng = np.random.default_rng(seed)
    theta_g = float(rng.uniform(0.0, math.pi / 3.0))
    R = _rot(theta_g)
    centers = []
    centers.append(np.array([0.0, 0.0]))
    for k in range(6):
        ang = math.radians(60.0 * k + 30.0)
        centers.append(2.0 * np.array([math.cos(ang), math.sin(ang)]))
    for k in range(4):
        ang = math.radians(90.0 * k)
        centers.append(3.464 * np.array([math.cos(ang), math.sin(ang)]))
    centers = np.array(centers)
    centers = centers @ R.T
    thetas = rng.uniform(-0.05, 0.05, size=N)
    params = np.empty(3 * N + 1)
    for i in range(N):
        params[3 * i] = centers[i][0]
        params[3 * i + 1] = centers[i][1]
        params[3 * i + 2] = thetas[i]
    params[3 * N] = theta_g
    params += rng.normal(0.0, 0.02, size=params.shape)
    return params


def _repair_collisions(params, max_iters=50):
    """Iteratively push colliding hexagons apart until no collisions remain."""
    for _ in range(max_iters):
        if time.time() > _DEADLINE:
            break
        verts = np.empty((N, 6, 2))
        for i in range(N):
            verts[i] = _hex_vertices(params[3 * i], params[3 * i + 1], params[3 * i + 2])
        any_collision = False
        for i in range(N):
            vi = verts[i]
            for j in range(i + 1, N):
                vj = verts[j]
                depth, axis = _sat_collision(vi, vj)
                if depth > 0 and axis is not None:
                    any_collision = True
                    ci = np.array([params[3 * i], params[3 * i + 1]])
                    cj = np.array([params[3 * j], params[3 * j + 1]])
                    d = cj - ci
                    if np.dot(axis, d) < 0:
                        axis = -axis
                    push = depth * 0.5
                    params[3 * i] -= push * axis[0]
                    params[3 * i + 1] -= push * axis[1]
                    params[3 * j] += push * axis[0]
                    params[3 * j + 1] += push * axis[1]
                    # Recompute vertices for both
                    verts[i] = _hex_vertices(params[3 * i], params[3 * i + 1], params[3 * i + 2])
                    verts[j] = _hex_vertices(params[3 * j], params[3 * j + 1], params[3 * j + 2])
        if not any_collision:
            break
    return params


def _pack():
    """Optimize 11 hexagons using simulated annealing with SAT-based collision resolution."""
    best_side = 1e18
    best_pen = 1e18
    best_params = None
    rng = np.random.default_rng(7)

    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() > _DEADLINE:
            break

        params = _initial_config(seed=7 + restart * 13)

        # Repair initial collisions
        params = _repair_collisions(params)

        T = 0.3
        T_min = 0.0005
        cooling = 0.997
        step_base = 0.08

        cur_side, cur_pen = _eval(params)
        cur_cost = cur_side + 200.0 * cur_pen

        if cur_pen <= 1e-6 and cur_side < best_side:
            best_side = cur_side
            best_pen = cur_pen
            best_params = params.copy()

        no_improve = 0
        max_no_improve = 3000

        while time.time() < _DEADLINE and no_improve < max_no_improve:
            idx = int(rng.integers(0, params.shape[0]))
            old = params[idx]

            step = step_base * (0.3 + T)
            delta = float(rng.normal(0.0, step))
            params[idx] = old + delta

            if idx == 3 * N:
                params[idx] = params[idx] % (math.pi / 3.0)

            ns, npen = _eval(params)
            ncost = ns + 200.0 * npen

            if ncost <= cur_cost:
                accept = True
            else:
                delta_cost = ncost - cur_cost
                if T > T_min:
                    accept = (rng.random() < math.exp(-delta_cost / T))
                else:
                    accept = False

            if accept:
                cur_cost, cur_side, cur_pen = ncost, ns, npen
                no_improve = 0
                T = max(T_min, T * cooling)

                if npen <= 1e-6 and ns < best_side:
                    best_side = ns
                    best_pen = npen
                    best_params = params.copy()
            else:
                params[idx] = old
                no_improve += 1
                if no_improve % 300 == 0 and T > T_min:
                    T = max(T_min, T * 1.5)

        # Final polish: fine-tune the best solution
        if best_params is not None and time.time() < _DEADLINE:
            params = best_params.copy()
            step = 0.02
            cur_side, cur_pen = _eval(params)
            cur_cost = cur_side + 200.0 * cur_pen
            no_improve = 0

            while time.time() < _DEADLINE and no_improve < 800:
                # Try moving each coordinate in small increments
                idx = int(rng.integers(0, params.shape[0]))
                old = params[idx]
                params[idx] = old + float(rng.normal(0.0, step))

                if idx == 3 * N:
                    params[idx] = params[idx] % (math.pi / 3.0)

                ns, npen = _eval(params)
                ncost = ns + 200.0 * npen

                if ncost < cur_cost:
                    cur_cost, cur_side, cur_pen = ncost, ns, npen
                    no_improve = 0
                    step = min(0.05, step * 1.03)
                else:
                    params[idx] = old
                    no_improve += 1
                    step = max(0.002, step * 0.97)

                if npen <= 1e-6 and ns < best_side:
                    best_side = ns
                    best_pen = npen
                    best_params = params.copy()

    return best_params, best_side, best_pen


def construct():
    try:
        params, side, pen = _pack()
        if params is None:
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 40.0, 0.0

        theta_g = float(params[3 * N]) % (math.pi / 3.0)
        inner = []
        for i in range(N):
            cx = float(params[3 * i])
            cy = float(params[3 * i + 1])
            ca = float(params[3 * i + 2])
            ang = (ca + theta_g) % 360.0
            inner.append([cx, cy, ang])
        return inner, (0.0, 0.0), float(side), float(theta_g)
    except Exception:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0


if __name__ == "__main__":
    res = construct()
    print("outer side:", res[2], "rotation:", res[3], "n inner:", len(res[0]))
    print("elapsed:", time.time() - _START)
