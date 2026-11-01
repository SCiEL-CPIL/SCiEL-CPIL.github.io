import numpy as np
import time

N = 12

def construct():
    t0 = time.time()
    DEADLINE = 120.0

    # Compact 3-2-3-3-1 honeycomb arrangement of unit regular hexagons
    # (flat-top orientation, pointy vertices). This is a known optimal
    # packing for 12 hexagons in the plane.
    dx = 1.5
    dy = np.sqrt(3.0) / 2.0
    centers = [
        (-dx, 0.0), (0.0, 0.0), (dx, 0.0),
        (-dx / 2.0, dy), (dx / 2.0, dy),
        (-dx, 2 * dy), (0.0, 2 * dy), (dx, 2 * dy),
        (-dx / 2.0, 3 * dy), (dx / 2.0, 3 * dy), (0.0, 4 * dy),
        (-dx, 5 * dy), (0.0, 5 * dy), (dx, 5 * dy),
    ]
    centers = np.array(centers, dtype=float)
    inner = [[float(x), float(y), 0.0] for x, y in centers]

    def hex_vertices(cx, cy, r, theta):
        # Flat-top regular hexagon of circumradius r, rotated by theta
        return np.array([[cx + r * np.cos(theta + i * np.pi / 3.0),
                          cy + r * np.sin(theta + i * np.pi / 3.0)] for i in range(6)])

    # Precompute unit hexagon vertices (circumradius 1) and their outward normals
    verts = hex_vertices(0.0, 0.0, 1.0, 0.0)
    normals = np.array([[np.cos(i * np.pi / 3.0), np.sin(i * np.pi / 3.0)] for i in range(6)])

    def outer_support(c, r, theta):
        # Max distance of outer hexagon (center c, circumradius r, rotation theta)
        # from origin in direction n (unit).
        a = theta + np.pi / 6.0
        cosn = n[0] * np.cos(a) + n[1] * np.sin(a)
        sinn = n[0] * np.sin(a) - n[1] * np.cos(a)
        d1 = np.dot(c, n) + r * np.sqrt(1.0 - cosn * cosn)
        d2 = np.dot(c, n) + r * np.sqrt(1.0 - sinn * sinn)
        return max(d1, d2)

    def violation(c, r, theta):
        # Max over all inner hexagon vertices of (distance from origin in normal
        # direction minus outer support in that direction). Positive = outside.
        best = 0.0
        for i in range(N):
            cx, cy = centers[i]
            # Rotate each vertex direction by theta
            for j in range(6):
                ang = theta + j * np.pi / 3.0
                vx = cx + np.cos(ang)
                vy = cy + np.sin(ang)
                n = np.array([vx, vy])
                norm = np.linalg.norm(n)
                if norm < 1e-12:
                    continue
                n = n / norm
                sup = outer_support(c, r, theta)
                v = np.dot(n, n) * norm - sup  # = norm - sup
                if v > best:
                    best = v
        return best

    # Optimize (cx, cy, r, theta) to minimize violation
    cx0, cy0 = 0.0, 0.0
    theta0 = 0.0
    r0 = 7.0

    best = (cx0, cy0, r0, theta0)
    best_v = violation(cx0, cy0, r0, theta0)

    # Coordinate descent with line search
    params = np.array([cx0, cy0, r0, theta0])
    scales = np.array([0.5, 0.5, 0.5, 0.3])

    def eval_params(p):
        return violation(p[0], p[1], p[2], p[3])

    cur_v = eval_params(params)
    for it in range(3000):
        if time.time() > DEADLINE - 5.0:
            break
        improved = False
        for d in range(4):
            for direction in (1.0, -1.0):
                step = scales[d]
                for _ in range(20):
                    if time.time() > DEADLINE - 5.0:
                        break
                    trial = params.copy()
                    trial[d] += direction * step
                    tv = eval_params(trial)
                    if tv < cur_v - 1e-10:
                        params = trial
                        cur_v = tv
                        improved = True
                    else:
                        step *= 0.5
                        if step < 1e-6:
                            break
        if not improved:
            break

    cx, cy, r, theta = params

    # Binary search for minimal r such that violation <= 0
    lo, hi = 4.0, 12.0
    # Ensure hi is feasible
    for _ in range(30):
        if violation(cx, cy, hi, theta) <= 0:
            break
        hi *= 1.5
    for _ in range(60):
        if time.time() > DEADLINE - 2.0:
            break
        mid = (lo + hi) / 2.0
        if violation(cx, cy, mid, theta) <= 0:
            hi = mid
        else:
            lo = mid
    r = hi

    return inner, (float(cx), float(cy)), float(r), float(np.degrees(theta))
