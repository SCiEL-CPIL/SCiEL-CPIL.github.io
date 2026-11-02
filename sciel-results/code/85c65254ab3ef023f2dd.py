import math
import time
import random
import numpy as np

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 6.0

def _hex_normals(angle_deg):
    a = math.radians(angle_deg)
    return np.array([[math.cos(a + math.pi/6 + k*math.pi/3),
                      math.sin(a + math.pi/6 + k*math.pi/3)] for k in range(6)])

def _min_separation(d, normals):
    # distance between two unit hexagons with the same orientation = min_k |d.k n_k| - apothem*2
    # (Minkowski sum is hexagon side 2, apothem sqrt(3) = 2*APOTHEM)
    projs = np.abs(d @ normals.T)
    return np.min(projs) - 2*APOTHEM

def _min_separation_and_dir(d, normals):
    projs = np.abs(d @ normals.T)
    idx = int(np.argmin(projs))
    n = normals[idx]
    # direction from j to i that increases separation is sign(d.n)*n
    sign = 1.0 if d @ n >= 0 else -1.0
    sep = projs[idx] - 2*APOTHEM
    return sep, sign * n

def _enclosing(pts, center, angle_deg):
    normals = _hex_normals(angle_deg)
    rel = pts - np.array(center)
    projs = rel @ normals.T  # (N,6)
    D = np.max(np.abs(projs))
    return D, int(np.unravel_index(np.argmax(np.abs(projs)), projs.shape)), int(np.argmax(np.abs(projs)))

def _S_from_D(D):
    return (D + APOTHEM) / APOTHEM

def _fits(pts, center, angle_deg):
    D, _, _ = _enclosing(pts, center, angle_deg)
    return D <= APOTHEM  # S <= 1 means fits in unit hexagon

def _valid_no_overlap(pts, angle_deg):
    normals = _hex_normals(angle_deg)
    n = len(pts)
    for i in range(n):
        for j in range(i+1, n):
            d = pts[i] - pts[j]
            if _min_separation(d, normals) < -1e-9:
                return False
    return True

def _anneal(start_pts, start_center, start_angle, time_budget, rng):
    pts = start_pts.copy()
    center = np.array(start_center, dtype=float)
    angle = start_angle

    # Precompute normals
    normals = _hex_normals(angle)
    n = len(pts)

    # Overlap fix helper
    def fix_overlaps(pts, normals, iterations=200):
        for _ in range(iterations):
            moved = False
            for i in range(n):
                for j in range(i+1, n):
                    d = pts[i] - pts[j]
                    sep, dirn = _min_separation_and_dir(d, normals)
                    if sep < -1e-12:
                        # push apart
                        push = -0.5 * sep
                        pts[i] += dirn * push
                        pts[j] -= dirn * push
                        moved = True
            if not moved:
                break
        return pts

    pts = fix_overlaps(pts, normals)

    # Main annealing loop
    step = 0.15
    t = 0.0
    dt = 0.02
    best_S = _S_from_D(_enclosing(pts, center, angle)[0])
    best_state = (pts.copy(), center.copy(), angle)

    while time.time() - START_TIME < time_budget:
        # Recompute normals for current angle
        normals = _hex_normals(angle)

        # 1) Fix overlaps
        pts = fix_overlaps(pts, normals, iterations=100)

        # 2) Compute enclosing
        D, pi, ki = _enclosing(pts, center, angle)
        S = _S_from_D(D)
        if S < best_S - 1e-12 and _valid_no_overlap(pts, angle):
            best_S = S
            best_state = (pts.copy(), center.copy(), angle)

        # 3) Move center toward the extreme point to reduce D
        normal_k = normals[ki]
        rel = pts[pi] - center
        # If rel . normal_k > 0, the extreme is on + side; move center toward it
        sign = 1.0 if rel @ normal_k > 0 else -1.0
        center += normal_k * sign * step * 0.3

        # 4) Nudge the extreme hexagon inward (opposite to its extreme direction)
        pts[pi] -= normal_k * sign * step * 0.5

        # 5) Random perturbation (annealing)
        temp = step * (0.3 + 0.7 * (t / max(1, t + 0.01)))
        for i in range(n):
            if rng.random() < 0.3:
                pts[i] += rng.gauss(0, temp) * np.array([1, 1])

        # 6) Occasionally sweep angle
        if step % 1 == 0 or rng.random() < 0.05:
            angle = (angle + rng.uniform(-0.05, 0.05)) % (math.pi / 6)
            normals = _hex_normals(angle)

        # 7) Re-fix overlaps after perturbation
        pts = fix_overlaps(pts, normals, iterations=50)

        # 8) Decay step
        step *= 0.995
        t += dt

    # Final polish: exact enclosing
    D, _, _ = _enclosing(pts, center, angle)
    if _valid_no_overlap(pts, angle) and _S_from_D(D) < best_S:
        best_S = _S_from_D(D)
        best_state = (pts.copy(), center.copy(), angle)

    return best_S, best_state[0], best_state[1], best_state[2]

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    rng = random.Random(42)
    np_rng = np.random.default_rng(42)

    best_S = float('inf')
    best_pts = None
    best_center = None
    best_angle = 0.0

    # Generate multiple random initial configurations
    n_restarts = 30
    per_restart_time = (TIME_LIMIT - SAFETY_MARGIN) / n_restarts

    for r in range(n_restarts):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        remaining = TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)
        per_restart = remaining / max(1, n_restarts - r)

        # Random initial positions: cluster around origin with some spread
        # Use a hexagonal grid as a loose initial guess, then jitter
        init_pts = []
        # Place N hexagons in a rough hexagonal cluster
        axial = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
                 (2,0), (-2,0), (0,2), (0,-2)]
        for q, rr in axial:
            x = math.sqrt(3) * (q + rr / 2.0)
            y = 1.5 * rr
            init_pts.append([x, y])
        init_pts = np.array(init_pts, dtype=float)

        # Jitter to break symmetry
        jitter = np_rng.normal(0, 0.3, size=init_pts.shape)
        init_pts += jitter

        # Random center and angle
        init_center = np_rng.normal(0, 0.5, size=2)
        init_angle = rng.uniform(0, math.pi / 6)

        S, pts, center, angle = _anneal(init_pts, init_center, init_angle,
                                         time.time() + per_restart, rng)

        if S < best_S:
            best_S = S
            best_pts = pts
            best_center = center
            best_angle = angle

    # If no valid result, fall back to a simple grid
    if best_pts is None or not _valid_no_overlap(best_pts, best_angle):
        axial = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
                 (2,0), (-2,0), (0,2), (0,-2)]
        best_pts = np.array([[math.sqrt(3)*(q + rr/2.0), 1.5*rr] for q, rr in axial])
        best_center = np.array([0.0, 0.0])
        best_angle = 0.0
        D, _, _ = _enclosing(best_pts, best_center, best_angle)
        best_S = _S_from_D(D)

    # Final safety: ensure no overlap by slight scaling if needed
    if not _valid_no_overlap(best_pts, best_angle):
        # Scale down positions relative to center
        for scale in [0.99, 0.98, 0.97, 0.95]:
            scaled_pts = best_center + (best_pts - best_center) * scale
            if _valid_no_overlap(scaled_pts, best_angle):
                best_pts = scaled_pts
                break

    # Compute final enclosing
    D, _, _ = _enclosing(best_pts, best_center, best_angle)
    final_S = _S_from_D(D)

    inner = []
    angle_deg = math.degrees(best_angle)
    for x, y in best_pts:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(best_center[0]), float(best_center[1])]
    side = float(final_S)
    rotation = float(angle_deg)

    return inner, center, side, rotation
