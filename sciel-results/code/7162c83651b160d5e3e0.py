"""Optimized hexagon packing: 11 unit hexagons in a regular hexagon.
Strategy: joint continuous optimization of inner positions/rotations and outer
hexagon parameters (center, side, angle) using BFGS with smooth overlap
penalties and a closed-form minimum-enclosing-hexagon sub-solve. Multi-start
from hexagonal-lattice and ring templates with a strict deadline guard.
"""
import numpy as np
import time
from itertools import combinations

N = 11
DEADLINE = 142.0  # safety margin before 150s kill
SQRT3 = np.sqrt(3.0)


def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy)."""
    angle = np.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = angle + np.pi * i / 3.0
        verts[i, 0] = cx + side * np.cos(theta)
        verts[i, 1] = cy + side * np.sin(theta)
    return verts


def _hex_edge_normals(angle_deg):
    """Return 6 outward unit normal vectors of a regular hexagon."""
    angle = np.radians(angle_deg)
    normals = np.empty((6, 2))
    for i in range(6):
        mid_angle = angle + np.pi / 6.0 + np.pi * i / 3.0
        normals[i, 0] = np.cos(mid_angle)
        normals[i, 1] = np.sin(mid_angle)
    return normals


def _min_enclosing_hex(centers, inner_angles, outer_angle_deg):
    """
    For fixed inner hexagon positions/angles and outer orientation, compute the
    minimum enclosing regular hexagon side length and optimal center.

    Convex formulation: for each of the 6 outward normals n_i, the constraint is
        max_p (p - c)·n_i <= s * sqrt(3)/2
    =>  s >= (A_i - c·n_i) / (sqrt(3)/2),  where A_i = max_p p·n_i.
    Minimize max_i (A_i - c·n_i) over c (2D convex). The normals come in 3
    opposite pairs, so for i=0,1,2 the binding constraint is
        max(A_i - c·n_i, A_{i+3} + c·n_i)
    which is minimized in the c·n_i direction at (A_{i+3}-A_i)/2.
    """
    normals = _hex_edge_normals(outer_angle_deg)  # 6x2
    all_points = np.empty((6 * len(centers), 2))
    for i in range(len(centers)):
        cx, cy = centers[i]
        all_points[6*i:6*i+6] = _hex_vertices(cx, cy, 1.0, inner_angles[i])

    projections = all_points @ normals.T  # (6N, 6)
    A = projections.max(axis=0)  # (6,)

    # Minimize f(c) = max_i (A_i - c·n_i). Use subgradient / active-set.
    # Initialize c at centroid of extreme points.
    extreme_points = np.empty((6, 2))
    for i in range(6):
        idx = projections[:, i].argmax()
        extreme_points[i] = all_points[idx]
    c = extreme_points.mean(axis=0)

    for _ in range(100):
        vals = A - c @ normals.T  # (6,)
        max_val = vals.max()
        active = vals >= max_val - 1e-10
        if active.sum() <= 1:
            break
        grad = normals[active].mean(axis=0)
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-12:
            break
        # Line search along grad (reduces all active vals simultaneously).
        best_alpha = 0.0
        best_f = max_val
        for alpha in np.linspace(-1.5, 1.5, 31):
            c_try = c + alpha * grad
            f_try = (A - c_try @ normals.T).max()
            if f_try < best_f - 1e-14:
                best_f = f_try
                best_alpha = alpha
        if abs(best_alpha) < 1e-11:
            break
        c = c + best_alpha * grad

    vals = A - c @ normals.T
    max_proj = vals.max()
    side = max_proj / (SQRT3 / 2.0)
    return side, c


def _hex_overlap(c1, a1, c2, a2, tol=1e-9):
    """SAT overlap test for two unit hexagons."""
    v1 = _hex_vertices(c1[0], c1[1], 1.0, a1)
    v2 = _hex_vertices(c2[0], c2[1], 1.0, a2)
    axes = []
    for verts in (v1, v2):
        for i in range(6):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % 6]
            ex = x2 - x1
            ey = y2 - y1
            norm = np.sqrt(ex*ex + ey*ey)
            if norm > 1e-12:
                axes.append((-ey / norm, ex / norm))
    for ax, ay in axes:
        proj1 = v1[:, 0] * ax + v1[:, 1] * ay
        proj2 = v2[:, 0] * ax + v2[:, 1] * ay
        if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
            return False
    return True


def _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle_deg, tol=1e-6):
    """Verify containment and non-overlap."""
    outer_normals = _hex_edge_normals(outer_angle_deg)
    inradius = outer_side * SQRT3 / 2.0
    for i in range(len(centers)):
        verts = _hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i])
        for vx, vy in verts:
            for nx, ny in outer_normals:
                dist = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny
                if dist > inradius + tol:
                    return False
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                return False
    return True


def _optimize_outer(centers, inner_angles, n_angles=91):
    """For fixed inner positions, optimize outer center, side, angle."""
    best_side = float('inf')
    best_center = np.array([0.0, 0.0])
    best_angle = 0.0
    for outer_angle in np.linspace(0, 60, n_angles):
        side, center = _min_enclosing_hex(centers, inner_angles, outer_angle)
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = outer_angle
    return best_side, best_center, best_angle


def _smooth_penalty(centers, inner_angles, outer_center, outer_side, outer_angle_deg):
    """
    Smooth objective: outer_side + penalty for overlaps + penalty for containment
    violations. Returns (value, grad) for BFGS over the variable vector.
    """
    # Variable vector layout:
    # [c0x, c0y, a0, c1x, c1y, a1, ..., c10x, c10y, a10, ox, oy, oangle]
    # We build it externally; here we just compute value given the decomposed arrays.
    normals = _hex_edge_normals(outer_angle_deg)
    inradius = outer_side * SQRT3 / 2.0

    # Containment violation (smooth max over all vertices and normals)
    cont = 0.0
    for i in range(len(centers)):
        verts = _hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i])
        for vx, vy in verts:
            for nx, ny in normals:
                d = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny - inradius
                if d > 0:
                    cont += d * d

    # Overlap penalty (smooth)
    overlap = 0.0
    for i in range(len(centers)):
        v1 = _hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i])
        for j in range(i + 1, len(centers)):
            v2 = _hex_vertices(centers[j][0], centers[j][1], 1.0, inner_angles[j])
            # Use a smooth min-distance via SAT: compute max over axes of
            # (min2 - max1) which is negative when separated.
            axes = []
            for verts in (v1, v2):
                for k in range(6):
                    x1, y1 = verts[k]
                    x2, y2 = verts[(k + 1) % 6]
                    ex = x2 - x1
                    ey = y2 - y1
                    nm = np.sqrt(ex*ex + ey*ey)
                    if nm > 1e-12:
                        axes.append((-ey / nm, ex / nm))
            sep = float('inf')
            for ax, ay in axes:
                p1 = v1[:, 0] * ax + v1[:, 1] * ay
                p2 = v2[:, 0] * ax + v2[:, 1] * ay
                gap = min(p2.min() - p1.max(), p1.min() - p2.max())
                if gap < sep:
                    sep = gap
            if sep < 0:
                overlap += sep * sep

    return outer_side + 100.0 * cont + 100.0 * overlap


def _lattice_template(scale=1.0):
    """Return (centers, inner_angles) for the classic 1+6+4 hex lattice."""
    d1 = SQRT3 * scale
    d2 = 2 * SQRT3 * scale
    centers = [(0.0, 0.0)]
    for i in range(6):
        ang = np.pi / 3 * i
        centers.append((d1 * np.cos(ang), d1 * np.sin(ang)))
    # 4 of the 12 second-ring positions (pick the 4 that are corners)
    sec = []
    for i in range(6):
        ang = np.pi / 3 * i
        sec.append((d2 * np.cos(ang), d2 * np.sin(ang)))
    # choose 4 alternating corners
    chosen = [0, 2, 4]  # 3 corners
    # add one more from the mid-positions
    ang = np.pi / 6
    sec.append((d2 * np.cos(ang), d2 * np.sin(ang)))
    chosen.append(6)
    for idx in chosen:
        centers.append(sec[idx])
    inner_angles = [0.0] * N
    return centers, inner_angles


def _bfgs_optimize(centers0, angles0, max_iter=200, deadline=DEADLINE):
    """
    Jointly optimize inner positions, inner angles, and outer params via BFGS
    on a smooth objective. Returns (centers, angles, outer_center, outer_side,
    outer_angle).
    """
    # Flatten variables
    x = np.empty(3 * N + 3)
    for i in range(N):
        x[3*i] = centers0[i][0]
        x[3*i+1] = centers0[i][1]
        x[3*i+2] = angles0[i]
    side0, c0, ang0 = _optimize_outer(centers0, angles0, n_angles=41)
    x[3*N] = c0[0]
    x[3*N+1] = c0[1]
    x[3*N+2] = ang0

    def unpack(v):
        centers = [(v[3*i], v[3*i+1]) for i in range(N)]
        angles = [v[3*i+2] for i in range(N)]
        oc = (v[3*N], v[3*N+1])
        oang = v[3*N+2]
        return centers, angles, oc, oang

    def objective(v):
        centers, angles, oc, oang = unpack(v)
        side, c_opt = _min_enclosing_hex(centers, angles, oang)
        # Use the actual enclosing side (re-optimized center) as the cost,
        # plus smooth overlap/containment penalties.
        normals = _hex_edge_normals(oang)
        inradius = side * SQRT3 / 2.0
        cont = 0.0
        for i in range(N):
            verts = _hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            for vx, vy in verts:
                for nx, ny in normals:
                    d = (vx - c_opt[0]) * nx + (vy - c_opt[1]) * ny - inradius
                    if d > 0:
                        cont += d * d
        overlap = 0.0
        for i in range(N):
            v1 = _hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            for j in range(i + 1, N):
                v2 = _hex_vertices(centers[j][0], centers[j][1], 1.0, angles[j])
                axes = []
                for verts in (v1, v2):
                    for k in range(6):
                        xa, ya = verts[k]
                        xb, yb = verts[(k+1) % 6]
                        ex = xb - xa
                        ey = yb - ya
                        nm = np.sqrt(ex*ex + ey*ey)
                        if nm > 1e-12:
                            axes.append((-ey/nm, ex/nm))
                sep = float('inf')
                for ax, ay in axes:
                    p1 = v1[:,0]*ax + v1[:,1]*ay
                    p2 = v2[:,0]*ax + v2[:,1]*ay
                    gap = min(p2.min() - p1.max(), p1.min() - p2.max())
                    if gap < sep:
                        sep = gap
                if sep < 0:
                    overlap += sep * sep
        return side + 50.0 * cont + 50.0 * overlap

    # Numerical gradient via central differences
    def grad(v):
        g = np.empty_like(v)
        h = 1e-4
        f0 = objective(v)
        for i in range(len(v)):
            vp = v.copy(); vp[i] += h
            vm = v.copy(); vm[i] -= h
            g[i] = (objective(vp) - objective(vm)) / (2*h)
        return g

    # Simple BFGS
    m = len(x)
    B = np.eye(m)  # Hessian approx
    xk = x.copy()
    fk = objective(xk)
    best_x = xk.copy()
    best_f = fk
    for it in range(max_iter):
        if time.time() > deadline:
            break
        gk = grad(xk)
        if np.linalg.norm(gk) < 1e-7:
            break
        # Newton step
        try:
            s = np.linalg.solve(B, -gk)
        except np.linalg.LinAlgError:
            s = -gk
        # Line search (backtracking)
        alpha = 1.0
        s_norm = np.linalg.norm(s)
        if s_norm < 1e-12:
            break
        s = s / s_norm
        for _ in range(30):
            x_new = xk + alpha * s
            f_new = objective(x_new)
            if f_new < fk - 1e-8 * alpha * np.dot(gk, s) + 1e-12:
                break
            alpha *= 0.5
        else:
            break
        # Update BFGS
        x_new = xk + alpha * s
        f_new = objective(x_new)
        g_new = grad(x_new)
        y = g_new - gk
        s_vec = x_new - xk
        sy = np.dot(s_vec, y)
        if sy > 1e-12:
            Hy = B @ y
            B = B + np.outer(y, y)/sy - np.outer(Hy, Hy)/np.dot(y, Hy)
        xk = x_new
        fk = f_new
        if fk < best_f:
            best_f = fk
            best_x = xk.copy()
        # Periodic re-optimization of outer to keep side tight
        if it % 10 == 0:
            centers, angles, oc, oang = unpack(best_x)
            side, c_opt, ang_opt = _optimize_outer(centers, angles, n_angles=61)
            best_x[3*N] = c_opt[0]
            best_x[3*N+1] = c_opt[1]
            best_x[3*N+2] = ang_opt

    centers, angles, oc, oang = unpack(best_x)
    side, c_opt, ang_opt = _optimize_outer(centers, angles, n_angles=121)
    return centers, angles, c_opt, side, ang_opt


def construct():
    start_time = time.time()

    best_side = float('inf')
    best_result = None

    # Template 1: hex lattice
    centers0, angles0 = _lattice_template()
    # Initial outer
    side0, c0, a0 = _optimize_outer(centers0, angles0, n_angles=61)
    if side0 < best_side and _check_validity(centers0, angles0, c0, side0, a0):
        best_side = side0
        best_result = (centers0, angles0, c0, side0, a0)

    # BFGS joint optimization from lattice start
    if time.time() < DEADLINE - 30:
        try:
            centers, angles, oc, side, oang = _bfgs_optimize(
                centers0, angles0, max_iter=150, deadline=DEADLINE - 20)
            if side < best_side and _check_validity(centers, angles, oc, side, oang):
                best_side = side
                best_result = (centers, angles, oc, side, oang)
        except Exception:
            pass

    # Try a few perturbed lattice scales as additional starts
    if time.time() < DEADLINE - 40:
        for scale in (0.98, 1.02, 0.95, 1.05):
            if time.time() > DEADLINE - 20:
                break
            try:
                cs, an = _lattice_template(scale)
                # Quick overlap check
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if _hex_overlap(cs[i], an[i], cs[j], an[j]):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                side, c, a = _optimize_outer(cs, an, n_angles=61)
                if side < best_side and _check_validity(cs, an, c, side, a):
                    best_side = side
                    best_result = (cs, an, c, side, a)
            except Exception:
                continue

    # Final refinement: re-optimize outer on best result with fine angle grid
    if best_result is not None and time.time() < DEADLINE - 5:
        cs, an, _, _, _ = best_result
        try:
            side, c, a = _optimize_outer(cs, an, n_angles=241)
            if side <= best_side + 1e-9 and _check_validity(cs, an, c, side, a):
                best_side = side
                best_result = (cs, an, c, side, a)
        except Exception:
            pass

    # Fallback: linear arrangement
    if best_result is None:
        centers = [(2 * SQRT3 * k, 0.0) for k in range(N)]
        inner_angles = [0.0] * N
        side, center, angle = _optimize_outer(centers, inner_angles)
        best_side = side
        best_result = (centers, inner_angles, center, side, angle)

    centers, inner_angles, outer_center, outer_side, outer_angle = best_result
    inner = [[float(centers[i][0]), float(centers[i][1]), float(inner_angles[i])] for i in range(N)]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
