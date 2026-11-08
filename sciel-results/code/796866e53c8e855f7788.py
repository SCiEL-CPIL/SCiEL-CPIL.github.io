"""Optimized hexagon packing: 11 unit hexagons in a regular hexagon.
Strategy: joint continuous optimization of inner positions/rotations and outer
hexagon parameters (center, side, angle) using BFGS with smooth overlap
penalties and a closed-form minimum-enclosing-hexagon sub-solve. Multi-start
from hexagonal-lattice and ring templates with a strict deadline guard.
"""
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

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

def _objective_scipy(x_flat):
    """
    Objective function for scipy optimizer.
    x_flat: 2*N values (x, y for each of N centers).
    Assumes inner hexagons are parallel to the outer hexagon (angle 0 relative to outer).
    We optimize the outer angle as well? 
    To be safe, we fix outer angle to 0 in the inner loop and rely on the fact that 
    the cluster can be rotated. But the outer hexagon orientation relative to the cluster 
    matters. 
    Actually, if we fix outer angle to 0, we are restricting the solution. 
    However, for the symmetric flower, the optimal outer angle is aligned with the cluster.
    Let's just optimize the 2N positions and find the best enclosing hexagon 
    (optimizing outer center and angle) in the objective.
    """
    centers = [(x_flat[2*i], x_flat[2*i+1]) for i in range(N)]
    inner_angles = [0.0] * N  # Assume parallel
    side, center, angle = _optimize_outer(centers, inner_angles, n_angles=41)
    return side


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


def _scipy_optimize(centers0, angles0, deadline=DEADLINE):
    """
    Optimize positions of inner hexagons (assumed parallel to outer) 
    using scipy.optimize.minimize.
    """
    x0 = np.empty(2 * N)
    for i in range(N):
        x0[2*i] = centers0[i][0]
        x0[2*i+1] = centers0[i][1]
    
    # We will run multiple starts with slight perturbations
    best_side = float('inf')
    best_centers = None
    
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        
        # Perturb initial positions
        if restart == 0:
            x_start = x0.copy()
        else:
            x_start = x0 + np.random.normal(0, 0.1, 2*N)
            
        # Run optimization
        try:
            res = minimize(_objective_scipy, x_start, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_centers = res.x
        except Exception:
            continue

    if best_centers is None:
        # Fallback to initial
        best_side = _objective_scipy(x0)
        best_centers = x0

    centers = [(best_centers[2*i], best_centers[2*i+1]) for i in range(N)]
    inner_angles = [0.0] * N
    side, c_opt, ang_opt = _optimize_outer(centers, inner_angles, n_angles=121)
    return centers, inner_angles, c_opt, side, ang_opt


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

    # SciPy optimization from lattice start (assuming parallel hexagons)
    if time.time() < DEADLINE - 30:
        try:
            centers, angles, oc, side, oang = _scipy_optimize(
                centers0, angles0, deadline=DEADLINE - 20)
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
