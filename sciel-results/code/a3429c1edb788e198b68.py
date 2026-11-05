import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 11
D = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _hexagons_overlap(v1, v2):
    """
    Check if two regular hexagons (given as vertices) overlap using Separating Axis Theorem.
    Returns True if they overlap, False otherwise.
    """
    # Axes to check: normals to the edges of both hexagons
    # For a regular hexagon, the normals are at angles offset by 30 deg from vertices
    # We can compute axes from the edges directly
    axes = []
    for i in range(6):
        # Edge of v1
        e1 = v1[(i+1)%6] - v1[i]
        # Normal
        n1 = np.array([-e1[1], e1[0]])
        axes.append(n1)
        # Edge of v2
        e2 = v2[(i+1)%6] - v2[i]
        n2 = np.array([-e2[1], e2[0]])
        axes.append(n2)
    
    for axis in axes:
        # Project v1
        proj1 = np.dot(v1, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        # Project v2
        proj2 = np.dot(v2, axis)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for gap
        if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
            return False
    return True

def _outer_hex_vertices(cx, cy, side, angle_deg):
    """Vertices of outer regular hexagon."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([side * np.cos(angles), side * np.sin(angles)])
    return vs + np.array([cx, cy])

def _point_in_polygon(pt, poly):
    """Check if point is inside convex polygon using cross products."""
    n = len(poly)
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i+1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cross > 1e-12:
            if sign < 0:
                return False
            sign = 1
        elif cross < -1e-12:
            if sign > 0:
                return False
            sign = -1
    return True

def _compute_side(inner_params, outer_angle_deg):
    """
    Compute the minimum outer hexagon side length such that all inner hexagons
    are contained within the outer hexagon.
    Uses analytic support-function computation (much faster than binary search).
    inner_params: array of shape (n, 3) with [x, y, angle_deg]
    """
    n = len(inner_params)
    
    # Outer hexagon: vertices at angles [0,60,...,300] + outer_angle_deg
    # The constraint for a point p to be inside outer hexagon of side S centered at origin:
    # For each of the 6 edges, the signed distance from p to the edge line must be <= S
    # Edge i has outward normal at angle (outer_angle_deg + 60*i + 30) degrees
    # The half-width of the hexagon in the direction of normal i is S * cos(30°) = S * sqrt(3)/2
    # So: dot(p, normal_i) <= S * sqrt(3)/2  for all i
    # => S >= 2/sqrt(3) * max_i dot(p, normal_i) for all p in all inner hexagons
    
    # For each inner hexagon, the max of dot(p, normal) over p in hexagon is:
    # dot(center, normal) + inradius_of_inner_hex = dot(center, normal) + sqrt(3)/2
    # (since inner hexagons have side 1, their inradius is sqrt(3)/2)
    
    # So S >= 2/sqrt(3) * max over all inner hexagons i and all 6 normals n_j of:
    #   dot(center_i, n_j) + sqrt(3)/2
    
    normals = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + outer_angle_deg + 30)
    normal_vectors = np.column_stack([np.cos(normals), np.sin(normals)])
    
    max_val = 0.0
    for i in range(n):
        cx = inner_params[i, 0]
        cy = inner_params[i, 1]
        # dot product of center with each normal
        dots = normal_vectors @ np.array([cx, cy])
        # add inradius of unit hexagon
        vals = dots + D / 2.0
        m = np.max(vals)
        if m > max_val:
            max_val = m
    
    return 2.0 / D * max_val

def _overlap_penalty(inner_params, penalty_weight=100.0):
    """
    Compute penalty for overlapping inner hexagons.
    Returns 0 if all hexagons are disjoint, positive otherwise.
    Uses fast SAT overlap check.
    """
    n = len(inner_params)
    penalty = 0.0
    for i in range(n):
        x1 = inner_params[i, 0]
        y1 = inner_params[i, 1]
        a1 = inner_params[i, 2]
        v1 = _hex_vertices(x1, y1, a1)
        
        for j in range(i+1, n):
            x2 = inner_params[j, 0]
            y2 = inner_params[j, 1]
            a2 = inner_params[j, 2]
            v2 = _hex_vertices(x2, y2, a2)
            
            if _hexagons_overlap(v1, v2):
                penalty += penalty_weight
    
    return penalty

def _objective(params, outer_angle_deg, n):
    """
    Objective function: outer side length + overlap penalty.
    params: flat array of length 3*n: [x1, y1, a1, x2, y2, a2, ...]
    """
    inner_params = params.reshape(n, 3)
    side = _compute_side(inner_params, outer_angle_deg)
    overlap_pen = _overlap_penalty(inner_params)
    return side + overlap_pen

def _smooth_objective(params, outer_angle_deg, n, t_init=5.0, beta=100.0):
    """
    Smooth epigraph objective: minimizes t + overlap_penalty
    subject to containment constraints (smoothly enforced via soft-max).
    params: flat array of length 3*n + 1: [x1, y1, a1, ..., x_n, y_n, a_n, t]
    """
    inner_params = params[:3*n].reshape(n, 3)
    t = params[3*n]
    
    # Compute containment requirements for each inner hexagon
    normals = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + outer_angle_deg + 30)
    normal_vectors = np.column_stack([np.cos(normals), np.sin(normals)])
    
    # For each inner hexagon i, required t >= 2/sqrt(3) * max_j (dot(center_i, n_j) + sqrt(3)/2)
    # We use a smooth max via log-sum-exp
    # Constraint: t - 2/sqrt(3) * (dot(c_i, n_j) + sqrt(3)/2) >= 0 for all i, j
    
    # Collect all constraint violations
    violations = []
    for i in range(n):
        cx = inner_params[i, 0]
        cy = inner_params[i, 1]
        dots = normal_vectors @ np.array([cx, cy]) + D / 2.0
        for j in range(6):
            # t must be >= 2/sqrt(3) * dots[j]
            violations.append(t - 2.0 / D * dots[j])
    
    violations = np.array(violations)
    
    # Soft-min of violations: if all >= 0, penalty is ~0
    # Use: -1/beta * log(sum(exp(-beta * v))) which is ~min(v) when beta is large
    # If min(v) < 0, this gives a positive penalty
    neg_beta_v = -beta * violations
    # Clip for numerical stability
    neg_beta_v = np.clip(neg_beta_v, -500, 500)
    max_val = np.max(neg_beta_v)
    soft_min = -max_val / beta - np.log(np.sum(np.exp(neg_beta_v - max_val))) / beta
    
    # Penalty: if soft_min < 0, we add -soft_min (i.e., the deficit)
    containment_penalty = max(0.0, -soft_min) * 10.0
    
    overlap_pen = _overlap_penalty(inner_params)
    
    return t + containment_penalty + overlap_pen

def _seed_solution():
    """Return the concentric ring seed solution."""
    eps = 1e-9
    centers = []
    centers.append((0.0, 0.0))
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (np.sqrt(3) + eps) * np.cos(angle)
        y = (np.sqrt(3) + eps) * np.sin(angle)
        centers.append((x, y))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = (3.0 + 2*eps) * np.cos(angle)
        y = (3.0 + 2*eps) * np.sin(angle)
        centers.append((x, y))
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
    
    max_x = (2 + eps) * np.sqrt(3)
    max_y = 4 + 2 * eps
    s_x = 2 * max_x / np.sqrt(3)
    s_y = max_y
    side = max(s_x, s_y) + 1e-9
    
    return inner, (0.0, 0.0), side, 30.0

def _is_valid(inner, outer_angle):
    """Check that all inner hexagons are inside outer and non-overlapping."""
    n = len(inner)
    inner_params = np.array(inner)
    side = _compute_side(inner_params, outer_angle)
    # Check containment with a small margin
    outer_v = _outer_hex_vertices(0.0, 0.0, side, outer_angle)
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        hex_v = _hex_vertices(x, y, a)
        for v in hex_v:
            if not _point_in_polygon(v, outer_v):
                return False
    # Check non-overlap using SAT
    for i in range(n):
        v1 = _hex_vertices(inner_params[i,0], inner_params[i,1], inner_params[i,2])
        for j in range(i+1, n):
            v2 = _hex_vertices(inner_params[j,0], inner_params[j,1], inner_params[j,2])
            if _hexagons_overlap(v1, v2):
                return False
    return True

def construct():
    start_time = time.time()
    DEADLINE = 150.0
    safety_margin = 8.0

    n = 11
    outer_angle = 30.0

    # Seed solution
    seed_inner, seed_center, seed_side, seed_angle = _seed_solution()
    best_inner = list(seed_inner)
    best_side = seed_side

    # Initial guess: concentric ring
    init_params = np.zeros(3 * n)
    for i, (x, y, a) in enumerate(seed_inner):
        init_params[i*3] = x
        init_params[i*3+1] = y
        init_params[i*3+2] = a

    def _elapsed():
        return time.time() - start_time

    # Phase 1: Nelder-Mead on the nonsmooth objective
    if _elapsed() < DEADLINE - safety_margin:
        try:
            res = minimize(_objective, init_params, args=(outer_angle, n), method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            cand_inner = [[res.x[i*3], res.x[i*3+1], res.x[i*3+2]] for i in range(n)]
            cand_side = _compute_side(np.array(cand_inner), outer_angle)
            if cand_side < best_side and _is_valid(cand_inner, outer_angle):
                best_inner = cand_inner
                best_side = cand_side
        except Exception:
            pass

    # Phase 2: Smooth epigraph L-BFGS-B for local refinement
    if _elapsed() < DEADLINE - safety_margin:
        try:
            # Use best_inner as starting point
            start = np.zeros(3 * n + 1)
            for i in range(n):
                start[i*3] = best_inner[i][0]
                start[i*3+1] = best_inner[i][1]
                start[i*3+2] = best_inner[i][2]
            start[3*n] = best_side
            
            res2 = minimize(_smooth_objective, start, args=(outer_angle, n, best_side, 100.0),
                           method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            cand_inner = [[res2.x[i*3], res2.x[i*3+1], res2.x[i*3+2]] for i in range(n)]
            cand_side = _compute_side(np.array(cand_inner), outer_angle)
            if cand_side < best_side and _is_valid(cand_inner, outer_angle):
                best_inner = cand_inner
                best_side = cand_side
        except Exception:
            pass

    # Phase 3: Removed to save time. The SAT check is fast enough for the previous phases.
    # If we have time, do a second pass of NM with a slightly different start or just refine.
    if _elapsed() < DEADLINE - safety_margin:
        try:
            # Quick refinement of the best solution found
            start_params = np.array(best_inner).flatten()
            res3 = minimize(_objective, start_params, args=(outer_angle, n), method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            cand_inner = [[res3.x[i*3], res3.x[i*3+1], res3.x[i*3+2]] for i in range(n)]
            cand_side = _compute_side(np.array(cand_inner), outer_angle)
            if cand_side < best_side and _is_valid(cand_inner, outer_angle):
                best_inner = cand_inner
                best_side = cand_side
        except Exception:
            pass

    # Final validation
    if best_inner:
        inner_params = np.array(best_inner)
        side = _compute_side(inner_params, outer_angle)
        if not _is_valid(best_inner, outer_angle):
            best_inner = list(seed_inner)
            best_side = seed_side
        else:
            best_side = side

    return best_inner, (0.0, 0.0), best_side, outer_angle
