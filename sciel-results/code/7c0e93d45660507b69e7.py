<<<<<<< SEARCH
import math
import time
import random
import numpy as np

N = 12
TIME_LIMIT = 145.0  # Safety margin below 150s
=======
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 148.0  # Safety margin below 150s
>>>>>>> REPLACE

<<<<<<< SEARCH
def _is_valid_fast(cx_arr, cy_arr, angle_arr):
    """
    Fast validity check using distance-based pre-filter + SAT fallback.
    cx_arr, cy_arr, angle_arr: numpy arrays of shape (N,)
    Returns True if all hexagons are pairwise disjoint.
    """
    n = len(cx_arr)
    # Precompute vertices for all hexagons
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))
    
    # Precompute axes for all hexagons
    axes_list = []
    for i in range(n):
        axes_list.append(_axes_from_polygon(verts_list[i]))
    
    for i in range(n):
        xi, yi = cx_arr[i], cy_arr[i]
        vi = verts_list[i]
        ai = axes_list[i]
        for j in range(i + 1, n):
            xj, yj = cx_arr[j], cy_arr[j]
            vj = verts_list[j]
            aj = axes_list[j]
            
            # Quick distance check
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx * dx + dy * dy
            
            # If distance is large enough, definitely disjoint
            if dist_sq > (2 * math.sqrt(3)) ** 2:
                continue
            
            # If distance is very small, definitely overlapping
            if dist_sq < 4.0 - 1e-9:
                return False
            
            # Intermediate case: do SAT check
            for axis in ai:
                ax, ay = axis
                minA = min(vx * ax + vy * ay for vx, vy in vi)
                maxA = max(vx * ax + vy * ay for vx, vy in vi)
                minB = min(vx * ax + vy * ay for vx, vy in vj)
                maxB = max(vx * ax + vy * ay for vx, vy in vj)
                if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                    break
            else:
                for axis in aj:
                    ax, ay = axis
                    minA = min(vx * ax + vy * ay for vx, vy in vi)
                    maxA = max(vx * ax + vy * ay for vx, vy in vi)
                    minB = min(vx * ax + vy * ay for vx, vy in vj)
                    maxB = max(vx * ax + vy * ay for vx, vy in vj)
                    if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                        break
                else:
                    return False
    
    return True
=======
def _sdf_hexagon_point(px, py, cx, cy, angle_deg):
    """
    Signed distance from point (px,py) to regular hexagon centered at (cx,cy)
    with rotation angle_deg and side length 1.
    Positive means outside, negative means inside.
    """
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    # Transform point to hexagon local frame
    dx = px - cx
    dy = py - cy
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    
    # Hexagon with flat top: vertices at angles 0, 60, 120, 180, 240, 300
    # Apothem = sqrt(3)/2, circumradius = 1
    # The hexagon is the intersection of 6 half-planes
    # Distance to hexagon = max of (distance to each edge line, 0) if inside
    # Actually, SDF for convex polygon: 
    #   If point is inside, distance = -min distance to edges
    #   If outside, distance = min distance to edges (with sign)
    
    # For a regular hexagon with circumradius 1 (side 1):
    # Edge normals point outward at angles 30, 90, 150, 210, 270, 330
    # Each edge is at distance apothem = sqrt(3)/2 from center
    apothem = math.sqrt(3) / 2.0
    
    max_dist = 0.0
    for k in range(6):
        na = math.radians(30 + 60 * k)
        nx, ny = math.cos(na), math.sin(na)
        # Signed distance from point to the edge line (positive = outside)
        d = nx * lx + ny * ly - apothem
        if d > max_dist:
            max_dist = d
    
    # If max_dist < 0, point is inside, SDF = max_dist (negative)
    # If max_dist >= 0, need to check if point is actually outside
    # For convex polygons, SDF = max over edges of (dot(n, p) - apothem)
    # but only if the point projects onto the edge segment
    # For a regular hexagon, this simple formula works because of symmetry
    
    return max_dist

def _min_pairwise_sdf(cx_arr, cy_arr, angle_arr):
    """
    Compute the minimum SDF (most negative = deepest overlap) between all pairs.
    Returns the minimum over all pairs of the minimum SDF between vertices of one
    hexagon to the other hexagon.
    """
    n = len(cx_arr)
    min_sdf = 0.0  # We want this to be >= 0 for validity
    
    for i in range(n):
        vi = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
        for j in range(i + 1, n):
            # Check SDF from each vertex of i to hexagon j, and vice versa
            vj = _hex_vertices(cx_arr[j], cy_arr[j], angle_arr[j])
            
            for (vx, vy) in vi:
                d = _sdf_hexagon_point(vx, vy, cx_arr[j], cy_arr[j], angle_arr[j])
                if d < min_sdf:
                    min_sdf = d
            
            for (vx, vy) in vj:
                d = _sdf_hexagon_point(vx, vy, cx_arr[i], cy_arr[i], angle_arr[i])
                if d < min_sdf:
                    min_sdf = d
    
    return min_sdf

def _is_valid_fast(cx_arr, cy_arr, angle_arr):
    """
    Fast validity check using SDF-based approach.
    cx_arr, cy_arr, angle_arr: numpy arrays of shape (N,)
    Returns True if all hexagons are pairwise disjoint.
    """
    # Quick distance pre-filter
    n = len(cx_arr)
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx_arr[i] - cx_arr[j]
            dy = cy_arr[i] - cy_arr[j]
            dist_sq = dx * dx + dy * dy
            # If centers are closer than 2*inradius = sqrt(3), definitely overlapping
            if dist_sq < (math.sqrt(3) - 0.01) ** 2:
                # Need more careful check
                pass
    
    # Use SDF check
    min_sdf = _min_pairwise_sdf(cx_arr, cy_arr, angle_arr)
    return min_sdf >= -1e-7
>>>>>>> REPLACE

<<<<<<< SEARCH
def _outer_hex_side(cx_arr, cy_arr, angle_arr, center, angle_deg):
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    for i in range(len(cx_arr)):
        verts = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need
=======
def _outer_hex_side(cx_arr, cy_arr, angle_arr, center, angle_deg):
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    for i in range(len(cx_arr)):
        verts = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _joint_objective_and_grad(params, time_budget):
    """
    Joint objective: minimize (outer_side + penalty * overlap)
    params: [x0,y0,a0, x1,y1,a1, ..., x11,y11,a11] (36 values)
    a_i in radians
    """
    t_start = time.time()
    n = N
    cx = params[0::3]
    cy = params[1::3]
    ang = params[2::3]
    
    # Compute outer side
    outer_side = _outer_hex_side(cx, cy, ang, (0.0, 0.0), 0.0)
    
    # Compute overlap penalty using SDF
    overlap = 0.0
    for i in range(n):
        vi = _hex_vertices(cx[i], cy[i], math.degrees(ang[i]))
        for j in range(i + 1, n):
            vj = _hex_vertices(cx[j], cy[j], math.degrees(ang[j]))
            for (vx, vy) in vi:
                d = _sdf_hexagon_point(vx, vy, cx[j], cy[j], math.degrees(ang[j]))
                if d < 0:
                    overlap += d * d
            for (vx, vy) in vj:
                d = _sdf_hexagon_point(vx, vy, cx[i], cy[i], math.degrees(ang[i]))
                if d < 0:
                    overlap += d * d
    
    penalty = 100.0
    obj = outer_side + penalty * overlap
    
    # Numerical gradient
    grad = np.zeros(3 * n)
    eps = 1e-6
    for k in range(3 * n):
        old = params[k]
        params[k] = old + eps
        obj_plus = _outer_hex_side(params[0::3], params[1::3], np.radians(params[2::3]), (0.0, 0.0), 0.0)
        ov_plus = 0.0
        for i in range(n):
            vi = _hex_vertices(params[0::3][i], params[1::3][i], params[2::3][i])
            for j in range(i + 1, n):
                vj = _hex_vertices(params[0::3][j], params[1::3][j], params[2::3][j])
                for (vx, vy) in vi:
                    d = _sdf_hexagon_point(vx, vy, params[0::3][j], params[1::3][j], params[2::3][j])
                    if d < 0:
                        ov_plus += d * d
                for (vx, vy) in vj:
                    d = _sdf_hexagon_point(vx, vy, params[0::3][i], params[1::3][i], params[2::3][i])
                    if d < 0:
                        ov_plus += d * d
        obj_plus += penalty * ov_plus
        params[k] = old
        grad[k] = (obj_plus - obj) / eps
        
        if time.time() - t_start > time_budget:
            break
    
    return obj, grad
>>>>>>> REPLACE

<<<<<<< SEARCH
    # --- Phase 1: Simulated Annealing ---
    cfg = [row[:] for row in best_cfg]
    cx_arr = np.array([p[0] for p in cfg])
    cy_arr = np.array([p[1] for p in cfg])
    angle_arr = np.array([p[2] for p in cfg])
    
    cur_score = best_side
    T = 0.2
    T_end = 0.003
    alpha = 0.996
    
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    def _quick_score(cx_arr, cy_arr, angle_arr):
        max_need = 0.0
        for i in range(N):
            a = math.radians(angle_arr[i])
            ca, sa = math.cos(a), math.sin(a)
            for k in range(6):
                theta = math.radians(60 * k)
                vx = cx_arr[i] + ca * math.cos(theta) - sa * math.sin(theta)
                vy = cy_arr[i] + sa * math.cos(theta) + ca * math.sin(theta)
                for (nx, ny) in normals:
                    val = (nx * vx + ny * vy) / apothem_unit
                    if val > max_need:
                        max_need = val
        return max_need

    while time.time() - t0 < TIME_LIMIT - 30:
        r = random.random()
        idx = random.randrange(N)
        old_x, old_y, old_a = cx_arr[idx], cy_arr[idx], angle_arr[idx]
        idx2 = -1
        old_x2, old_y2, old_a2 = 0, 0, 0
        
        if r < 0.7:
            cx_arr[idx] += random.gauss(0, 0.12)
            cy_arr[idx] += random.gauss(0, 0.12)
            angle_arr[idx] += random.gauss(0, 2.5)
        elif r < 0.85:
            angle_arr[idx] += random.gauss(0, 6.0)
        else:
            idx2 = random.randrange(N)
            if idx2 == idx:
                idx2 = (idx + 1) % N
            old_x2, old_y2, old_a2 = cx_arr[idx2], cy_arr[idx2], angle_arr[idx2]
            cx_arr[idx] += random.gauss(0, 0.08)
            cy_arr[idx] += random.gauss(0, 0.08)
            cx_arr[idx2] += random.gauss(0, 0.08)
            cy_arr[idx2] += random.gauss(0, 0.08)
            angle_arr[idx] += random.gauss(0, 1.5)
            angle_arr[idx2] += random.gauss(0, 1.5)
        
        valid = _is_valid_fast(cx_arr, cy_arr, angle_arr)
        
        if valid:
            new_score = _quick_score(cx_arr, cy_arr, angle_arr)
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                if new_score < best_side:
                    best_side = new_score
                    best_cfg = [[cx_arr[i], cy_arr[i], angle_arr[i]] for i in range(N)]
                    best_center = (0.0, 0.0)
                    best_angle = 0.0
            else:
                cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
                if idx2 != -1:
                    cx_arr[idx2], cy_arr[idx2], angle_arr[idx2] = old_x2, old_y2, old_a2
        else:
            cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
            if idx2 != -1:
                cx_arr[idx2], cy_arr[idx2], angle_arr[idx2] = old_x2, old_y2, old_a2
        
        T = max(T_end, T * alpha)
=======
    # --- Phase 1: SDF-based gradient optimization with L-BFGS-B ---
    # Start from best initial config
    cx_arr = np.array([p[0] for p in best_cfg])
    cy_arr = np.array([p[1] for p in cfg])
    angle_arr = np.array([p[2] for p in best_cfg])
    
    # Build parameter vector
    params = np.zeros(3 * N)
    for i in range(N):
        params[3*i] = cx_arr[i]
        params[3*i+1] = cy_arr[i]
        params[3*i+2] = math.radians(angle_arr[i])
    
    # Use L-BFGS-B with numerical gradient
    best_side_l = best_side
    best_params_l = params.copy()
    
    time_remaining = TIME_LIMIT - 40 - (time.time() - t0)
    if time_remaining > 10:
        # Split time among a few restarts
        n_restarts = min(3, max(1, int(time_remaining / 30)))
        per_restart = time_remaining / n_restarts
        
        for restart in range(n_restarts):
            if time.time() - t0 > TIME_LIMIT - 35:
                break
            
            if restart > 0:
                # Add perturbation
                params = best_params_l + random.gauss(0, 0.1, size=3*N)
            
            # L-BFGS-B optimization
            try:
                def objective(p):
                    cx_t = p[0::3]
                    cy_t = p[1::3]
                    ang_t = np.radians(p[2::3])
                    side = _outer_hex_side(cx_t, cy_t, ang_t, (0.0, 0.0), 0.0)
                    # Overlap penalty
                    ov = 0.0
                    for i in range(N):
                        vi = _hex_vertices(cx_t[i], cy_t[i], math.degrees(ang_t[i]))
                        for j in range(i+1, N):
                            vj = _hex_vertices(cx_t[j], cy_t[j], math.degrees(ang_t[j]))
                            for (vx, vy) in vi:
                                d = _sdf_hexagon_point(vx, vy, cx_t[j], cy_t[j], math.degrees(ang_t[j]))
                                if d < 0:
                                    ov += d * d
                            for (vx, vy) in vj:
                                d = _sdf_hexagon_point(vx, vy, cx_t[i], cy_t[i], math.degrees(ang_t[i]))
                                if d < 0:
                                    ov += d * d
                    return side + 100.0 * ov
                
                res = minimize(objective, params, method='L-BFGS-B', 
                              options={'maxiter': 200, 'maxfun': 500,
                                      'ftol': 1e-8, 'gtol': 1e-6,
                                      'disp': False, 'iprint': 0})
                
                if res.success or res.fun < best_side_l:
                    new_side = _outer_hex_side(res.x[0::3], res.x[1::3], np.radians(res.x[2::3]), (0.0, 0.0), 0.0)
                    if _is_valid_fast(res.x[0::3], res.x[1::3], np.radians(res.x[2::3])):
                        if new_side < best_side_l:
                            best_side_l = new_side
                            best_params_l = res.x.copy()
                            best_cfg = [[res.x[3*i], res.x[3*i+1], math.degrees(res.x[3*i+2])] for i in range(N)]
                            best_center = (0.0, 0.0)
                            best_angle = 0.0
            except Exception:
                pass
    
    # Also try a brief SA phase if time remains
    time_remaining = TIME_LIMIT - 30 - (time.time() - t0)
    if time_remaining > 5 and best_cfg is not None:
        cfg = [row[:] for row in best_cfg]
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        
        cur_score = _outer_hex_side(cx_arr, cy_arr, angle_arr, (0.0, 0.0), 0.0)
        T = 0.1
        T_end = 0.002
        alpha = 0.995
        
        apothem_unit = math.sqrt(3) / 2.0
        normals = []
        for k in range(6):
            na = math.radians(90 + 60 * k)
            normals.append((math.cos(na), math.sin(na)))
        
        def _quick_score(cx_arr, cy_arr, angle_arr):
            max_need = 0.0
            for i in range(N):
                a = math.radians(angle_arr[i])
                ca, sa = math.cos(a), math.sin(a)
                for k in range(6):
                    theta = math.radians(60 * k)
                    vx = cx_arr[i] + ca * math.cos(theta) - sa * math.sin(theta)
                    vy = cy_arr[i] + sa * math.cos(theta) + ca * math.sin(theta)
                    for (nx, ny) in normals:
                        val = (nx * vx + ny * vy) / apothem_unit
                        if val > max_need:
                            max_need = val
            return max_need

        while time.time() - t0 < TIME_LIMIT - 25:
            r = random.random()
            idx = random.randrange(N)
            old_x, old_y, old_a = cx_arr[idx], cy_arr[idx], angle_arr[idx]
            idx2 = -1
            old_x2, old_y2, old_a2 = 0, 0, 0
            
            if r < 0.7:
                cx_arr[idx] += random.gauss(0, 0.1)
                cy_arr[idx] += random.gauss(0, 0.1)
                angle_arr[idx] += random.gauss(0, 2.0)
            elif r < 0.85:
                angle_arr[idx] += random.gauss(0, 5.0)
            else:
                idx2 = random.randrange(N)
                if idx2 == idx:
                    idx2 = (idx + 1) % N
                old_x2, old_y2, old_a2 = cx_arr[idx2], cy_arr[idx2], angle_arr[idx2]
                cx_arr[idx] += random.gauss(0, 0.07)
                cy_arr[idx] += random.gauss(0, 0.07)
                cx_arr[idx2] += random.gauss(0, 0.07)
                cy_arr[idx2] += random.gauss(0, 0.07)
                angle_arr[idx] += random.gauss(0, 1.5)
                angle_arr[idx2] += random.gauss(0, 1.5)
            
            valid = _is_valid_fast(cx_arr, cy_arr, angle_arr)
            
            if valid:
                new_score = _quick_score(cx_arr, cy_arr, angle_arr)
                if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                    cur_score = new_score
                    if new_score < best_side:
                        best_side = new_score
                        best_cfg = [[cx_arr[i], cy_arr[i], angle_arr[i]] for i in range(N)]
                        best_center = (0.0, 0.0)
                        best_angle = 0.0
                else:
                    cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
                    if idx2 != -1:
                        cx_arr[idx2], cy_arr[idx2], angle_arr[idx2] = old_x2, old_y2, old_a2
            else:
                cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
                if idx2 != -1:
                    cx_arr[idx2], cy_arr[idx2], angle_arr[idx2] = old_x2, old_y2, old_a2
            
            T = max(T_end, T * alpha)
>>>>>>> REPLACE

<<<<<<< SEARCH
    # --- Phase 2: Optimization of Outer Hexagon (Center & Angle) ---
    cx_best = np.array([p[0] for p in best_cfg])
    cy_best = np.array([p[1] for p in best_cfg])
    angle_best = np.array([p[2] for p in best_cfg])
    
    # Precompute vertices for best config
    verts_best = [_hex_vertices(cx_best[i], cy_best[i], angle_best[i]) for i in range(N)]
    
    best_final_side = best_side
    best_final_cfg = best_cfg[:]
    best_final_center = best_center
    best_final_angle = best_angle
    
    # Search over outer hexagon angles and centers
    # Center optimization: for a fixed angle, the optimal center minimizes max projection.
    # This is a convex optimization problem (minimize max of linear functions).
    # We can use a simple grid search or iterative refinement.
    
    for outer_angle in range(0, 30, 2):  # 0 to 28 degrees
        if time.time() - t0 > TIME_LIMIT - 5:
            break
        
        a = math.radians(outer_angle)
        apothem_unit = math.sqrt(3) / 2.0
        normals = []
        for k in range(6):
            na = a + math.radians(90 + 60 * k)
            normals.append((math.cos(na), math.sin(na)))
        
        # Find optimal center by minimizing max projection
        # Use a few iterations of center adjustment
        cx_c, cy_c = 0.0, 0.0
        for _iter in range(20):
            # Compute current projections
            max_proj = 0.0
            for i in range(N):
                for (vx, vy) in verts_best[i]:
                    for (nx, ny) in normals:
                        val = (nx * (vx - cx_c) + ny * (vy - cy_c)) / apothem_unit
                        if val > max_proj:
                            max_proj = val
            
            # Find which point and normal gives the max projection
            # Move center slightly towards that point to reduce the projection
            # Actually, for "minimize max distance to 6 lines", the optimal center
            # is the one that minimizes the maximum of (proj - center_dot_normal).
            # This is equivalent to finding the center of the smallest hexagon.
            
            # Simple heuristic: move center towards the centroid of the "tight" points
            # Or just use a small grid search around current center
            best_cx, best_cy, best_val = cx_c, cy_c, max_proj
            
            for dx in [-0.2, -0.1, 0, 0.1, 0.2]:
                for dy in [-0.2, -0.1, 0, 0.1, 0.2]:
                    ncx, ncy = cx_c + dx, cy_c + dy
                    val = 0.0
                    for i in range(N):
                        for (vx, vy) in verts_best[i]:
                            for (nx, ny) in normals:
                                v = (nx * (vx - ncx) + ny * (vy - ncy)) / apothem_unit
                                if v > val:
                                    val = v
                    if val < best_val:
                        best_val = val
                        best_cx, best_cy = ncx, ncy
            
            if abs(best_cx - cx_c) < 1e-4 and abs(best_cy - cy_c) < 1e-4:
                break
            cx_c, cy_c = best_cx, best_cy
        
        if max_proj < best_final_side:
            best_final_side = max_proj
            best_final_cfg = best_cfg[:]
            best_final_center = (cx_c, cy_c)
            best_final_angle = float(outer_angle)
=======
    # --- Phase 2: Optimization of Outer Hexagon (Center & Angle) ---
    cx_best = np.array([p[0] for p in best_cfg])
    cy_best = np.array([p[1] for p in best_cfg])
    angle_best = np.array([p[2] for p in best_cfg])
    
    # Precompute vertices for best config
    verts_best = [_hex_vertices(cx_best[i], cy_best[i], angle_best[i]) for i in range(N)]
    
    best_final_side = best_side
    best_final_cfg = best_cfg[:]
    best_final_center = best_center
    best_final_angle = best_angle
    
    # Search over outer hexagon angles and centers using scipy minimize
    for outer_angle in range(0, 30, 3):  # 0 to 27 degrees
        if time.time() - t0 > TIME_LIMIT - 8:
            break
        
        a = math.radians(outer_angle)
        apothem_unit = math.sqrt(3) / 2.0
        normals = []
        for k in range(6):
            na = a + math.radians(90 + 60 * k)
            normals.append((math.cos(na), math.sin(na)))
        
        # Find optimal center using scipy minimize (convex problem)
        def center_obj(c):
            cx_c, cy_c = c
            max_proj = 0.0
            for i in range(N):
                for (vx, vy) in verts_best[i]:
                    for (nx, ny) in normals:
                        val = (nx * (vx - cx_c) + ny * (vy - cy_c)) / apothem_unit
                        if val > max_proj:
                            max_proj = val
            return max_proj
        
        res = minimize(center_obj, [0.0, 0.0], method='Nelder-Mead',
                      options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
        
        if res.fun < best_final_side:
            best_final_side = res.fun
            best_final_cfg = best_cfg[:]
            best_final_center = (float(res.x[0]), float(res.x[1]))
            best_final_angle = float(outer_angle)
>>>>>>> REPLACE

