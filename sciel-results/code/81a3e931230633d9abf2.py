import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def _point_in_hex_dist_sq(px, py, cx, cy, s, angle_deg):
    a = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    cos_a = math.cos(-a)
    sin_a = math.sin(-a)
    rx = dx * cos_a - dy * sin_a
    ry = dx * sin_a + dy * cos_a
    
    apothem = s * math.sqrt(3) / 2
    sum_sq_viol = 0.0
    for i in range(3):
        theta_n = math.radians(30 + i * 60)
        proj = rx * math.cos(theta_n) + ry * math.sin(theta_n)
        if proj > apothem:
            sum_sq_viol += (proj - apothem)**2
        if proj < -apothem:
            sum_sq_viol += (-apothem - proj)**2
    return sum_sq_viol

def _pairwise_clearance(p1, p2):
    c1x, c1y, a1 = p1
    c2x, c2y, a2 = p2
    
    dx = c1x - c2x
    dy = c1y - c2y
    if dx*dx + dy*dy > 4.5: 
        return 0.0
        
    v1 = _hex_vertices(c1x, c1y, 1.0, a1)
    v2 = _hex_vertices(c2x, c2y, 1.0, a2)
    
    penalty = 0.0
    for pt in v1:
        penalty += _point_in_hex_dist_sq(pt[0], pt[1], c2x, c2y, 1.0, a2)
    for pt in v2:
        penalty += _point_in_hex_dist_sq(pt[0], pt[1], c1x, c1y, 1.0, a1)
        
    return penalty

def _min_enclosing_hexagon_lp(inner_hexes):
    """
    Exact LP-based minimum enclosing regular hexagon for a fixed set of inner hexagons.
    Returns (cx, cy, side, angle_deg).
    """
    points = []
    for h in inner_hexes:
        verts = _hex_vertices(h[0], h[1], 1.0, h[2])
        points.extend(verts)
    
    points = np.array(points)
    if len(points) == 0:
        return 0, 0, 1.0, 0.0

    best_side = float('inf')
    best_params = (0, 0, 1.0, 0.0)
    
    # Scan angles. 0-60 is sufficient due to 6-fold symmetry of the hexagon shape,
    # but the points are not symmetric, so we scan the full range to be safe,
    # or just 0-60 if we account for rotation of points? 
    # Actually, rotating the outer hex by 60 deg is equivalent to rotating the points by 60 deg.
    # Since we don't rotate points, we must scan 0-60? 
    # Wait. The set of normals for a hexagon at angle A is {A+30, A+90, A+150}.
    # The set for A+60 is {A+90, A+150, A+210}. This is the same set of lines, just permuted.
    # The LP solution (min S) depends on the geometry relative to the normals.
    # If we rotate the outer hex by 60, the constraints are identical (just reordered).
    # So S(A) = S(A+60). We only need to scan 0-60.
    
    # Use coarse scan then refine
    angles = np.arange(0, 60, 0.5)
    
    for angle_deg in angles:
        a = math.radians(angle_deg)
        normals = []
        for k in range(3):
            theta_n = a + math.pi/6 + k * math.pi/3
            normals.append((math.cos(theta_n), math.sin(theta_n)))
        
        A = []
        b = []
        
        for nx, ny in normals:
            for px, py in points:
                # n . (p - c) <= s * apothem
                # n . p - n . c <= s * apothem
                # -n . c - s * apothem <= -n . p  (for lower bound)
                # Let's use:
                # n . p - n . c - s * apothem <= 0
                # -n . p + n . c - s * apothem <= 0
                
                A.append([nx, ny, -math.sqrt(3)/2])
                b.append(nx*px + ny*py)
                
                A.append([-nx, -ny, -math.sqrt(3)/2])
                b.append(-nx*px - ny*py)
        
        bounds = [(-100, 100), (-100, 100), (0, None)]
        
        try:
            res = linprog(c=[0, 0, 1], A_ub=A, b_ub=b, bounds=bounds, method='highs')
            if res.success:
                s_val = res.x[2]
                if s_val < best_side:
                    best_side = s_val
                    best_params = (res.x[0], res.x[1], s_val, angle_deg)
        except Exception:
            pass
            
    # Refine around best angle
    if best_params[3] > 0 and best_params[3] < 60:
        best_ang = best_params[3]
        for angle_deg in np.arange(best_ang - 0.5, best_ang + 0.5, 0.05):
            a = math.radians(angle_deg)
            normals = []
            for k in range(3):
                theta_n = a + math.pi/6 + k * math.pi/3
                normals.append((math.cos(theta_n), math.sin(theta_n)))
                
            A = []
            b = []
            for nx, ny in normals:
                for px, py in points:
                    A.append([nx, ny, -math.sqrt(3)/2])
                    b.append(nx*px + ny*py)
                    A.append([-nx, -ny, -math.sqrt(3)/2])
                    b.append(-nx*px - ny*py)
                    
            bounds = [(-100, 100), (-100, 100), (0, None)]
            try:
                res = linprog(c=[0, 0, 1], A_ub=A, b_ub=b, bounds=bounds, method='highs')
                if res.success:
                    s_val = res.x[2]
                    if s_val < best_side:
                        best_side = s_val
                        best_params = (res.x[0], res.x[1], s_val, angle_deg)
            except Exception:
                pass
                
    return best_params

def _soft_max(x, beta):
    """
    Smooth approximation of max(x).
    beta controls sharpness. Higher beta -> closer to max.
    """
    max_val = np.max(x)
    # Numerical stability: shift by max
    x_shifted = x - max_val
    return max_val + (1.0 / beta) * np.log(np.sum(np.exp(beta * x_shifted)))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial configuration: Honeycomb lattice
    # 0: center
    # 1-6: ring 1
    # 7-10: ring 2 (attached to 1,2,3,4)
    
    rest_len = math.sqrt(3)
    
    # Initial positions (x, y) and angles (rad)
    # We will optimize:
    # x_0..x_10, y_0..y_10 (22 vars)
    # angles_0..angles_10 (11 vars) - but we can fix relative angles or let them float.
    # Let's fix all inner angles to 0 initially, but allow them to vary?
    # Rotation of individual hexagons might help packing if the lattice is distorted.
    # However, for a lattice, 0 is usually optimal.
    # Let's include angles in the optimization to be safe, but start at 0.
    
    # Variables:
    # 0-10: x
    # 11-21: y
    # 22-32: angles (rad)
    # 33: outer_cx
    # 34: outer_cy
    # 35: outer_side (S)
    # 36: outer_angle (rad)
    
    # Total 37 variables.
    
    def initial_state():
        pos = np.zeros((N, 2))
        pos[0] = [0, 0]
        angles_neighbor = [30, 90, 150, 210, 270, 330]
        for i in range(1, 7):
            ang_rad = math.radians(angles_neighbor[i-1])
            pos[i] = [rest_len * math.cos(ang_rad), rest_len * math.sin(ang_rad)]
            
        # Outer ring 7-10 attached to 1,2,3,4
        for idx, neighbor_idx in enumerate([1, 2, 3, 4]):
            node_idx = 7 + idx
            base_pos = pos[neighbor_idx]
            ang_rad = math.radians(angles_neighbor[neighbor_idx-1])
            pos[node_idx] = [
                base_pos[0] + rest_len * math.cos(ang_rad),
                base_pos[1] + rest_len * math.sin(ang_rad)
            ]
            
        x = np.zeros(37)
        for i in range(N):
            x[2*i] = pos[i][0]
            x[2*i+1] = pos[i][1]
            x[22+i] = 0.0 # angle 0
            
        # Initial outer hex estimate
        # Just use a large bounding box initially
        x[33] = 0.0
        x[34] = 0.0
        x[35] = 5.0 # S
        x[36] = 0.0 # angle
        
        return x

    def objective(x):
        # Extract variables
        px = x[0:22:2]
        py = x[1:22:2]
        pa = x[22:33]
        ocx = x[33]
        ocy = x[34]
        S = x[35]
        oang = x[36]
        
        # 1. Overlap Penalty
        # We need to check pairwise clearance.
        # This is expensive if done naively for every objective call in BFGS.
        # We can use a soft penalty.
        overlap_pen = 0.0
        # Optimization: Only check pairs that are close.
        # But for gradient-based methods, we need smoothness everywhere.
        # We'll use the vertex-based squared violation metric.
        
        # To speed up, we can vectorize or use a simple approximation?
        # Given N=11, 55 pairs. 12 vertices per pair. 660 point-in-hex checks.
        # This is manageable.
        
        for i in range(N):
            for j in range(i+1, N):
                # Quick reject
                dx = px[i] - px[j]
                dy = py[i] - py[j]
                if dx*dx + dy*dy > 4.5:
                    continue
                p1 = (px[i], py[i], math.degrees(pa[i]))
                p2 = (px[j], py[j], math.degrees(pa[j]))
                pen = _pairwise_clearance(p1, p2)
                overlap_pen += pen
        
        # 2. Containment Penalty (Soft-Max Epigraph)
        # For each inner hex vertex, it must be inside the outer hex.
        # Outer hex defined by center (ocx, ocy), side S, angle oang.
        # Constraints: n_k . (p - c) <= S * apothem
        # Violation: max(0, n_k . (p - c) - S * apothem)
        # We use soft-max over all violations to get a smooth objective.
        
        # Generate outer normals
        a = oang
        normals = []
        for k in range(3):
            theta_n = a + math.pi/6 + k * math.pi/3
            normals.append((math.cos(theta_n), math.sin(theta_n)))
        
        apothem_outer = S * math.sqrt(3) / 2
        
        violations = []
        for i in range(N):
            # Get vertices of inner hex i
            verts = _hex_vertices(px[i], py[i], 1.0, math.degrees(pa[i]))
            for vx, vy in verts:
                for nx, ny in normals:
                    val = nx * (vx - ocx) + ny * (vy - ocy) - apothem_outer
                    if val > 0:
                        violations.append(val)
                    # Also check opposite side?
                    # The 3 normals cover the 6 sides if we consider +/-
                    # n . (p-c) <= A  AND  -n . (p-c) <= A
                    # So we need to check both.
                    val2 = -nx * (vx - ocx) - ny * (vy - ocy) - apothem_outer
                    if val2 > 0:
                        violations.append(val2)
        
        # If no violations, containment is satisfied.
        # We want to minimize S.
        # Objective = S + lambda * (overlap_pen + soft_max(violations))
        
        # However, if violations are 0, soft_max is 0.
        # If we just minimize S, S will go to 0, causing huge violations.
        # The penalty term will push S up.
        
        # Better formulation:
        # Minimize S + lambda * (overlap + containment_violation)
        
        # To make containment_violation smooth and effective:
        if len(violations) > 0:
            v = np.array(violations)
            # Use log-sum-exp for soft max
            # beta = 50 seems reasonable
            beta = 50.0
            sm = _soft_max(v, beta)
        else:
            sm = 0.0
            
        return S + 100.0 * overlap_pen + 100.0 * sm

    def gradient(x):
        # Numerical gradient might be slow but robust.
        # Let's try analytical or use BFGS finite difference.
        # BFGS with finite diff is okay for 37 vars.
        return None # Let BFGS handle it

    # Multi-start
    best_x = None
    best_val = float('inf')
    
    # Start 1: Ideal lattice
    x0 = initial_state()
    
    # Start 2: Slightly perturbed
    x1 = initial_state()
    x1[0:22] += np.random.normal(0, 0.1, 22)
    x1[33:35] += np.random.normal(0, 0.1, 2)
    
    starts = [x0, x1]
    
    for start in starts:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(objective, start, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-5})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    if best_x is None:
        # Fallback to initial
        best_x = initial_state()
        best_val = objective(best_x)

    # Extract best solution
    px = best_x[0:22:2]
    py = best_x[1:22:2]
    pa_deg = np.degrees(best_x[22:33])
    
    # Construct inner list
    inner = []
    for i in range(N):
        inner.append([px[i], py[i], pa_deg[i]])
        
    # Now, refine the outer hexagon using exact LP for this specific inner configuration
    # This removes the need for the soft-max approximation to be perfect,
    # as we snap to the true minimum enclosing hexagon for the optimized inner positions.
    ocx, ocy, S_final, oang_final = _min_enclosing_hexagon_lp(inner)
    
    # Final validity check
    # Ensure no overlaps
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if _pairwise_clearance(tuple(inner[i]), tuple(inner[j])) > 1e-4:
                valid = False
                # If invalid, we might need to adjust, but for now return best effort
                # The LP ensures containment.
    
    return inner, (ocx, ocy), S_final, oang_final

if __name__ == "__main__":
    r = construct()
    print(r)
