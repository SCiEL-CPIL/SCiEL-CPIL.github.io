import numpy as np
import time
import math
import random
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # Safety margin below 150s kill

def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with rotation rot_deg."""
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def sat_overlap(v1, v2):
    """
    Returns penetration depth if overlapping (negative), or separation distance if not.
    Uses Separating Axis Theorem for convex polygons.
    """
    axes = []
    for v in (v1, v2):
        for i in range(len(v)):
            p1 = v[i]
            p2 = v[(i+1) % len(v)]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                axes.append(n)
    
    min_overlap = float('inf')
    for n in axes:
        proj1 = v1 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        proj2 = v2 @ n
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap  # positive = separated
        
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    
    return -min_overlap  # negative = overlapping

def outer_hex_side(points, outer_rot_deg):
    """
    Minimum side length of regular hexagon centered at origin with given rotation
    that encloses all points.
    """
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    # Apothem = (sqrt(3)/2) * side
    # max_proj is the max distance from center to a side (apothem)
    # So side = 2 * apothem / sqrt(3)
    return (2.0 / np.sqrt(3.0)) * max_proj

def smooth_overlap_penalty(xys_rots, beta=50.0):
    """
    Compute smooth overlap penalty using log-sum-exp for numerical stability.
    Returns a smooth approximation of the sum of squared penetrations.
    """
    centers = xys_rots[:, :2]
    
    # Precompute vertices for all hexagons
    vertices_list = []
    for i in range(N):
        cx, cy, rot = xys_rots[i]
        v = get_hex_vertices(cx, cy, 1.0, rot)
        vertices_list.append(v)
    
    # Compute pairwise penetrations
    penetrations = []
    for i in range(N):
        for j in range(i+1, N):
            # Quick distance check
            d = np.linalg.norm(centers[i] - centers[j])
            if d < 2.0:  # Only compute SAT if close enough
                d_sat = sat_overlap(vertices_list[i], vertices_list[j])
                if d_sat < 0:
                    penetrations.append(-d_sat)
    
    if len(penetrations) == 0:
        return 0.0
    
    # Use log-sum-exp for smooth approximation of sum of squares
    p_arr = np.array(penetrations)
    scaled = 2.0 * p_arr / beta
    max_s = np.max(scaled)
    if max_s > 500:  # Avoid overflow
        return np.sum(p_arr**2)
    lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
    return beta * lse

def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def is_valid(xys_rots, tol=1e-6):
        """Check if configuration has no overlaps."""
        vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
        
        for i in range(N):
            for j in range(i+1, N):
                d = sat_overlap(vertices_list[i], vertices_list[j])
                if d < -tol:
                    return False
        return True
    
    def compute_L(xys_rots, outer_rot):
        """Compute outer side length for a valid configuration."""
        all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
        return outer_hex_side(all_points, outer_rot)
    
    def make_honeycomb_3x4():
        """
        Create a 3x4 honeycomb arrangement of unit hexagons.
        This is the most compact arrangement for 12 hexagons.
        """
        pts = []
        sqrt3 = np.sqrt(3)
        spacing_x = sqrt3
        spacing_y = 1.5
        
        for r in range(3):
            for c in range(4):
                x = c * spacing_x
                y = r * spacing_y
                if r % 2 == 1:
                    x += spacing_x / 2.0
                # Rotate each hexagon to align with honeycomb
                rot = 0.0 if r % 2 == 0 else 0.0
                pts.append([x, y, rot])
        
        pts = np.array(pts)
        # Center the configuration
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    def make_honeycomb_4x3():
        """
        Create a 4x3 honeycomb arrangement of unit hexagons.
        """
        pts = []
        sqrt3 = np.sqrt(3)
        spacing_x = sqrt3
        spacing_y = 1.5
        
        for r in range(4):
            for c in range(3):
                x = c * spacing_x
                y = r * spacing_y
                if r % 2 == 1:
                    x += spacing_x / 2.0
                rot = 0.0
                pts.append([x, y, rot])
        
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    def make_ring_12():
        """12 hexagons in a ring around center."""
        pts = []
        R = 2.5
        for i in range(N):
            theta = 2 * np.pi * i / N
            x = R * np.cos(theta)
            y = R * np.sin(theta)
            angle_deg = math.degrees(theta)
            pts.append([x, y, angle_deg])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    def golden_section_search(f, a, b, tol=1e-8, max_iter=100):
        """
        Golden section search for minimum of f on [a, b].
        """
        gr = (np.sqrt(5.0) - 1.0) / 2.0  # golden ratio conjugate
        
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = f(c)
        fd = f(d)
        
        for _ in range(max_iter):
            if time.time() - start_time > DEADLINE - 10:
                break
            if abs(b - a) < tol:
                break
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = f(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = f(d)
        
        return (a + b) / 2.0
    
    best_L = float('inf')
    best_params = None
    
    # Generate candidate internal configurations
    configs = []
    configs.append(('honeycomb_3x4', make_honeycomb_3x4()))
    configs.append(('honeycomb_4x3', make_honeycomb_4x3()))
    configs.append(('ring_12', make_ring_12()))
    
    # For each configuration, do golden-section search over global rotation
    for config_name, cfg in configs:
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # Check validity of base configuration
        if not is_valid(cfg):
            continue
        
        # Define objective for golden section: outer side length as function of rotation
        def obj_for_rotation(rot_deg):
            # Rotate all hexagons by rot_deg
            rotated_cfg = cfg.copy()
            for i in range(N):
                # Rotate position
                theta = np.radians(rot_deg)
                x, y = rotated_cfg[i][0], rotated_cfg[i][1]
                new_x = x * np.cos(theta) - y * np.sin(theta)
                new_y = x * np.sin(theta) + y * np.cos(theta)
                rotated_cfg[i][0] = new_x
                rotated_cfg[i][1] = new_y
                # Also rotate the hexagon itself
                rotated_cfg[i][2] = cfg[i][2] + rot_deg
            
            # The outer hexagon rotation is 0 (we're optimizing the body rotation)
            return compute_L(rotated_cfg, 0.0)
        
        # Golden section search over [0, 60) degrees (hexagonal symmetry)
        optimal_rot = golden_section_search(obj_for_rotation, 0.0, 60.0, tol=1e-6, max_iter=80)
        
        # Compute final configuration
        theta = np.radians(optimal_rot)
        final_cfg = cfg.copy()
        for i in range(N):
            x, y = final_cfg[i][0], final_cfg[i][1]
            new_x = x * np.cos(theta) - y * np.sin(theta)
            new_y = x * np.sin(theta) + y * np.cos(theta)
            final_cfg[i][0] = new_x
            final_cfg[i][1] = new_y
            final_cfg[i][2] = cfg[i][2] + optimal_rot
        
        # Verify validity
        if is_valid(final_cfg):
            L_val = compute_L(final_cfg, 0.0)
            if L_val < best_L:
                best_L = L_val
                best_params = np.concatenate((final_cfg.flatten(), [0.0])).copy()
    
    # Also try with outer hexagon rotation optimization
    if best_params is not None and time.time() - start_time < DEADLINE - 20:
        cfg = best_params[:3*N].reshape(N, 3).copy()
        
        def obj_outer_rot(rot_deg):
            return compute_L(cfg, rot_deg)
        
        optimal_outer_rot = golden_section_search(obj_outer_rot, 0.0, 60.0, tol=1e-6, max_iter=50)
        L_val = compute_L(cfg, optimal_outer_rot)
        if L_val < best_L:
            best_L = L_val
            best_params = np.concatenate((cfg.flatten(), [optimal_outer_rot])).copy()
    
    # Local optimization to fine-tune positions
    if best_params is not None and time.time() - start_time < DEADLINE - 15:
        def objective(params):
            if time.time() - start_time > DEADLINE - 5:
                return 1e6
            
            xys_rots = params[:3*N].reshape(N, 3)
            outer_rot = params[3*N]
            
            # Re-center
            centers = xys_rots[:, :2]
            mean_center = np.mean(centers, axis=0)
            xys_rots_rc = xys_rots.copy()
            xys_rots_rc[:, :2] -= mean_center
            
            penalty = smooth_overlap_penalty(xys_rots_rc, beta=50.0)
            
            all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots_rc])
            L = outer_hex_side(all_points, outer_rot)
            
            return L + 100.0 * penalty
        
        try:
            res = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                options={
                    'maxiter': 300,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'maxfun': 800
                }
            )
            
            if is_valid(res.x[:3*N].reshape(N, 3)):
                L_val = compute_L(res.x[:3*N].reshape(N, 3), res.x[3*N])
                if L_val < best_L:
                    best_L = L_val
                    best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        # Fallback: use honeycomb_3x4 with no rotation
        cfg = make_honeycomb_3x4()
        if is_valid(cfg):
            best_L = compute_L(cfg, 0.0)
            best_params = np.concatenate((cfg.flatten(), [0.0])).copy()
        else:
            # Absolute fallback
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 40.0, 0.0
    
    # Extract final result
    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_rot = best_params[3*N]
    
    inner = []
    for x, y, rot in xys_rots:
        inner.append([x, y, rot])
    
    return inner, (0.0, 0.0), best_L, outer_rot
