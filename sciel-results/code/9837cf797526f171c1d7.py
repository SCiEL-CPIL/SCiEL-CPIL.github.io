import numpy as np
import time
import math
import random
from scipy.optimize import minimize, differential_evolution

N = 12
DEADLINE = 142.0  # Safety margin below 150s kill

def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with rotation rot_deg."""
    angles = np.radians(np.array([0, 60, 120, 180, 240, 330])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def sat_overlap(v1, v2):
    """
    Returns separation distance if not overlapping (positive), or negative penetration depth if overlapping.
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
    Exact minimum side length of regular hexagon centered at origin with given rotation
    that encloses all points.
    """
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def smooth_overlap_penalty(params, beta=30.0):
    """
    Compute smooth overlap penalty using log-sum-exp for numerical stability.
    Returns a smooth approximation of the sum of squared penetrations.
    """
    xys_rots = params[:3*N].reshape(N, 3)
    centers = xys_rots[:, :2]
    
    vertices_list = []
    for i in range(N):
        cx, cy, rot = xys_rots[i]
        v = get_hex_vertices(cx, cy, 1.0, rot)
        vertices_list.append(v)
    
    penetrations = []
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < 2.0:
                d_sat = sat_overlap(vertices_list[i], vertices_list[j])
                if d_sat < 0:
                    penetrations.append(-d_sat)
    
    if len(penetrations) == 0:
        return 0.0
    
    p_arr = np.array(penetrations)
    scaled = 2.0 * p_arr / beta
    max_s = np.max(scaled)
    if max_s > 500:
        return np.sum(p_arr**2)
    lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
    return beta * lse

def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def objective(params, use_smooth=True):
        if time.time() - start_time > DEADLINE - 5:
            return 1e6
        
        xys_rots = params[:3*N].reshape(N, 3)
        outer_rot = params[3*N]
        
        penalty = smooth_overlap_penalty(params, beta=30.0)
        
        all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
        
        if use_smooth:
            # Smooth outer hex side
            angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot)
            normals = np.column_stack((np.cos(angles), np.sin(angles)))
            proj = all_points @ normals.T
            beta_lse = 5.0
            scaled_proj = proj / beta_lse
            max_scaled = np.max(scaled_proj, axis=0)
            sum_exp = np.sum(np.exp(scaled_proj - max_scaled), axis=0)
            max_proj_k = max_scaled * beta_lse + beta_lse * np.log(sum_exp)
            # Also need abs
            scaled_abs = np.abs(proj) / beta_lse
            max_scaled_abs = np.max(scaled_abs, axis=0)
            sum_exp_abs = np.sum(np.exp(scaled_abs - max_scaled_abs), axis=0)
            max_proj_k_abs = max_scaled_abs * beta_lse + beta_lse * np.log(sum_exp_abs)
            max_proj_k = np.maximum(max_proj_k, max_proj_k_abs)
            
            scaled_max_proj = max_proj_k / beta_lse
            max_scaled_k = np.max(scaled_max_proj)
            sum_exp_k = np.sum(np.exp(scaled_max_proj - max_scaled_k))
            final_max_proj = max_scaled_k * beta_lse + beta_lse * np.log(sum_exp_k)
            L = (2.0 / np.sqrt(3.0)) * final_max_proj
        else:
            L = outer_hex_side(all_points, outer_rot)
            
        return L + 100.0 * penalty
    
    def make_hex_grid(rows, cols, rot_offset=0.0, gap=0.0):
        """Create a hex grid of unit hexagons with optional gap."""
        pts = []
        sqrt3 = np.sqrt(3)
        spacing_x = sqrt3 + gap
        spacing_y = 1.5 + gap * 0.5
        for r in range(rows):
            y = r * spacing_y
            for c in range(cols):
                x = c * spacing_x
                if r % 2 == 1:
                    x += spacing_x / 2.0
                pts.append([x, y, rot_offset])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    def make_ring(gap=0.0):
        """1 center + 6 ring + 5 outer with gap."""
        pts = []
        sqrt3 = np.sqrt(3)
        d1 = sqrt3 + gap
        d2 = 2 * sqrt3 + gap
        pts.append([0, 0, 0.0])
        for k in range(6):
            ang = np.radians(60 * k)
            pts.append([d1 * np.cos(ang), d1 * np.sin(ang), 0.0])
        for k in range(5):
            ang = np.radians(60 * k + 30)
            pts.append([d2 * np.cos(ang), d2 * np.sin(ang), 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    def make_compact_cluster(gap=0.0):
        """Compact 3x4 honeycomb cluster."""
        sqrt3 = np.sqrt(3)
        pts = []
        for r in range(3):
            for c in range(4):
                x = c * sqrt3 + (r % 2) * sqrt3 / 2
                y = r * 1.5
                pts.append([x, y, 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
        
    def is_valid(params, tol=1e-6):
        """Check if configuration has no overlaps."""
        xys_rots = params[:3*N].reshape(N, 3)
        vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
        
        for i in range(N):
            for j in range(i+1, N):
                d = sat_overlap(vertices_list[i], vertices_list[j])
                if d < -tol:
                    return False
        return True
    
    def get_L(params):
        """Compute outer side length for a valid configuration."""
        xys_rots = params[:3*N].reshape(N, 3)
        outer_rot = params[3*N]
        all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
        return outer_hex_side(all_points, outer_rot)
    
    def polish(params, max_iter=200):
        """High-precision polish pass with tight tolerances."""
        try:
            res = minimize(
                lambda p: objective(p, use_smooth=False),
                params,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': max_iter * 3}
            )
            return res.x
        except:
            return params

    def golden_section_search(func, a, b, tol=1e-6, max_iter=100):
        """Golden section search for minimum of unimodal function on [a, b]."""
        gr = (np.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = func(c)
        fd = func(d)
        
        for _ in range(max_iter):
            if b - a < tol:
                break
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = func(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = func(d)
        
        return (a + b) / 2
    
    def optimize_rigid_cluster(base_cfg):
        """
        Optimize the rigid cluster by rotating the whole body and finding the best outer hex rotation.
        This is a fast 1D+1D search.
        """
        if time_left() < 5:
            return None, float('inf'), 0.0
        
        best_L = float('inf')
        best_rot = 0.0
        best_orot = 0.0
        best_cfg = None
        
        # Coarse grid search over rotation (0 to 60 degrees)
        for rot_deg in np.linspace(0, 60, 30):
            if time_left() < 3:
                break
            
            cfg = base_cfg.copy()
            rad = np.radians(rot_deg)
            cos_r, sin_r = np.cos(rad), np.sin(rad)
            x_new = cfg[:, 0] * cos_r - cfg[:, 1] * sin_r
            y_new = cfg[:, 0] * sin_r + cfg[:, 1] * cos_r
            cfg[:, 0] = x_new
            cfg[:, 1] = y_new
            cfg[:, 2] = (cfg[:, 2] + rot_deg) % 360
            
            all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in cfg])
            
            # Find best outer rotation for this configuration
            for orot in np.linspace(0, 60, 15):
                L = outer_hex_side(all_points, orot)
                if L < best_L:
                    best_L = L
                    best_rot = rot_deg
                    best_orot = orot
                    best_cfg = cfg.copy()
        
        if best_cfg is None:
            return None, float('inf'), 0.0
        
        # Fine golden section search around best rotation
        a = max(0, best_rot - 2)
        b = min(60, best_rot + 2)
        
        def rot_func(r):
            cfg = base_cfg.copy()
            rad = np.radians(r)
            cos_r, sin_r = np.cos(rad), np.sin(rad)
            x_new = cfg[:, 0] * cos_r - cfg[:, 1] * sin_r
            y_new = cfg[:, 0] * sin_r + cfg[:, 1] * cos_r
            cfg[:, 0] = x_new
            cfg[:, 1] = y_new
            cfg[:, 2] = (cfg[:, 2] + r) % 360
            
            all_points = np.vstack([get_hex_vertices(x, y, 1.0, r_) for x, y, r_ in cfg])
            
            best_outer_L = float('inf')
            for orot in np.linspace(0, 60, 15):
                L = outer_hex_side(all_points, orot)
                if L < best_outer_L:
                    best_outer_L = L
            return best_outer_L
        
        opt_rot = golden_section_search(rot_func, a, b, tol=0.005, max_iter=40)
        
        # Apply final rotation and find best outer rotation
        cfg = base_cfg.copy()
        rad = np.radians(opt_rot)
        cos_r, sin_r = np.cos(rad), np.sin(rad)
        x_new = cfg[:, 0] * cos_r - cfg[:, 1] * sin_r
        y_new = cfg[:, 0] * sin_r + cfg[:, 1] * cos_r
        cfg[:, 0] = x_new
        cfg[:, 1] = y_new
        cfg[:, 2] = (cfg[:, 2] + opt_rot) % 360
        
        all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in cfg])
        final_L = float('inf')
        final_orot = 0.0
        for orot in np.linspace(0, 60, 30):
            L = outer_hex_side(all_points, orot)
            if L < final_L:
                final_L = L
                final_orot = orot
        
        return cfg, final_L, final_orot

    def simulated_annealing(init_params, max_iter=5000, T0=1.0, T_end=0.01):
        """Simulated annealing for exploration."""
        if time_left() < 5:
            return init_params
        
        current = init_params.copy()
        current_val = objective(current, use_smooth=True)
        best = current.copy()
        best_val = current_val
        
        for i in range(max_iter):
            if time_left() < 5:
                break
            
            T = T0 * (T_end / T0) ** (i / max_iter)
            
            # Random perturbation
            new = current.copy()
            step_size = 0.1 * (1.0 + 0.5 * np.exp(-i / 1000))
            new[:3*N] += np.random.normal(0, step_size, 3*N)
            new[3*N] = (new[3*N] + np.random.normal(0, 0.5)) % 60
            
            new_val = objective(new, use_smooth=True)
            
            if new_val < current_val or np.random.random() < np.exp(-(new_val - current_val) / max(T, 1e-10)):
                current = new
                current_val = new_val
                if new_val < best_val:
                    best = new.copy()
                    best_val = new_val
        
        return best

    best_L = float('inf')
    best_params = None
    
    # Generate diverse initial configurations
    gap = 0.02
    configs = []
    configs.append(make_hex_grid(4, 3, gap=gap))
    configs.append(make_hex_grid(3, 4, gap=gap))
    configs.append(make_hex_grid(2, 6, gap=gap))
    configs.append(make_hex_grid(6, 2, gap=gap))
    configs.append(make_ring(gap=gap))
    configs.append(make_compact_cluster(gap=gap))
    
    # Add rotated variants
    for base_cfg in configs[:4]:
        rotated = base_cfg.copy()
        rotated[:, 2] = 30.0
        configs.append(rotated)
    
    # Add random perturbations
    for base_cfg in configs[:3]:
        perturbed = base_cfg.copy()
        perturbed[:, :2] += np.random.normal(0, 0.1, (N, 2))
        configs.append(perturbed)
        
    outer_rots = [0.0, 30.0, 15.0, 45.0, 75.0]
    
    for cfg in configs:
        if time_left() < 10:
            break
        
        # 1. Rigid Body Optimization (Fast, low-dimensional)
        if time_left() > 15:
            opt_cfg, opt_L, opt_orot = optimize_rigid_cluster(cfg)
            if opt_cfg is not None and opt_L < best_L:
                # Verify validity
                test_params = np.concatenate((opt_cfg.flatten(), [opt_orot]))
                if is_valid(test_params):
                    best_L = opt_L
                    best_params = test_params.copy()
        
        # 2. Local Optimization with Smooth Objective
        for orot in outer_rots:
            if time_left() < 5:
                break
            
            init_params = np.concatenate((cfg.flatten(), [orot]))
            
            # Check initial validity
            if is_valid(init_params):
                L_val = get_L(init_params)
                if L_val < best_L:
                    best_L = L_val
                    best_params = init_params.copy()
            
            # Phase 1: L-BFGS-B optimization with smooth objective
            if time_left() > 8:
                try:
                    res = minimize(
                        lambda p: objective(p, use_smooth=True),
                        init_params,
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 600}
                    )
                    
                    if is_valid(res.x):
                        L_val = get_L(res.x)
                        if L_val < best_L:
                            best_L = L_val
                            best_params = res.x.copy()
                            
                except Exception:
                    pass
            
            # Phase 2: Polish the best result so far (using exact objective)
            if best_params is not None and time_left() > 15:
                polished = polish(best_params, max_iter=100)
                if is_valid(polished):
                    L_val = get_L(polished)
                    if L_val < best_L:
                        best_L = L_val
                        best_params = polished.copy()
    
    # Simulated annealing exploration on the best configuration
    if best_params is not None and time_left() > 20:
        sa_result = simulated_annealing(best_params, max_iter=3000, T0=0.5, T_end=0.005)
        if is_valid(sa_result):
            L_val = get_L(sa_result)
            if L_val < best_L:
                best_L = L_val
                best_params = sa_result.copy()
        
        # Polish after SA
        if best_params is not None and time_left() > 10:
            polished = polish(best_params, max_iter=200)
            if is_valid(polished):
                L_val = get_L(polished)
                if L_val < best_L:
                    best_L = L_val
                    best_params = polished.copy()
    
    # Final polish pass with remaining time
    if best_params is not None and time_left() > 10:
        polished = polish(best_params, max_iter=300)
        if is_valid(polished):
            L_val = get_L(polished)
            if L_val < best_L:
                best_L = L_val
                best_params = polished.copy()
    
    # Try rigid cluster optimization on best configuration as a final refinement
    if best_params is not None and time_left() > 15:
        xys_rots = best_params[:3*N].reshape(N, 3)
        rigid_cfg, rigid_L, rigid_orot = optimize_rigid_cluster(xys_rots)
        if rigid_cfg is not None and rigid_L < best_L:
            test_params = np.concatenate((rigid_cfg.flatten(), [rigid_orot]))
            if is_valid(test_params):
                best_L = rigid_L
                best_params = test_params.copy()
    
    if best_params is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Extract final result
    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_rot = best_params[3*N]
    
    inner = []
    for x, y, rot in xys_rots:
        inner.append([x, y, rot])
    
    return inner, (0.0, 0.0), best_L, outer_rot
