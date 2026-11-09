"""Direct construction of 12 unit hexagons inside a regular hexagon, minimizing outer side."""
N = 12

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

def _hex_verts(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _hex_edge_normals(angle_deg):
    """Unit normals to the 6 edges of a hexagon with given rotation."""
    a = np.radians(angle_deg)
    normals = []
    for i in range(6):
        # Edge from vertex i to vertex i+1; normal points inward
        v1 = np.array([np.cos(a + np.radians(60 * i)), np.sin(a + np.radians(60 * i))])
        v2 = np.array([np.cos(a + np.radians(60 * (i + 1))), np.sin(a + np.radians(60 * (i + 1)))])
        edge = v2 - v1
        # Inward normal: rotate edge 90 deg clockwise, normalize
        n = np.array([edge[1], -edge[0]])
        n = n / np.linalg.norm(n)
        normals.append(n)
    return np.array(normals)

def _point_in_hex_dist(px, py, cx, cy, side, angle_deg):
    """Return signed distance: positive if inside, negative if outside.
    Uses the 6 half-plane constraints of the hexagon."""
    a = np.radians(angle_deg)
    # Hexagon vertices
    verts = _hex_verts(cx, cy, side, angle_deg)
    # For each edge, compute distance to the edge line (positive = inside)
    min_dist = float('inf')
    for i in range(6):
        v1 = verts[i]
        v2 = verts[(i + 1) % 6]
        edge = v2 - v1
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-12:
            continue
        # Inward normal
        n = np.array([edge[1], -edge[0]]) / edge_len
        # Distance from point to edge line, positive if on inward side
        d = np.dot(np.array([px, py]) - v1, n)
        min_dist = min(min_dist, d)
    return min_dist

def _sat_overlap_depth_smooth(h1, h2, beta=50.0):
    """Smooth approximation of minimum penetration depth between two convex hexagons.
    Returns 0 if disjoint, positive if overlapping. Uses log-sum-exp for smooth min."""
    axes = []
    for poly in [h1, h2]:
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            norm = np.linalg.norm(edge)
            if norm < 1e-10:
                continue
            normal = np.array([-edge[1], edge[0]]) / norm
            axes.append(normal)
    
    axes = np.array(axes)
    depths = []
    for axis in axes:
        proj1 = np.dot(h1, axis)
        proj2 = np.dot(h2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Overlap on this axis
        overlap = min(max1, max2) - max(min1, min2)
        if overlap <= 0:
            return 0.0  # Disjoint
        depths.append(overlap)
    
    if not depths:
        return 0.0
    
    # Smooth minimum via log-sum-exp
    depths = np.array(depths)
    max_d = np.max(depths)
    if max_d < 1e-12:
        return 0.0
    log_sum = np.log(np.sum(np.exp(-beta * depths))) - beta * max_d
    smooth_min = max_d - np.log(len(depths)) / beta - log_sum / beta
    # Actually: soft_min = -1/beta * log(sum(exp(-beta * d_i)))
    smooth_min = -1.0 / beta * (np.log(np.sum(np.exp(-beta * depths))))
    return max(0.0, smooth_min)

def _total_penalty(params, outer_side, outer_center, outer_angle, beta_sat=30.0, beta_cont=30.0):
    """Total penalty: smooth containment + smooth pairwise disjointness."""
    # Unpack
    inner_verts_list = []
    for i in range(N):
        cx = params[3 * i]
        cy = params[3 * i + 1]
        ang = params[3 * i + 2]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    
    outer_verts = _hex_verts(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # Containment: all vertices of each inner hex must be inside outer hex
    cont_pen = 0.0
    for iv in inner_verts_list:
        for pt in iv:
            d = _point_in_hex_dist(pt[0], pt[1], outer_center[0], outer_center[1], outer_side, outer_angle)
            if d < 0:
                # Smooth penalty
                cont_pen += d * d  # Quadratic penalty for being outside
    
    # Disjointness: pairwise SAT
    disj_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth_smooth(inner_verts_list[i], inner_verts_list[j], beta_sat)
            if depth > 0:
                disj_pen += depth * depth
    
    return cont_pen + disj_pen

def _total_penalty_fast(params, outer_side, outer_center, outer_angle):
    """Faster penalty using vectorized containment and exact SAT (no smooth)."""
    inner_verts_list = []
    for i in range(N):
        cx = params[3 * i]
        cy = params[3 * i + 1]
        ang = params[3 * i + 2]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    
    outer_verts = _hex_verts(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # Containment
    cont_pen = 0.0
    for iv in inner_verts_list:
        for pt in iv:
            d = _point_in_hex_dist(pt[0], pt[1], outer_center[0], outer_center[1], outer_side, outer_angle)
            if d < 0:
                cont_pen += (-d)
    
    # Disjointness
    disj_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth_exact(inner_verts_list[i], inner_verts_list[j])
            if depth > 0:
                disj_pen += depth
    
    return cont_pen + disj_pen

def _sat_overlap_depth_exact(h1, h2):
    """Exact minimum penetration depth between two convex hexagons via SAT."""
    min_depth = float('inf')
    axes = []
    for poly in [h1, h2]:
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            norm = np.linalg.norm(edge)
            if norm < 1e-10:
                continue
            normal = np.array([-edge[1], edge[0]]) / norm
            axes.append(normal)
    
    for axis in axes:
        proj1 = np.dot(h1, axis)
        proj2 = np.dot(h2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 <= min2 or max2 <= min1:
            return 0.0
        
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_depth:
            min_depth = overlap
    
    return min_depth

def _build_config_3x4():
    """3 rows x 4 cols hexagonal lattice, centered."""
    positions = []
    for r in range(3):
        for c in range(4):
            x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
            y = r * np.sqrt(3)
            positions.append((x, y))
    arr = np.array(positions)
    centroid = arr.mean(axis=0)
    return (arr - centroid).tolist()

def _build_config_4x3():
    """4 rows x 3 cols hexagonal lattice, centered."""
    positions = []
    for r in range(4):
        for c in range(3):
            x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
            y = r * np.sqrt(3)
            positions.append((x, y))
    arr = np.array(positions)
    centroid = arr.mean(axis=0)
    return (arr - centroid).tolist()

def _build_config_ring():
    """1 center + 6 ring + 5 more, centered."""
    positions = [(0.0, 0.0)]
    for i in range(6):
        ang = np.radians(60 * i)
        positions.append((2.0 * np.cos(ang), 2.0 * np.sin(ang)))
    for i in range(6):
        ang = np.radians(60 * i + 30)
        positions.append((3.464 * np.cos(ang), 3.464 * np.sin(ang)))
    arr = np.array(positions[:12])
    centroid = arr.mean(axis=0)
    return (arr - centroid).tolist()

def _build_config_brick():
    """Brick wall 4x3 with tighter spacing."""
    positions = []
    for r in range(4):
        for c in range(3):
            x = c * 1.9 + (0.95 if r % 2 == 1 else 0.0)
            y = r * 1.65
            positions.append((x, y))
    arr = np.array(positions)
    centroid = arr.mean(axis=0)
    return (arr - centroid).tolist()

def _build_config_2x6():
    """2 rows x 6 cols."""
    positions = []
    for r in range(2):
        for c in range(6):
            x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
            y = r * np.sqrt(3)
            positions.append((x, y))
    arr = np.array(positions)
    centroid = arr.mean(axis=0)
    return (arr - centroid).tolist()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    configs = [_build_config_3x4(), _build_config_4x3(), _build_config_ring(),
               _build_config_brick(), _build_config_2x6()]
    
    best_score = float('inf')
    best_result = None
    
    # Phase 1: For each config and outer angle, optimize inner positions+angles
    # for a given outer side, then binary search on side.
    outer_angles = [0.0, 30.0]
    
    for outer_angle in outer_angles:
        if time.time() > deadline - 30.0:
            break
        
        outer_center = (0.0, 0.0)
        
        for config in configs:
            if time.time() > deadline - 25.0:
                break
            
            # Start with a reasonable outer side
            init_side = 5.0
            
            # First, optimize inner positions for this outer side
            init_params = []
            for (x, y) in config:
                init_params.extend([float(x), float(y), 0.0])
            init_params = np.array(init_params)
            
            def objective(p, s=init_side, oc=outer_center, oa=outer_angle):
                return _total_penalty(p, s, oc, oa)
            
            try:
                result = minimize(
                    objective,
                    init_params,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'maxfun': 3000, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                opt_params = result.x
                opt_val = result.fun
            except Exception:
                continue
            
            if opt_val > 1e-4:
                continue
            
            # Binary search for minimum side
            low = 3.0
            high = init_side
            current_params = opt_params.copy()
            current_side = init_side
            
            for _ in range(10):
                if time.time() > deadline - 15.0:
                    break
                mid = (low + high) / 2.0
                if mid <= low + 1e-6:
                    break
                
                def shrink_obj(p, s=mid, oc=outer_center, oa=outer_angle):
                    return _total_penalty(p, s, oc, oa)
                
                try:
                    shrink_result = minimize(
                        shrink_obj,
                        current_params,
                        method='L-BFGS-B',
                        options={'maxiter': 60, 'maxfun': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    if shrink_result.fun < 1e-6:
                        high = mid
                        current_params = shrink_result.x
                        current_side = mid
                    else:
                        low = mid
                except Exception:
                    low = mid
            
            if current_side < best_score:
                best_score = current_side
                best_result = (current_params, outer_center, current_side, outer_angle)
    
    # Phase 2: Joint optimization - optimize outer center and angle too
    if best_result is not None and time.time() < deadline - 20.0:
        opt_params, outer_center, outer_side, outer_angle = best_result
        
        # Joint objective: inner params + outer center + outer angle
        def joint_objective(params_joint, s=outer_side):
            # Unpack: first 3*N are inner, next 3 are outer (cx, cy, angle)
            inner_p = params_joint[:3 * N]
            ocx = params_joint[3 * N]
            ocy = params_joint[3 * N + 1]
            oang = params_joint[3 * N + 2]
            return _total_penalty(inner_p, s, (ocx, ocy), oang)
        
        joint_init = np.concatenate([opt_params, [outer_center[0], outer_center[1], outer_angle]])
        
        try:
            joint_result = minimize(
                joint_objective,
                joint_init,
                method='L-BFGS-B',
                options={'maxiter': 80, 'maxfun': 2500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if joint_result.fun < 1e-6:
                new_params = joint_result.x[:3 * N]
                new_oc = (float(joint_result.x[3 * N]), float(joint_result.x[3 * N + 1]))
                new_oa = float(joint_result.x[3 * N + 2])
                
                # Try to shrink further
                low = 3.0
                high = outer_side
                cp = new_params.copy()
                cs = outer_side
                co = new_oc
                ca = new_oa
                
                for _ in range(8):
                    if time.time() > deadline - 10.0:
                        break
                    mid = (low + high) / 2.0
                    if mid <= low + 1e-6:
                        break
                    
                    def shrink_joint(p, s=mid):
                        return joint_objective(p, s)
                    
                    try:
                        sr = minimize(
                            shrink_joint,
                            np.concatenate([cp, [co[0], co[1], ca]]),
                            method='L-BFGS-B',
                            options={'maxiter': 50, 'maxfun': 1500, 'ftol': 1e-14, 'gtol': 1e-10}
                        )
                        if sr.fun < 1e-6:
                            high = mid
                            cp = sr.x[:3 * N]
                            co = (float(sr.x[3 * N]), float(sr.x[3 * N + 1]))
                            ca = float(sr.x[3 * N + 2])
                            cs = mid
                        else:
                            low = mid
                    except Exception:
                        low = mid
                
                if cs < best_score:
                    best_score = cs
                    best_result = (cp, co, cs, ca)
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if best_result is not None and time.time() < deadline - 5.0:
        opt_params, outer_center, outer_side, outer_angle = best_result
        
        # Tighter tolerance polish
        def final_obj(p, s=outer_side, oc=outer_center, oa=outer_angle):
            return _total_penalty(p, s, oc, oa)
        
        try:
            final_result = minimize(
                final_obj,
                opt_params,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 3000, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            if final_result.fun < 1e-8:
                opt_params = final_result.x
        except Exception:
            pass
        
        best_result = (opt_params, outer_center, outer_side, outer_angle)
    
    # Fallback
    if best_result is None:
        inner = []
        pos = _build_config_3x4()
        for (x, y) in pos:
            inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 5.5, 0.0
    
    opt_params, outer_center, outer_side, outer_angle = best_result
    
    inner = []
    for i in range(N):
        x = float(opt_params[3 * i])
        y = float(opt_params[3 * i + 1])
        ang = float(opt_params[3 * i + 2])
        inner.append([x, y, ang])
    
    # Final validation
    inner_verts_list = []
    for i in range(N):
        cx, cy, ang = opt_params[3 * i], opt_params[3 * i + 1], opt_params[3 * i + 2]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    outer_verts = _hex_verts(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    cont_pen = 0.0
    for iv in inner_verts_list:
        for pt in iv:
            d = _point_in_hex_dist(pt[0], pt[1], outer_center[0], outer_center[1], outer_side, outer_angle)
            if d < 0:
                cont_pen += (-d)
    
    disj_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = _sat_overlap_depth_exact(inner_verts_list[i], inner_verts_list[j])
            if d > 0:
                disj_pen += d
    
    if cont_pen > 1e-4 or disj_pen > 1e-4:
        inner = []
        pos = _build_config_3x4()
        for (x, y) in pos:
            inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 5.5, 0.0
    
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)

# EVOLVE-BLOCK-END
