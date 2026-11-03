import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    def hex_vertices(cx, cy, angle_deg):
        """Return 6 vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + math.cos(a)
            y = cy + math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    def sat_gap(v1, v2):
        """
        SAT gap between two convex hexagons.
        Returns positive if separated, negative if overlapping.
        """
        max_gap = -float('inf')
        for poly in [v1, v2]:
            for i in range(6):
                e = poly[(i+1)%6] - poly[i]
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-9:
                    continue
                n = n / norm
                proj1 = np.dot(v1, n)
                proj2 = np.dot(v2, n)
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                if max1 < min2:
                    gap = min2 - max1
                elif max2 < min1:
                    gap = min1 - max2
                else:
                    gap = min(max1, max2) - max(min1, min2)
                if gap > max_gap:
                    max_gap = gap
        return max_gap

    def is_valid(centers, angles):
        """Check all pairs are non-overlapping."""
        verts_list = [hex_vertices(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                gap = sat_gap(verts_list[i], verts_list[j])
                if gap < -1e-4:
                    return False
        return True

    def compute_outer_side(centers, angles, outer_cx, outer_cy, outer_angle_deg):
        """
        Minimum side length of outer hexagon centered at (outer_cx, outer_cy)
        with rotation outer_angle_deg to contain all inner hexagons.
        
        The outer hexagon has 6 half-widths. For a regular hexagon with side s,
        the distance from center to each side (apothem) is s * sqrt(3)/2.
        So we need apothem >= max projection of any vertex onto any of the 6 normals.
        side = 2 * apothem / sqrt(3)
        """
        outer_center = np.array([outer_cx, outer_cy])
        outer_angle_rad = math.radians(outer_angle_deg)
        
        max_proj = 0.0
        for i in range(N):
            cx, cy = centers[i]
            ang = angles[i]
            verts = hex_vertices(cx, cy, ang)
            rel_verts = verts - outer_center
            for k in range(6):
                normal_angle = outer_angle_rad + k * math.pi / 3
                normal = np.array([math.cos(normal_angle), math.sin(normal_angle)])
                proj = np.dot(rel_verts, normal)
                max_proj = max(max_proj, np.max(proj))
        
        # apothem = max_proj, side = 2 * apothem / sqrt(3)
        return 2.0 * max_proj / math.sqrt(3)

    def objective(params):
        """
        Objective: outer side length + overlap penalty.
        params: 3*N + 3 = 39 values
        [0..35]: centers (x,y) and angle for each hex
        [36]: outer cx
        [37]: outer cy
        [38]: outer angle
        """
        centers = [(params[3*i], params[3*i+1]) for i in range(N)]
        angles = [params[3*i+2] for i in range(N)]
        outer_cx = params[36]
        outer_cy = params[37]
        outer_angle = params[38]
        
        side = compute_outer_side(centers, angles, outer_cx, outer_cy, outer_angle)
        
        # Overlap penalty
        verts_list = [hex_vertices(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                gap = sat_gap(verts_list[i], verts_list[j])
                if gap < 0:
                    penalty += (-gap) ** 2
        
        return side + 100.0 * penalty

    # ---- Generate initial configurations ----
    def make_honeycomb():
        """Hexagonal packing: 1 center + 6 ring + 5 outer = 12"""
        s = 1.0
        dx = 2.0 * s
        dy = math.sqrt(3) * s
        centers = [(0.0, 0.0)]  # center
        # Ring of 6
        for k in range(6):
            a = k * math.pi / 3
            centers.append((dx * math.cos(a), dx * math.sin(a)))
        # 5 more in second ring (not all 12, pick 5)
        for k in range(6):
            a = k * math.pi / 3 + math.pi / 6
            r2 = math.sqrt(7) * s
            centers.append((r2 * math.cos(a), r2 * math.sin(a)))
            if len(centers) >= N:
                break
        centers = centers[:N]
        # Center them
        cx_mean = sum(c[0] for c in centers) / N
        cy_mean = sum(c[1] for c in centers) / N
        centers = [(x - cx_mean, y - cy_mean) for x, y in centers]
        angles = [0.0] * N
        return centers, angles

    def make_grid_3x4():
        """3 columns x 4 rows staggered grid"""
        s = 1.0
        dx = 2.0 * s
        dy = math.sqrt(3) * s
        offset = s
        centers = []
        for r in range(4):
            for c in range(3):
                x = c * dx
                y = r * dy
                if r % 2 == 1:
                    x += offset
                centers.append((x, y))
        cx_mean = sum(c[0] for c in centers) / N
        cy_mean = sum(c[1] for c in centers) / N
        centers = [(x - cx_mean, y - cy_mean) for x, y in centers]
        angles = [0.0] * N
        return centers, angles

    def make_grid_4x3():
        """4 columns x 3 rows staggered grid"""
        s = 1.0
        dx = 2.0 * s
        dy = math.sqrt(3) * s
        offset = s
        centers = []
        for r in range(3):
            for c in range(4):
                x = c * dx
                y = r * dy
                if r % 2 == 1:
                    x += offset
                centers.append((x, y))
        cx_mean = sum(c[0] for c in centers) / N
        cy_mean = sum(c[1] for c in centers) / N
        centers = [(x - cx_mean, y - cy_mean) for x, y in centers]
        angles = [0.0] * N
        return centers, angles

    def make_ring_plus_center():
        """1 center + 6 inner ring + 5 outer ring"""
        s = 1.0
        centers = [(0.0, 0.0)]
        # Inner ring at distance 2
        for k in range(6):
            a = k * math.pi / 3
            centers.append((2.0 * s * math.cos(a), 2.0 * s * math.sin(a)))
        # Outer ring: pick 5 from 12 positions
        for k in range(5):
            a = k * 2 * math.pi / 5
            r = math.sqrt(13) * s  # distance for second shell in hex packing
            centers.append((r * math.cos(a), r * math.sin(a)))
        centers = centers[:N]
        cx_mean = sum(c[0] for c in centers) / N
        cy_mean = sum(c[1] for c in centers) / N
        centers = [(x - cx_mean, y - cy_mean) for x, y in centers]
        angles = [0.0] * N
        return centers, angles

    def make_linear():
        """Two rows of 6, offset"""
        s = 1.0
        dx = 2.0 * s
        dy = math.sqrt(3) * s
        centers = []
        for r in range(2):
            for c in range(6):
                x = c * dx
                y = r * dy
                if r == 1:
                    x += s
                centers.append((x, y))
        cx_mean = sum(c[0] for c in centers) / N
        cy_mean = sum(c[1] for c in centers) / N
        centers = [(x - cx_mean, y - cy_mean) for x, y in centers]
        angles = [0.0] * N
        return centers, angles

    def pack_to_params(centers, angles):
        """Convert centers/angles to parameter vector with outer at origin."""
        params = np.zeros(39)
        for i in range(N):
            params[3*i] = centers[i][0]
            params[3*i+1] = centers[i][1]
            params[3*i+2] = angles[i]
        params[36] = 0.0
        params[37] = 0.0
        params[38] = 0.0
        return params

    # ---- Optimization ----
    def optimize_config(params0, max_iter=3000, timeout=None):
        """Use SLSQP to minimize the objective."""
        if timeout is not None and time.time() > timeout:
            return params0, objective(params0)
        
        # Use Nelder-Mead first for robustness, then SLSQP for polish
        try:
            res = minimize(objective, params0, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-5, 'fatol': 1e-6,
                                   'adaptive': True})
            best_params = res.x
            best_val = res.fun
            
            # Polish with SLSQP
            if time.time() < deadline - 10:
                try:
                    res2 = minimize(objective, best_params, method='SLSQP',
                                   options={'maxiter': 500, 'ftol': 1e-8})
                    if res2.fun < best_val:
                        best_params = res2.x
                        best_val = res2.fun
                except:
                    pass
            return best_params, best_val
        except:
            return params0, objective(params0)

    def local_search(params, steps=500, step_size=0.3, timeout=None):
        """Simple local search / hill climbing."""
        current = params.copy()
        current_val = objective(current)
        best = current.copy()
        best_val = current_val
        
        for step in range(steps):
            if timeout is not None and time.time() > timeout:
                break
            # Perturb a few random parameters
            new = current.copy()
            n_perturb = np.random.randint(1, 4)
            idxs = np.random.choice(len(new), n_perturb, replace=False)
            for idx in idxs:
                new[idx] += np.random.normal(0, step_size)
            
            new_val = objective(new)
            if new_val < current_val:
                current = new
                current_val = new_val
                if new_val < best_val:
                    best = new.copy()
                    best_val = new_val
            else:
                # Occasional larger jump
                if np.random.random() < 0.05:
                    new2 = current.copy()
                    idx = np.random.randint(len(new2))
                    new2[idx] += np.random.normal(0, step_size * 3)
                    val2 = objective(new2)
                    if val2 < current_val:
                        current = new2
                        current_val = val2
                        if val2 < best_val:
                            best = current.copy()
                            best_val = val2
        return best, best_val

    # ---- Main optimization loop ----
    initial_configs = [
        make_honeycomb(),
        make_grid_3x4(),
        make_grid_4x3(),
        make_ring_plus_center(),
        make_linear(),
    ]
    
    best_params = None
    best_side = float('inf')
    best_valid = False

    # First, evaluate all initial configs and find the best starting point
    for centers, angles in initial_configs:
        if time.time() > deadline - 20:
            break
        params = pack_to_params(centers, angles)
        # Compute actual outer side
        side = compute_outer_side(centers, angles, 0, 0, 0)
        valid = is_valid(centers, angles)
        if valid and side < best_side:
            best_side = side
            best_params = params.copy()
            best_valid = True

    # If no valid initial config, use the one with least overlap
    if not best_valid:
        for centers, angles in initial_configs:
            params = pack_to_params(centers, angles)
            val = objective(params)
            if val < best_side:
                best_side = val
                best_params = params.copy()

    # Run optimization from multiple starting points
    start_configs = []
    for centers, angles in initial_configs:
        start_configs.append(pack_to_params(centers, angles))
    # Also add perturbed versions of the best
    if best_params is not None:
        for _ in range(3):
            perturbed = best_params + np.random.normal(0, 0.5, size=39)
            start_configs.append(perturbed)

    for cfg in start_configs:
        if time.time() > deadline - 30:
            break
        
        # Phase 1: Nelder-Mead / SLSQP optimization
        opt_params, opt_val = optimize_config(cfg, max_iter=2000, timeout=deadline - 25)
        
        # Phase 2: Local search polish
        if time.time() < deadline - 15:
            ls_params, ls_val = local_search(opt_params, steps=300, step_size=0.2, timeout=deadline - 10)
            if ls_val < opt_val:
                opt_params = ls_params
                opt_val = ls_val
        
        # Evaluate final
        centers = [(opt_params[3*i], opt_params[3*i+1]) for i in range(N)]
        angles = [opt_params[3*i+2] for i in range(N)]
        outer_cx = opt_params[36]
        outer_cy = opt_params[37]
        outer_angle = opt_params[38]
        
        side = compute_outer_side(centers, angles, outer_cx, outer_cy, outer_angle)
        valid = is_valid(centers, angles)
        
        if valid and side < best_side:
            best_side = side
            best_params = opt_params.copy()
            best_valid = True
        elif not best_valid and opt_val < best_side:
            best_side = opt_val
            best_params = opt_params.copy()

    # Final polish: optimize outer hexagon parameters for the best inner config
    if best_params is not None:
        if time.time() < deadline - 5:
            # Try to optimize just the outer parameters
            inner_part = best_params[:36].copy()
            
            def outer_obj(outer_params):
                outer_cx, outer_cy, outer_angle = outer_params
                centers = [(inner_part[3*i], inner_part[3*i+1]) for i in range(N)]
                angles = [inner_part[3*i+2] for i in range(N)]
                return compute_outer_side(centers, angles, outer_cx, outer_cy, outer_angle)
            
            try:
                res = minimize(outer_obj, np.array([best_params[36], best_params[37], best_params[38]]),
                              method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-6})
                if res.fun < best_side:
                    best_params[36] = res.x[0]
                    best_params[37] = res.x[1]
                    best_params[38] = res.x[2]
                    best_side = res.fun
            except:
                pass

    # Construct return value
    if best_params is None:
        # Absolute fallback
        inner = [[0.0, 0.0, 0.0] for _ in range(N)]
        return inner, (0.0, 0.0), 100.0, 0.0

    inner = []
    for i in range(N):
        cx = best_params[3*i]
        cy = best_params[3*i+1]
        ang = best_params[3*i+2]
        inner.append([float(cx), float(cy), float(ang)])
    
    outer_cx = float(best_params[36])
    outer_cy = float(best_params[37])
    outer_angle = float(best_params[38])
    
    # Recompute exact side with small safety margin
    centers = [(best_params[3*i], best_params[3*i+1]) for i in range(N)]
    angles = [best_params[3*i+2] for i in range(N)]
    final_side = compute_outer_side(centers, angles, outer_cx, outer_cy, outer_angle)
    final_side *= 1.005  # Tiny safety margin
    
    return inner, (outer_cx, outer_cy), float(final_side), float(outer_angle)
