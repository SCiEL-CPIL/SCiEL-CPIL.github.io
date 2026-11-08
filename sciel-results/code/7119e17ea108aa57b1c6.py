import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Helper: Generate vertices of a regular hexagon
    def hex_verts(cx, cy, s, rot):
        angles = np.deg2rad(np.arange(6) * 60 + rot)
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def compute_min_dist(v1, v2):
        """Compute minimum distance between two convex hexagons (arrays of 6 vertices).
        Returns 0 if overlapping, else min edge-vertex distance."""
        # Check for intersection first
        def point_in_hex(pt, hexv):
            # Winding number for convex hexagon
            n = 6
            inside = 0
            for i in range(n):
                p1 = hexv[i]
                p2 = hexv[(i+1) % n]
                if p1[1] <= pt[1]:
                    if p2[1] > pt[1] and (p2[0] - p1[0]) * (pt[1] - p1[1]) > (pt[0] - p1[0]) * (p2[1] - p1[1]):
                        inside += 1
                else:
                    if p2[1] <= pt[1] and (p2[0] - p1[0]) * (pt[1] - p1[1]) > (pt[0] - p1[0]) * (p2[1] - p1[1]):
                        inside -= 1
            return inside != 0

        for i in range(6):
            if point_in_hex(v1[i], v2):
                return 0.0
        for i in range(6):
            if point_in_hex(v2[i], v1):
                return 0.0

        # Edge intersection check
        def seg_intersect(a, b, c, d):
            def cross(o, p, q):
                return (p[0]-o[0])*(q[1]-o[1]) - (p[1]-o[1])*(q[0]-o[0])
            d1 = cross(c, d, a)
            d2 = cross(c, d, b)
            d3 = cross(a, b, c)
            d4 = cross(a, b, d)
            if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
               ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                return True
            return False

        for i in range(6):
            a = v1[i]
            b = v1[(i+1) % 6]
            for j in range(6):
                c = v2[j]
                d = v2[(j+1) % 6]
                if seg_intersect(a, b, c, d):
                    return 0.0

        # Min distance between edges (point-to-segment)
        min_d = float('inf')
        for i in range(6):
            a = v1[i]
            b = v1[(i+1) % 6]
            ab = b - a
            ab_len_sq = np.dot(ab, ab)
            for j in range(6):
                c = v2[j]
                d = v2[(j+1) % 6]
                # Point c to segment ab
                if ab_len_sq > 1e-12:
                    t = np.dot(c - a, ab) / ab_len_sq
                    t = max(0.0, min(1.0, t))
                    closest = a + t * ab
                    dist = np.linalg.norm(c - closest)
                else:
                    dist = np.linalg.norm(c - a)
                if dist < min_d:
                    min_d = dist
                # Point d to segment ab
                if ab_len_sq > 1e-12:
                    t = np.dot(d - a, ab) / ab_len_sq
                    t = max(0.0, min(1.0, t))
                    closest = a + t * ab
                    dist = np.linalg.norm(d - closest)
                else:
                    dist = np.linalg.norm(d - a)
                if dist < min_d:
                    min_d = dist
        return min_d

    def evaluate_solution(params):
        """
        params: [x1, y1, a1, ..., x12, y12, a12, outer_s, outer_rot]
        Returns: (outer_side_length, validity)
        """
        xs = params[0:12]
        ys = params[12:24]
        as_ = params[24:36]
        outer_s = params[36]
        outer_rot = params[37]
        
        inner_verts = []
        for i in range(12):
            v = hex_verts(xs[i], ys[i], 1.0, as_[i])
            inner_verts.append(v)
            
        # Check disjointness
        for i in range(12):
            for j in range(i+1, 12):
                d = compute_min_dist(inner_verts[i], inner_verts[j])
                if d < 1e-4:
                    return float('inf'), False

        # Check containment in outer hexagon
        r = outer_s * np.sqrt(3) / 2.0
        normals = np.deg2rad(30 + np.arange(6) * 60 + outer_rot)
        n_vecs = np.column_stack([np.cos(normals), np.sin(normals)])
        
        for i in range(12):
            for v in inner_verts[i]:
                dists = np.dot(v, n_vecs.T)
                if np.any(dists > r + 1e-4):
                    return float('inf'), False

        return outer_s, True

    def objective(params):
        """Penalty-based objective for optimization."""
        val, valid = evaluate_solution(params)
        if not valid:
            return 100.0 + val
        return val

    # Generate multiple heuristic initial configurations
    def gen_hexagonal_grid_config():
        """6-1-6 hexagonal arrangement (like honeycomb cluster)."""
        params = np.zeros(38)
        # Center
        params[0], params[12], params[24] = 0.0, 0.0, 0.0
        # 6 around center at distance ~1.732 (sqrt(3)*2 for touching)
        d = 1.732
        for i in range(6):
            ang = i * 60
            params[i*3+1] = d * np.cos(np.deg2rad(ang))   # x offset by 1
            params[i*3+13] = d * np.sin(np.deg2rad(ang))   # y
            params[i*3+25] = 0.0                            # rot
        # Outer ring: 6 more hexagons
        # These go in the gaps between the 6 surrounding the center
        for i in range(6):
            ang = i * 60 + 30
            rd = d * 1.732  # ~3.0
            idx = 7 + i
            params[idx*3] = rd * np.cos(np.deg2rad(ang))
            params[idx*3+12] = rd * np.sin(np.deg2rad(ang))
            params[idx*3+24] = 0.0
        params[36] = 6.0
        params[37] = 0.0
        return params

    def gen_circular_config():
        """12 hexagons in a circle."""
        params = np.zeros(38)
        r = 2.5
        for i in range(12):
            ang = i * 30
            params[i*3] = r * np.cos(np.deg2rad(ang))
            params[i*3+12] = r * np.sin(np.deg2rad(ang))
            params[i*3+24] = 0.0
        params[36] = 6.0
        params[37] = 0.0
        return params

    def gen_random_config():
        """Random scattered configuration."""
        np.random.seed(np.random.randint(0, 100000))
        params = np.zeros(38)
        for i in range(12):
            ang = np.random.uniform(0, 360)
            r = np.random.uniform(0.5, 4.0)
            params[i*3] = r * np.cos(np.deg2rad(ang))
            params[i*3+12] = r * np.sin(np.deg2rad(ang))
            params[i*3+24] = np.random.uniform(0, 60)
        params[36] = 7.0
        params[37] = np.random.uniform(0, 60)
        return params

    # Try multiple starting configurations
    initial_configs = [gen_hexagonal_grid_config(), gen_circular_config()]
    for _ in range(3):
        initial_configs.append(gen_random_config())

    best_params = None
    best_val = float('inf')

    # Evaluate initial configs
    for cfg in initial_configs:
        val, valid = evaluate_solution(cfg)
        if valid:
            if val < best_val:
                best_val = val
                best_params = cfg.copy()

    # If no valid config found, use a very large outer hexagon
    if best_params is None:
        cfg = gen_circular_config()
        cfg[36] = 15.0
        val, valid = evaluate_solution(cfg)
        if valid:
            best_params = cfg.copy()
            best_val = val

    # L-BFGS-B optimization with multiple restarts
    bounds = [(-8, 8)] * 12 + [(-8, 8)] * 12 + [(0, 60)] * 12 + [(2.0, 15.0)] + [(0, 60)]

    # Simpler objective that's smoother for optimization
    def smooth_objective(params):
        """Smoother objective using soft penalties."""
        xs = params[0:12]
        ys = params[12:24]
        as_ = params[24:36]
        outer_s = params[36]
        outer_rot = params[37]
        
        inner_verts = []
        for i in range(12):
            v = hex_verts(xs[i], ys[i], 1.0, as_[i])
            inner_verts.append(v)
        
        penalty = 0.0
        # Overlap penalties
        for i in range(12):
            for j in range(i+1, 12):
                d = compute_min_dist(inner_verts[i], inner_verts[j])
                if d < 0.01:
                    penalty += (0.01 - d) * 1000.0
                elif d < 0.1:
                    penalty += (0.1 - d) * 100.0
        
        # Containment penalties
        r = outer_s * np.sqrt(3) / 2.0
        normals = np.deg2rad(30 + np.arange(6) * 60 + outer_rot)
        n_vecs = np.column_stack([np.cos(normals), np.sin(normals)])
        
        for i in range(12):
            for v in inner_verts[i]:
                dists = np.dot(v, n_vecs.T)
                violations = np.maximum(0, dists - r)
                if np.any(violations > 0):
                    penalty += np.sum(violations * violations) * 100.0
        
        return outer_s + penalty

    n_restarts = 8
    for r in range(n_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            curr_params = best_params.copy() if best_params is not None else initial_configs[0].copy()
        else:
            if best_params is not None:
                curr_params = best_params.copy()
                curr_params[0:36] += np.random.normal(0, 0.3, 36)
                curr_params[24:36] = np.mod(curr_params[24:36], 60)
            else:
                curr_params = initial_configs[r % len(initial_configs)].copy()
                curr_params[0:36] += np.random.normal(0, 0.5, 36)
        
        try:
            res = minimize(smooth_objective, curr_params, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})
            
            final_params = res.x
            final_params[24:36] = np.mod(final_params[24:36], 60)
            final_params[37] = np.mod(final_params[37], 60)
            
            val, valid = evaluate_solution(final_params)
            if valid:
                if val < best_val:
                    best_val = val
                    best_params = final_params.copy()
        except Exception:
            pass

    # Final polish: try to shrink outer hexagon from best valid config
    if best_params is not None:
        # Binary search on outer side length
        for _ in range(10):
            if time.time() > deadline:
                break
            outer_s = best_params[36]
            # Try slightly smaller
            test_params = best_params.copy()
            test_params[36] = outer_s * 0.98
            val, valid = evaluate_solution(test_params)
            if valid:
                best_params = test_params.copy()
                best_val = val
            else:
                # Try smaller step
                test_params[36] = outer_s * 0.995
                val, valid = evaluate_solution(test_params)
                if valid:
                    best_params = test_params.copy()
                    best_val = val

    # Fallback
    if best_params is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    xs = best_params[0:12]
    ys = best_params[12:24]
    as_ = best_params[24:36]
    outer_s = best_params[36]
    outer_rot = best_params[37]
    
    inner = []
    for i in range(12):
        inner.append([float(xs[i]), float(ys[i]), float(as_[i])])
        
    return inner, (0.0, 0.0), float(outer_s), float(outer_rot)
