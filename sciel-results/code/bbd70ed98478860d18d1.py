"""Direct construction of 12 unit hexagons inside a regular hexagon, minimizing outer side."""
N = 12

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math
import random

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _point_in_polygon(pt, poly):
    """Check if point is inside a convex polygon using cross products."""
    n = len(poly)
    for i in range(n):
        v1 = poly[i]
        v2 = poly[(i + 1) % n]
        edge = v2 - v1
        rel = pt - v1
        cross = edge[0] * rel[1] - edge[1] * rel[0]
        if cross < -1e-9:
            return False
    return True

def _containment_penalty(inner_verts, outer_verts):
    """Penalty for inner hexagon not being fully inside outer hexagon."""
    penalty = 0.0
    n_out = len(outer_verts)
    for iv in inner_verts:
        for i in range(n_out):
            v1 = outer_verts[i]
            v2 = outer_verts[(i + 1) % n_out]
            edge = v2 - v1
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            rel = iv - v1
            cross = edge[0] * rel[1] - edge[1] * rel[0]
            if cross < 0:
                penalty += (-cross) / edge_len
    return penalty

def _sat_overlap_depth(h1, h2):
    """Minimum penetration depth between two convex polygons using SAT. Returns 0 if disjoint."""
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

def _total_penalty(params, outer_side, outer_center, outer_angle):
    """Total penalty: containment + pairwise disjointness."""
    inner_verts_list = []
    for i in range(N):
        cx, cy, ang = params[3*i], params[3*i+1], params[3*i+2]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    
    outer_verts = _hex_verts(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # Containment
    cont_pen = 0.0
    for iv in inner_verts_list:
        cont_pen += _containment_penalty(iv, outer_verts)
    
    # Disjointness
    disj_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if depth > 0:
                disj_pen += depth
    
    return cont_pen + disj_pen

def _check_valid(params, outer_side, outer_center, outer_angle, tol=1e-6):
    """Check if configuration is valid (no overlaps, all contained)."""
    inner_verts_list = []
    for i in range(N):
        cx, cy, ang = params[3*i], params[3*i+1], params[3*i+2]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    
    outer_verts = _hex_verts(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    cont_pen = 0.0
    for iv in inner_verts_list:
        cont_pen += _containment_penalty(iv, outer_verts)
    
    disj_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if depth > 0:
                disj_pen += depth
    
    return cont_pen < tol and disj_pen < tol

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    def build_hexagonal_lattice(nrows, ncols):
        """Build a hexagonal lattice of positions (offset rows)."""
        positions = []
        for r in range(nrows):
            for c in range(ncols):
                x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
                y = r * np.sqrt(3)
                positions.append((x, y))
        return positions
    
    def build_ring_config():
        """6 around center + 5 more in second ring."""
        positions = [(0.0, 0.0)]
        for i in range(6):
            ang = np.radians(60 * i)
            positions.append((2.0 * np.cos(ang), 2.0 * np.sin(ang)))
        for i in range(6):
            ang = np.radians(60 * i + 30)
            positions.append((3.464 * np.cos(ang), 3.464 * np.sin(ang)))
        return positions[:12]
    
    def make_params_from_positions(positions, angles=None):
        """Create parameter vector from positions and optional angles."""
        if angles is None:
            angles = [0.0] * len(positions)
        params = []
        for i, (x, y) in enumerate(positions):
            params.extend([float(x), float(y), float(angles[i])])
        return np.array(params)
    
    # Generate candidate initial configurations
    configs = []
    
    # 3 rows x 4 cols hex grid
    pos_3x4 = build_hexagonal_lattice(3, 4)
    if len(pos_3x4) == N:
        arr = np.array(pos_3x4)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
    
    # 4 rows x 3 cols hex grid
    pos_4x3 = build_hexagonal_lattice(4, 3)
    if len(pos_4x3) == N:
        arr = np.array(pos_4x3)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
    
    # 2 rows x 6 cols
    pos_2x6 = build_hexagonal_lattice(2, 6)
    if len(pos_2x6) == N:
        arr = np.array(pos_2x6)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
    
    # Ring configuration
    ring_pos = build_ring_config()
    if len(ring_pos) == N:
        arr = np.array(ring_pos)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
    
    # 6 rows x 2 cols
    pos_6x2 = build_hexagonal_lattice(6, 2)
    if len(pos_6x2) == N:
        arr = np.array(pos_6x2)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
    
    # 12 in a line
    line_pos = [(float(i - 5.5), 0.0) for i in range(12)]
    configs.append(line_pos)
    
    # Compact 2x6 with offset
    compact_pos = []
    for r in range(2):
        for c in range(6):
            x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
            y = r * np.sqrt(3)
            compact_pos.append((x, y))
    arr = np.array(compact_pos)
    centroid = arr.mean(axis=0)
    configs.append((arr - centroid).tolist())
    
    # Additional configurations with different angles
    for base_config in configs[:3]:
        for ang_offset in [30.0, 60.0]:
            angled = []
            for (x, y) in base_config:
                angled.append((x, y, ang_offset))
            configs.append(angled)
    
    best_score = float('inf')
    best_result = None
    
    outer_angles = [0.0, 30.0]
    
    for outer_angle in outer_angles:
        outer_center = (0.0, 0.0)
        
        # Binary search on outer side
        lo, hi = 3.0, 6.0
        
        # First, find a feasible upper bound
        for test_side in [5.0, 4.5, 4.0, 3.8, 3.6, 3.4, 3.2, 3.0]:
            if time.time() > deadline - 30.0:
                break
            
            found = False
            for config in configs:
                if time.time() > deadline - 25.0:
                    break
                
                if len(config[0]) == 2:
                    init_params = make_params_from_positions(config)
                else:
                    init_params = np.array([item for tri in config for item in tri])
                
                # Quick check
                pen = _total_penalty(init_params, test_side, outer_center, outer_angle)
                
                if pen < 1e-6:
                    found = True
                    if test_side < best_score:
                        best_score = test_side
                        best_result = (init_params.copy(), outer_center, test_side, outer_angle)
                    break
                
                # Try local optimization
                if time.time() > deadline - 20.0:
                    continue
                
                def objective(p):
                    return _total_penalty(p, test_side, outer_center, outer_angle)
                
                try:
                    result = minimize(
                        objective,
                        init_params,
                        method='L-BFGS-B',
                        options={'maxiter': 150, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    opt_params = result.x
                    opt_val = result.fun
                except Exception:
                    continue
                
                if opt_val < 1e-6:
                    found = True
                    if test_side < best_score:
                        best_score = test_side
                        best_result = (opt_params, outer_center, test_side, outer_angle)
                    break
            
            if found:
                hi = test_side + 0.1
            else:
                lo = test_side - 0.1
                if lo <= 3.0:
                    break
        
        # Now binary search between lo and hi for tighter fit
        if time.time() < deadline - 40.0:
            for _ in range(8):  # ~8 bisection steps
                if time.time() > deadline - 35.0:
                    break
                
                mid = (lo + hi) / 2.0
                found = False
                
                for config in configs[:4]:  # Use fewer configs for speed
                    if time.time() > deadline - 30.0:
                        break
                    
                    if len(config[0]) == 2:
                        init_params = make_params_from_positions(config)
                    else:
                        init_params = np.array([item for tri in config for item in tri])
                    
                    pen = _total_penalty(init_params, mid, outer_center, outer_angle)
                    
                    if pen < 1e-6:
                        found = True
                        if mid < best_score:
                            best_score = mid
                            best_result = (init_params.copy(), outer_center, mid, outer_angle)
                        break
                    
                    if time.time() > deadline - 25.0:
                        continue
                    
                    def objective(p):
                        return _total_penalty(p, mid, outer_center, outer_angle)
                    
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
                    
                    if opt_val < 1e-6:
                        found = True
                        if mid < best_score:
                            best_score = mid
                            best_result = (opt_params, outer_center, mid, outer_angle)
                        break
                
                if found:
                    hi = mid
                else:
                    lo = mid
        
        # Try Differential Evolution for global optimization
        if time.time() < deadline - 50.0:
            # Use DE to search for tighter packing
            side_to_try = min(hi, best_score if best_score < float('inf') else hi)
            
            if side_to_try > 3.1:
                bounds = []
                for i in range(N):
                    bounds.append((-3.0, 3.0))  # x
                    bounds.append((-3.0, 3.0))  # y
                    bounds.append((0.0, 180.0))  # angle
                
                def de_objective(p):
                    return _total_penalty(p, side_to_try, outer_center, outer_angle)
                
                try:
                    from scipy.optimize import differential_evolution
                    de_result = differential_evolution(
                        de_objective,
                        bounds,
                        seed=42,
                        maxiter=50,
                        tol=1e-12,
                        polish=True,
                        workers=1
                    )
                    if de_result.fun < 1e-6:
                        if side_to_try < best_score:
                            best_score = side_to_try
                            best_result = (de_result.x, outer_center, side_to_try, outer_angle)
                except Exception:
                    pass
    
    # Simulated annealing on discrete grid for better packing
    if best_result is not None and time.time() < deadline - 30.0:
        opt_params, outer_center, outer_side, outer_angle = best_result
        
        # Grid-based simulated annealing
        grid_step = 0.05
        x_range = np.arange(-3.0, 3.0 + grid_step, grid_step)
        y_range = np.arange(-3.0, 3.0 + grid_step, grid_step)
        angle_options = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]
        
        # Start from current best
        sa_params = opt_params.copy()
        current_side = outer_side
        
        # Try to reduce side length using SA
        for target_side in [outer_side - 0.02, outer_side - 0.04, outer_side - 0.06, outer_side - 0.08]:
            if time.time() > deadline - 25.0:
                break
            
            if target_side < 3.0:
                break
            
            # Run SA for this target side
            temp = 0.1
            cooling = 0.95
            min_temp = 0.001
            
            best_sa_params = sa_params.copy()
            best_sa_pen = _total_penalty(sa_params, target_side, outer_center, outer_angle)
            
            while temp > min_temp:
                if time.time() > deadline - 20.0:
                    break
                
                # Random move: pick a random hexagon and perturb it
                idx = random.randint(0, N - 1)
                new_params = sa_params.copy()
                
                # Random displacement
                dx = random.uniform(-0.2, 0.2)
                dy = random.uniform(-0.2, 0.2)
                da = random.choice(angle_options)
                
                new_params[3*idx] += dx
                new_params[3*idx+1] += dy
                new_params[3*idx+2] = da
                
                new_pen = _total_penalty(new_params, target_side, outer_center, outer_angle)
                
                current_pen = _total_penalty(sa_params, target_side, outer_center, outer_angle)
                
                if new_pen < best_sa_pen:
                    best_sa_pen = new_pen
                    best_sa_params = new_params.copy()
                    sa_params = new_params.copy()
                elif random.random() < math.exp(-(new_pen - current_pen) / temp):
                    sa_params = new_params.copy()
                
                temp *= cooling
            
            if best_sa_pen < 1e-6:
                outer_side = target_side
                opt_params = best_sa_params
                best_score = target_side
                best_result = (opt_params, outer_center, outer_side, outer_angle)
    
    # Final refinement: try to squeeze out more
    if best_result is not None and time.time() < deadline - 10.0:
        opt_params, outer_center, outer_side, outer_angle = best_result
        
        # Try progressively smaller sides
        for delta in [0.05, 0.03, 0.02, 0.01, 0.005]:
            if time.time() > deadline - 8.0:
                break
            new_side = outer_side - delta
            if new_side < 3.0:
                break
            
            def final_objective(p):
                return _total_penalty(p, new_side, outer_center, outer_angle)
            
            try:
                result = minimize(
                    final_objective,
                    opt_params,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 8000, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if result.fun < 1e-6:
                    outer_side = new_side
                    opt_params = result.x
            except Exception:
                pass
        
        best_result = (opt_params, outer_center, outer_side, outer_angle)
    
    # Fallback
    if best_result is None:
        inner = []
        pos = build_hexagonal_lattice(3, 4)
        arr = np.array(pos)
        centroid = arr.mean(axis=0)
        for (x, y) in (arr - centroid):
            inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 5.5, 0.0
    
    opt_params, outer_center, outer_side, outer_angle = best_result
    
    inner = []
    for i in range(N):
        x = float(opt_params[3*i])
        y = float(opt_params[3*i+1])
        ang = float(opt_params[3*i+2])
        inner.append([x, y, ang])
    
    # Final validation
    inner_verts_list = []
    for i in range(N):
        cx, cy, ang = opt_params[3*i], opt_params[3*i+1], opt_params[3*i+2]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    outer_verts = _hex_verts(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    cont_pen = sum(_containment_penalty(iv, outer_verts) for iv in inner_verts_list)
    disj_pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if d > 0:
                disj_pen += d
    
    if cont_pen > 1e-4 or disj_pen > 1e-4:
        inner = []
        pos = build_hexagonal_lattice(3, 4)
        arr = np.array(pos)
        centroid = arr.mean(axis=0)
        for (x, y) in (arr - centroid):
            inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 5.5, 0.0
    
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)

# EVOLVE-BLOCK-END
