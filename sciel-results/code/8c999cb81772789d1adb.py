"""
Strategy: Boundary Tracing with Epigraph Formulation and Smooth Clearance.
Arrange 12 unit hexagons in a 6-fold symmetric honeycomb cluster.
Use an epigraph formulation with auxiliary variable t and smooth log-sum-exp
clearance constraints. Final polish with exact objective and tighter tolerances.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize, minimize_scalar

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # Precompute unit hexagon vertices at angle 0
    unit_hex_verts = np.array([
        [math.cos(math.radians(a)), math.sin(math.radians(a))]
        for a in range(0, 360, 60)
    ])

    def get_inner_positions(r1, a1, r2, a2):
        """Generate 12 inner hexagon positions: (x, y, angle_deg)."""
        positions = []
        positions.append((0.0, 0.0, 0.0))
        for k in range(6):
            angle = math.radians(60 * k)
            x = r1 * math.cos(angle)
            y = r1 * math.sin(angle)
            positions.append((x, y, a1))
        for k in range(5):
            angle = math.radians(60 * k + 30)
            x = r2 * math.cos(angle)
            y = r2 * math.sin(angle)
            positions.append((x, y, a2))
        return positions

    def compute_min_side_fast(r1, a1, r2, a2, outer_angle_deg):
        """Compute minimal outer hexagon side length using vectorized support function."""
        outer_angle_rad = math.radians(outer_angle_deg)
        normal_angles = outer_angle_rad + math.radians(30) + math.radians(60) * np.arange(6)
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6,2)
        
        # Get all inner positions
        positions = get_inner_positions(r1, a1, r2, a2)
        
        max_required = 0.0
        for (cx, cy, alpha_deg) in positions:
            alpha_rad = math.radians(alpha_deg)
            cos_a = math.cos(alpha_rad)
            sin_a = math.sin(alpha_rad)
            local_normals = np.column_stack([
                normals[:, 0] * cos_a + normals[:, 1] * sin_a,
                -normals[:, 0] * sin_a + normals[:, 1] * cos_a
            ])
            dots = local_normals @ unit_hex_verts.T  # (6, 6)
            supports = np.max(dots, axis=1)  # (6,)
            center_proj = cx * normals[:, 0] + cy * normals[:, 1]  # (6,)
            vals = center_proj + supports  # (6,)
            max_required = max(max_required, np.max(vals))
        
        apothem = max_required
        side = apothem / (math.sqrt(3) / 2)
        return side

    def soft_max(values, beta=100.0):
        """Smooth approximation of max using log-sum-exp."""
        scaled = beta * values
        max_val = np.max(scaled)
        return max_val / beta + math.log(np.sum(np.exp(scaled - max_val))) / beta

    def compute_clearances(r1, a1, r2, a2, outer_angle_deg):
        """Compute the 6 clearance values (one per outer normal)."""
        outer_angle_rad = math.radians(outer_angle_deg)
        normal_angles = outer_angle_rad + math.radians(30) + math.radians(60) * np.arange(6)
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        positions = get_inner_positions(r1, a1, r2, a2)
        
        clearances = np.zeros(6)
        for (cx, cy, alpha_deg) in positions:
            alpha_rad = math.radians(alpha_deg)
            cos_a = math.cos(alpha_rad)
            sin_a = math.sin(alpha_rad)
            local_normals = np.column_stack([
                normals[:, 0] * cos_a + normals[:, 1] * sin_a,
                -normals[:, 0] * sin_a + normals[:, 1] * cos_a
            ])
            dots = local_normals @ unit_hex_verts.T  # (6, 6)
            supports = np.max(dots, axis=1)  # (6,)
            center_proj = cx * normals[:, 0] + cy * normals[:, 1]  # (6,)
            vals = center_proj + supports  # (6,)
            clearances = np.maximum(clearances, vals)
        
        return clearances

    def objective_epigraph(params, beta=100.0):
        """Epigraph formulation: minimize t subject to smooth clearance <= t.
        params = [r1, a1, r2, a2, o_a, t]
        """
        if time.time() > deadline - 3:
            return 1e6
        r1, a1, r2, a2, o_a, t = params
        if r1 < 0.1 or r2 < 0.1 or t < 0.1:
            return 1e6
        
        clearances = compute_clearances(r1, a1, r2, a2, o_a)
        # Smooth constraint violation: sum of soft violations where clearance > t
        violations = np.maximum(0, clearances - t)
        smooth_violation = np.sum(np.log1p(np.exp(beta * violations))) / beta
        side = t / (math.sqrt(3) / 2)
        return side + 100.0 * smooth_violation

    def check_disjoint_fast(r1, a1, r2, a2):
        """Check if inner hexagons are pairwise disjoint using SAT (fast version)."""
        positions = get_inner_positions(r1, a1, r2, a2)
        n_pos = len(positions)
        
        all_verts = []
        for (xi, yi, ai) in positions:
            ai_rad = math.radians(ai)
            v = np.array([
                [xi + math.cos(ai_rad + math.radians(60*k)), yi + math.sin(ai_rad + math.radians(60*k))]
                for k in range(6)
            ])
            all_verts.append(v)
        
        all_normals = []
        for v in all_verts:
            normals = []
            for k in range(6):
                p1 = v[k]
                p2 = v[(k + 1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-9:
                    normals.append(normal / norm)
            all_normals.append(normals)
        
        for i in range(n_pos):
            for j in range(i + 1, n_pos):
                vi = all_verts[i]
                vj = all_verts[j]
                axes = all_normals[i] + all_normals[j]
                
                separated = False
                for axis in axes:
                    proj_i = vi @ axis
                    proj_j = vj @ axis
                    if proj_i.max() < proj_j.min() - 1e-9 or proj_j.max() < proj_i.min() - 1e-9:
                        separated = True
                        break
                
                if not separated:
                    return False
        return True

    def objective_exact(params):
        """Exact objective with disjointness penalty."""
        if time.time() > deadline - 3:
            return 1e6
        r1, a1, r2, a2, o_a = params
        if r1 < 0.1 or r2 < 0.1:
            return 1e6
        if not check_disjoint_fast(r1, a1, r2, a2):
            return 50.0 + compute_min_side_fast(r1, a1, r2, a2, o_a)
        return compute_min_side_fast(r1, a1, r2, a2, o_a)

    # Multi-start global search
    best_side = float('inf')
    best_params = None
    
    # Initial guesses based on honeycomb geometry
    initial_guesses = [
        [2.0, 0.0, 2*math.sqrt(3), 0.0, 30.0],
        [2.0, 0.0, 2*math.sqrt(3), 30.0, 0.0],
        [2.0, 30.0, 2*math.sqrt(3), 0.0, 0.0],
        [2.0, 0.0, 2*math.sqrt(3), 0.0, 0.0],
        [2.0, 0.0, 2*math.sqrt(3), 30.0, 30.0],
        [1.95, 0.0, 3.45, 0.0, 30.0],
        [2.05, 0.0, 3.5, 0.0, 30.0],
        [1.98, 10.0, 3.48, 20.0, 15.0],
        [2.02, 5.0, 3.52, 10.0, 25.0],
    ]
    
    np.random.seed(42)
    for _ in range(10):
        r1 = np.random.uniform(1.85, 2.15)
        a1 = np.random.uniform(0, 60)
        r2 = np.random.uniform(3.3, 3.65)
        a2 = np.random.uniform(0, 60)
        o_a = np.random.uniform(0, 60)
        initial_guesses.append([r1, a1, r2, a2, o_a])
    
    bounds_5d = [(1.5, 2.5), (0, 60), (2.5, 4.5), (0, 60), (0, 60)]
    bounds_6d = bounds_5d + [(0.5, 5.0)]
    
    for guess in initial_guesses:
        if time.time() > deadline - 10:
            break
        
        # Phase 1: Epigraph formulation with L-BFGS-B
        try:
            t_init = compute_min_side_fast(*guess) * (math.sqrt(3) / 2) * 1.05
            guess_6d = guess + [t_init]
            result = minimize(
                objective_epigraph,
                x0=np.array(guess_6d, dtype=float),
                method='L-BFGS-B',
                bounds=bounds_6d,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000}
            )
            if result.success or result.fun < 50:
                params_5d = result.x[:5]
                exact_val = compute_min_side_fast(*params_5d)
                if check_disjoint_fast(*params_5d[:4]) and exact_val < best_side:
                    best_side = exact_val
                    best_params = params_5d.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 5:
            break
        
        # Phase 2: Polish with Nelder-Mead on exact objective
        if best_params is not None:
            try:
                result2 = minimize(
                    objective_exact,
                    x0=best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if result2.fun < 50:
                    params = result2.x
                    exact_val = compute_min_side_fast(*params)
                    if check_disjoint_fast(*params[:4]) and exact_val < best_side:
                        best_side = exact_val
                        best_params = params.copy()
            except Exception:
                pass
        else:
            try:
                result2 = minimize(
                    objective_exact,
                    x0=np.array(guess, dtype=float),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if result2.fun < 50:
                    params = result2.x
                    exact_val = compute_min_side_fast(*params)
                    if check_disjoint_fast(*params[:4]) and exact_val < best_side:
                        best_side = exact_val
                        best_params = params.copy()
            except Exception:
                pass
    
    # Final high-precision polish
    if best_params is not None and time.time() < deadline - 3:
        try:
            result_final = minimize(
                objective_exact,
                x0=best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if result_final.fun < 50:
                params = result_final.x
                exact_val = compute_min_side_fast(*params)
                if check_disjoint_fast(*params[:4]) and exact_val < best_side:
                    best_side = exact_val
                    best_params = params.copy()
        except Exception:
            pass
    
    # If no valid solution found, use a known valid configuration
    if best_params is None:
        best_params = np.array([2.0, 0.0, 2*math.sqrt(3), 0.0, 30.0])
        best_side = compute_min_side_fast(*best_params)
    
    r1, a1, r2, a2, o_a = best_params
    a1 = a1 % 360
    a2 = a2 % 360
    o_a = o_a % 360
    
    inner = []
    inner.append((0.0, 0.0, 0.0))
    for k in range(6):
        angle = math.radians(60 * k)
        x = r1 * math.cos(angle)
        y = r1 * math.sin(angle)
        inner.append((x, y, a1))
    for k in range(5):
        angle = math.radians(60 * k + 30)
        x = r2 * math.cos(angle)
        y = r2 * math.sin(angle)
        inner.append((x, y, a2))
    
    oc = (0.0, 0.0)
    oa = o_a
    os = compute_min_side_fast(r1, a1, r2, a2, o_a)
    
    return inner, oc, os, oa
