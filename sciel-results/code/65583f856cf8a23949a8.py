"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3.0)
DEADLINE = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon with side s, center (cx,cy), rotation angle_deg."""
    a = math.radians(angle_deg)
    angles = a + np.pi / 3 * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def compute_required_side(inner_hexagons, outer_center, outer_angle):
    """Compute minimum outer hexagon side length that contains all inner hexagons."""
    ox, oy = outer_center
    oa = math.radians(outer_angle)
    
    # The 6 outward normals of the outer hexagon (pointing to face midpoints)
    normals = np.array([[math.cos(oa + math.radians(30 + 60 * k)), 
                         math.sin(oa + math.radians(30 + 60 * k))]
                        for k in range(6)])
    
    # Collect all vertices of all inner hexagons
    all_verts = []
    for cx, cy, ang in inner_hexagons:
        verts = get_hex_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # Translate relative to outer center
    all_verts_rel = all_verts - np.array([ox, oy])
    
    # For a regular hexagon with side S, the apothem is S*sqrt(3)/2
    # Constraint: for each face normal n_k, max dot(v_rel, n_k) <= S * sqrt(3)/2
    # Also: max dot(v_rel, -n_k) <= S * sqrt(3)/2
    all_normals = np.vstack([normals, -normals])
    supports = np.max(all_verts_rel @ all_normals.T, axis=1)
    
    min_S = np.max(supports) / (SQRT3 / 2.0)
    return max(min_S, 0.0)

def compute_required_side_smooth(inner_hexagons, outer_center, outer_angle, beta=50.0):
    """Smooth version of compute_required_side using log-sum-exp for differentiability."""
    ox, oy = outer_center
    oa = math.radians(outer_angle)
    
    normals = np.array([[math.cos(oa + math.radians(30 + 60 * k)), 
                         math.sin(oa + math.radians(30 + 60 * k))]
                        for k in range(6)])
    
    all_verts = []
    for cx, cy, ang in inner_hexagons:
        verts = get_hex_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    all_verts_rel = all_verts - np.array([ox, oy])
    
    all_normals = np.vstack([normals, -normals])
    dots = all_verts_rel @ all_normals.T  # (n_verts, 12)
    
    # Soft max over vertices for each normal direction
    # soft_max_i = (1/beta) * log(sum_j exp(beta * dots[j,i]))
    dots_scaled = beta * dots
    max_dots = np.max(dots_scaled, axis=0)
    soft_max = (1.0 / beta) * (max_dots + np.log(np.sum(np.exp(dots_scaled - max_dots), axis=0)))
    
    # The side length constraint: S >= soft_max_i / (sqrt(3)/2) for all i
    # Use a smooth max over the 12 constraints
    sm_scaled = beta * soft_max
    sm_max = np.max(sm_scaled)
    soft_max_all = (1.0 / beta) * (sm_max + np.log(np.sum(np.exp(sm_scaled - sm_max))))
    
    return soft_max_all / (SQRT3 / 2.0)

def sat_overlap_check(v1, v2, tol=1e-7):
    """Check if two convex polygons overlap using SAT. Returns True if they overlap."""
    axes = []
    for v in (v1, v2):
        n = len(v)
        for i in range(n):
            edge = v[(i + 1) % n] - v[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                axes.append(normal / norm)
    
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - tol or max2 < min1 - tol:
            return False
    return True

def check_all_disjoint(inner_hexagons, tol=1e-7):
    """Check if all inner hexagons are pairwise disjoint."""
    n = len(inner_hexagons)
    verts = []
    for cx, cy, ang in inner_hexagons:
        verts.append(get_hex_vertices(cx, cy, 1.0, ang))
    
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap_check(verts[i], verts[j], tol):
                return False
    return True

def check_inside(inner_hexagons, outer_center, outer_side, outer_angle, tol=1e-7):
    """Check if all inner hexagons are inside the outer hexagon."""
    ox, oy = outer_center
    outer_verts = get_hex_vertices(ox, oy, outer_side, outer_angle)
    n_outer = len(outer_verts)
    
    for cx, cy, ang in inner_hexagons:
        verts = get_hex_vertices(cx, cy, 1.0, ang)
        for v in verts:
            px, py = v
            inside = True
            for i in range(n_outer):
                v1 = outer_verts[i]
                v2 = outer_verts[(i + 1) % n_outer]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                d1 = np.dot(normal, np.array([px, py]) - v1)
                d2 = np.dot(normal, outer_verts[(i + 2) % n_outer] - v1)
                if d1 * d2 < -tol:
                    inside = False
                    break
            if not inside:
                return False
    return True

def check_validity(inner_hexagons, outer_center, outer_side, outer_angle):
    """Check if the configuration is valid: no overlaps, all inside outer hexagon."""
    if not check_all_disjoint(inner_hexagons):
        return False
    if not check_inside(inner_hexagons, outer_center, outer_side, outer_angle):
        return False
    return True

def overlap_penalty(inner_hexagons, strength=500.0):
    """Compute a smooth penalty for overlapping inner hexagons using center distance."""
    penalty = 0.0
    n = len(inner_hexagons)
    for i in range(n):
        for j in range(i + 1, n):
            cx1, cy1, a1 = inner_hexagons[i]
            cx2, cy2, a2 = inner_hexagons[j]
            d = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
            # Minimum center distance for non-overlapping unit hexagons
            min_d = 2.0 * SQRT3 - 0.1
            if d < min_d:
                excess = min_d - d
                penalty += strength * excess ** 2
    return penalty

def objective(params, deadline, beta=50.0):
    """Objective function: smooth outer side length + penalty for overlaps."""
    if time.time() > deadline:
        return float('inf')
    
    n = N
    inner = []
    for i in range(n):
        cx = params[3 * i]
        cy = params[3 * i + 1]
        ang = math.degrees(params[3 * i + 2]) % 360
        inner.append((cx, cy, ang))
    ox = params[3 * n]
    oy = params[3 * n + 1]
    outer_angle = math.degrees(params[3 * n + 2]) % 360
    
    S_smooth = compute_required_side_smooth(inner, (ox, oy), outer_angle, beta)
    penalty = overlap_penalty(inner)
    
    return S_smooth + penalty

def _build_hexagonal_lattice():
    """Build hexagonal lattice positions for N hexagons."""
    positions = [(0.0, 0.0)]
    for k in range(6):
        angle = math.radians(60 * k)
        x = SQRT3 * math.cos(angle)
        y = SQRT3 * math.sin(angle)
        positions.append((x, y))
    for k in range(6):
        angle = math.radians(60 * k)
        x = 2 * SQRT3 * math.cos(angle)
        y = 2 * SQRT3 * math.sin(angle)
        positions.append((x, y))
        if len(positions) >= N:
            break
    return positions[:N]

def _build_initial_conditions():
    """Build diverse initial conditions for optimization."""
    conditions = []
    hex_positions = _build_hexagonal_lattice()
    
    # Condition 1: Standard hexagonal lattice
    ic = np.zeros(3 * N + 3)
    for i, (px, py) in enumerate(hex_positions):
        ic[3 * i] = px
        ic[3 * i + 1] = py
        ic[3 * i + 2] = 0.0
    ic[3 * N] = 0.0
    ic[3 * N + 1] = 0.0
    ic[3 * N + 2] = 0.0
    conditions.append(ic)
    
    # Condition 2: Perturbed lattice
    np.random.seed(123)
    ic2 = ic + np.random.randn(3 * N + 3) * 0.15
    conditions.append(ic2)
    
    # Condition 3: Different outer angle
    ic3 = ic.copy()
    ic3[3 * N + 2] = math.radians(30)
    conditions.append(ic3)
    
    # Condition 4: Rotated inner hexagons
    ic4 = ic.copy()
    for i in range(N):
        ic4[3 * i + 2] = math.radians(30)
    conditions.append(ic4)
    
    # Condition 5: Compressed lattice
    ic5 = ic.copy()
    for i in range(N):
        ic5[3 * i] *= 0.92
        ic5[3 * i + 1] *= 0.92
    conditions.append(ic5)
    
    # Condition 6: Random scatter
    np.random.seed(456)
    ic6 = np.random.randn(3 * N + 3) * 2.0
    ic6[3 * N + 2] = 0.0
    conditions.append(ic6)
    
    # Condition 7: Linear arrangement
    ic7 = np.zeros(3 * N + 3)
    for i in range(N):
        ic7[3 * i] = (i - N // 2) * 2 * SQRT3
        ic7[3 * i + 1] = 0.0
    conditions.append(ic7)
    
    return conditions

def construct():
    """Main entry point: find optimal packing of N unit hexagons in minimum outer hexagon."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_result = None
    best_score = float('inf')
    
    def try_result(params):
        nonlocal best_result, best_score
        inner = []
        for i in range(N):
            cx = params[3 * i]
            cy = params[3 * i + 1]
            ang = math.degrees(params[3 * i + 2]) % 360
            inner.append([cx, cy, ang])
        ox = params[3 * N]
        oy = params[3 * N + 1]
        outer_angle = math.degrees(params[3 * N + 2]) % 360
        
        S = compute_required_side(inner, (ox, oy), outer_angle)
        
        if check_validity(inner, (ox, oy), S, outer_angle):
            if S < best_score:
                best_score = S
                best_result = (inner, (ox, oy), S, outer_angle)
    
    # Phase 1: Multi-start L-BFGS-B with smooth objective
    initial_conditions = _build_initial_conditions()
    
    for ic in initial_conditions:
        if time.time() > deadline - 40:
            break
        
        try:
            res = minimize(objective, ic, args=(deadline, 50.0), method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.fun < float('inf'):
                try_result(res.x)
        except Exception:
            pass
    
    # Phase 2: Refinement with tighter tolerance
    if best_result is not None and time.time() < deadline - 30:
        try:
            params_best = np.zeros(3 * N + 3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3 * i] = cx
                params_best[3 * i + 1] = cy
                params_best[3 * i + 2] = math.radians(ang)
            params_best[3 * N] = best_result[1][0]
            params_best[3 * N + 1] = best_result[1][1]
            params_best[3 * N + 2] = math.radians(best_result[3])
            
            res = minimize(objective, params_best, args=(deadline, 100.0), method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < float('inf'):
                try_result(res.x)
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead refinement of outer parameters
    if best_result is not None and time.time() < deadline - 20:
        try:
            inner_fixed = best_result[0]
            
            def outer_obj(params_outer):
                ox, oy, oa = params_outer
                S = compute_required_side(inner_fixed, (ox, oy), oa)
                return S
            
            outer_ic = np.array([best_result[1][0], best_result[1][1], math.radians(best_result[3])])
            res_outer = minimize(outer_obj, outer_ic, method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
            
            if res_outer.fun < best_score:
                ox, oy, oa = res_outer.x
                outer_angle = math.degrees(oa) % 360
                S = res_outer.fun
                if check_validity(inner_fixed, (ox, oy), S, outer_angle):
                    best_result = (inner_fixed, (ox, oy), S, outer_angle)
                    best_score = S
        except Exception:
            pass
    
    # Phase 4: Local search - try small perturbations of inner hexagons
    if best_result is not None and time.time() < deadline - 15:
        try:
            inner_current = best_result[0]
            outer_current = best_result[1]
            outer_angle_current = best_result[3]
            
            # Try perturbing each inner hexagon slightly
            for i in range(N):
                if time.time() > deadline - 10:
                    break
                cx, cy, ang = inner_current[i]
                for dx in [-0.05, 0.0, 0.05]:
                    for dy in [-0.05, 0.0, 0.05]:
                        for dang in [-5.0, 0.0, 5.0]:
                            new_inner = [list(x) for x in inner_current]
                            new_inner[i] = [cx + dx, cy + dy, (ang + dang) % 360]
                            
                            S = compute_required_side(new_inner, outer_current, outer_angle_current)
                            if S < best_score and check_validity(new_inner, outer_current, S, outer_angle_current):
                                best_score = S
                                best_result = (new_inner, outer_current, S, outer_angle_current)
        except Exception:
            pass
    
    # Fallback: hexagonal lattice
    if best_result is None:
        hex_positions = _build_hexagonal_lattice()
        inner = [[px, py, 0.0] for px, py in hex_positions]
        S = compute_required_side(inner, (0.0, 0.0), 0.0)
        best_result = (inner, (0.0, 0.0), S, 0.0)
    
    return best_result

# EVOLVE-BLOCK-END
