"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np
from scipy.optimize import differential_evolution

N = 12
TIME_LIMIT = 145.0  # Safety margin below 150s

def _hex_support(cx, cy, angle_deg, nx, ny, side=1.0):
    """Max of dot product of hex vertices with direction (nx,ny), offset by center."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    max_dot = 0.0
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        rx = ca * x - sa * y
        ry = sa * x + ca * y
        d = nx * rx + ny * ry
        if d > max_dot:
            max_dot = d
    return max_dot + nx * cx + ny * cy

def _hex_min_support(cx, cy, angle_deg, nx, ny, side=1.0):
    """Min of dot product of hex vertices with direction (nx,ny), offset by center."""
    return -_hex_support(cx, cy, angle_deg, -nx, -ny, side)

def _outer_hex_side_from_params(cx_arr, cy_arr, angle_arr, ocx, ocy, oangle_deg):
    """Compute minimum side length of outer hexagon enclosing all inner hexagons."""
    apothem_unit = math.sqrt(3) / 2.0
    oa = math.radians(oangle_deg)
    max_need = 0.0
    for k in range(6):
        na = oa + math.radians(90 + 60 * k)
        nx, ny = math.cos(na), math.sin(na)
        for i in range(N):
            s = _hex_support(cx_arr[i], cy_arr[i], angle_arr[i], nx, ny)
            val = (s - nx * ocx - ny * ocy) / apothem_unit
            if val > max_need:
                max_need = val
    return max_need

def _is_valid_fast(cx_arr, cy_arr, angle_arr):
    """Check that all pairs of hexagons are disjoint using SAT."""
    n = len(cx_arr)
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx_arr[i] - cx_arr[j]
            dy = cy_arr[i] - cy_arr[j]
            dist_sq = dx * dx + dy * dy
            if dist_sq > (2 * math.sqrt(3)) ** 2:
                continue
            if dist_sq < 4.0 - 1e-9:
                return False
            # SAT test: check all 12 edge normals (6 from each hex)
            for ai in range(6):
                aia = math.radians(angle_arr[i] + 90 + 60 * ai)
                anx, any_ = math.cos(aia), math.sin(aia)
                minA = _hex_min_support(cx_arr[i], cy_arr[i], angle_arr[i], anx, any_)
                maxA = _hex_support(cx_arr[i], cy_arr[i], angle_arr[i], anx, any_)
                minB = _hex_min_support(cx_arr[j], cy_arr[j], angle_arr[j], anx, any_)
                maxB = _hex_support(cx_arr[j], cy_arr[j], angle_arr[j], anx, any_)
                if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                    break
            else:
                for aj in range(6):
                    aja = math.radians(angle_arr[j] + 90 + 60 * aj)
                    anx, any_ = math.cos(aja), math.sin(aja)
                    minA = _hex_min_support(cx_arr[i], cy_arr[i], angle_arr[i], anx, any_)
                    maxA = _hex_support(cx_arr[i], cy_arr[i], angle_arr[i], anx, any_)
                    minB = _hex_min_support(cx_arr[j], cy_arr[j], angle_arr[j], anx, any_)
                    maxB = _hex_support(cx_arr[j], cy_arr[j], angle_arr[j], anx, any_)
                    if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                        break
                else:
                    return False
    return True

def _collision_penalty(cx_arr, cy_arr, angle_arr, eps=0.05):
    """Compute total collision penalty for all pairs using support functions."""
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx_arr[i] - cx_arr[j]
            dy = cy_arr[i] - cy_arr[j]
            dist = math.hypot(dx, dy)
            if dist < 1e-12:
                penalty += 1e4
                continue
            nx, ny = dx / dist, dy / dist
            sup_i = _hex_support(cx_arr[i], cy_arr[i], angle_arr[i], nx, ny)
            sup_j = _hex_support(cx_arr[j], cy_arr[j], angle_arr[j], -nx, -ny)
            gap_i = sup_i - (nx * cx_arr[i] + ny * cy_arr[i])
            gap_j = sup_j - (-nx * cx_arr[j] - ny * cy_arr[j])
            sep = dist - gap_i - gap_j
            if sep < eps:
                penalty += (eps - sep) ** 2
    return penalty

def _objective(params, ocx, ocy, oangle_deg, beta=50.0):
    """
    Objective: outer side length + beta * collision penalty.
    params: flattened [x0,y0,a0, x1,y1,a1, ..., x11,y11,a11]
    """
    cx = params[0::3]
    cy = params[1::3]
    ang = params[2::3]
    
    side = _outer_hex_side_from_params(cx, cy, ang, ocx, ocy, oangle_deg)
    penalty = _collision_penalty(cx, cy, ang, eps=0.05)
    
    return side + beta * penalty

def _make_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # Flower pattern
    pts = [[0.0, 0.0, 0.0]]
    r1 = 2 * math.sqrt(3)
    for i in range(6):
        angle = math.radians(60 * i)
        pts.append([r1 * math.cos(angle), r1 * math.sin(angle), 0.0])
    r2 = r1 * 1.732
    for deg in [30, 90, 150, 210, 270]:
        rad = math.radians(deg)
        pts.append([r2 * math.cos(rad), r2 * math.sin(rad), 0.0])
    configs.append(pts[:12])
    
    # Lattice
    pts = []
    rh = math.sqrt(3)
    cw = 2.0
    for c in range(3):
        pts.append([c * cw - cw, -rh, 0.0])
    for c in range(4):
        pts.append([c * cw - cw - cw / 2, 0.0, 0.0])
    for c in range(3):
        pts.append([c * cw - cw, rh, 0.0])
    pts.append([-cw * 2 - cw / 2 - cw, 0.0, 0.0])
    pts.append([cw * 2 - cw / 2 + cw, 0.0, 0.0])
    configs.append(pts)
    
    # Ring
    pts = [[0.0, 0.0, 0.0]]
    r = 2 * math.sqrt(3)
    for i in range(11):
        angle = math.radians(360 * i / 11 + 15)
        pts.append([r * math.cos(angle), r * math.sin(angle), 0.0])
    configs.append(pts)
    
    # Compact hex
    pts = [[0.0, 0.0, 0.0]]
    r = 2 * math.sqrt(3)
    for i in range(6):
        angle = math.radians(60 * i)
        pts.append([r * math.cos(angle), r * math.sin(angle), 0.0])
    r2 = r * 1.8
    for i in range(5):
        angle = math.radians(60 * i + 30)
        pts.append([r2 * math.cos(angle), r2 * math.sin(angle), 0.0])
    configs.append(pts)
    
    # Random
    for _ in range(3):
        pts = []
        for _ in range(N):
            pts.append([random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(0, 60)])
        configs.append(pts)
    
    return configs

def construct():
    t0 = time.time()
    random.seed(42)
    
    best_side = float('inf')
    best_cfg = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    configs = _make_initial_configs()
    
    # Phase 1: Evaluate initial configs
    for cfg in configs:
        if time.time() - t0 > TIME_LIMIT - 60:
            break
        if len(cfg) != N:
            continue
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        
        if _is_valid_fast(cx_arr, cy_arr, angle_arr):
            for oangle in [0, 30, 60, 90, 120, 150]:
                side = _outer_hex_side_from_params(cx_arr, cy_arr, angle_arr, 0.0, 0.0, float(oangle))
                if side < best_side:
                    best_side = side
                    best_cfg = [row[:] for row in cfg]
                    best_center = (0.0, 0.0)
                    best_angle = float(oangle)
    
    if best_cfg is None:
        best_cfg = configs[0]
        best_side = _outer_hex_side_from_params(
            np.array([p[0] for p in best_cfg]),
            np.array([p[1] for p in best_cfg]),
            np.array([p[2] for p in best_cfg]),
            0.0, 0.0, 0.0
        )
        best_center = (0.0, 0.0)
        best_angle = 0.0
    
    # Phase 2: Differential Evolution with collision penalty
    # Parameters: [x0,y0,a0, x1,y1,a1, ..., x11,y11,a11]
    # Bounds: positions in [-5,5], angles in [0,60]
    bounds = []
    for i in range(N):
        bounds.append((-5.0, 5.0))  # x
        bounds.append((-5.0, 5.0))  # y
        bounds.append((0.0, 60.0))  # angle
    
    def de_objective(params):
        if time.time() - t0 > TIME_LIMIT - 10:
            return 1e6
        return _objective(params, best_center[0], best_center[1], best_angle, beta=50.0)
    
    # Run DE from multiple starting points
    for start_idx, cfg in enumerate(configs[:4]):
        if time.time() - t0 > TIME_LIMIT - 30:
            break
        
        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3*i] = cfg[i][0]
            x0[3*i+1] = cfg[i][1]
            x0[3*i+2] = cfg[i][2]
        
        try:
            result = differential_evolution(
                de_objective,
                bounds,
                x0=x0,
                maxiter=30,
                popsize=10,
                tol=1e-4,
                seed=42 + start_idx,
                polish=False
            )
            
            if result.fun < best_side + 1.0:  # Allow some penalty
                x_opt = result.x
                cx_opt = x_opt[0::3]
                cy_opt = x_opt[1::3]
                ang_opt = x_opt[2::3]
                
                # Check validity
                if _is_valid_fast(cx_opt, cy_opt, ang_opt):
                    # Optimize outer hexagon
                    best_local_side = float('inf')
                    for oangle in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150]:
                        for ocx in [-0.5, 0, 0.5]:
                            for ocy in [-0.5, 0, 0.5]:
                                s = _outer_hex_side_from_params(cx_opt, cy_opt, ang_opt, ocx, ocy, float(oangle))
                                if s < best_local_side:
                                    best_local_side = s
                                    best_center = (ocx, ocy)
                                    best_angle = float(oangle)
                    
                    if best_local_side < best_side:
                        best_side = best_local_side
                        best_cfg = [[float(cx_opt[i]), float(cy_opt[i]), float(ang_opt[i])] for i in range(N)]
        except Exception:
            pass
    
    # Phase 3: Local refinement with Nelder-Mead
    if best_cfg is not None and time.time() - t0 < TIME_LIMIT - 15:
        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3*i] = best_cfg[i][0]
            x0[3*i+1] = best_cfg[i][1]
            x0[3*i+2] = best_cfg[i][2]
        
        from scipy.optimize import minimize as sp_minimize
        try:
            result = sp_minimize(
                lambda p: _objective(p, best_center[0], best_center[1], best_angle, beta=50.0),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            
            x_opt = result.x
            cx_opt = x_opt[0::3]
            cy_opt = x_opt[1::3]
            ang_opt = x_opt[2::3]
            
            if _is_valid_fast(cx_opt, cy_opt, ang_opt):
                best_local_side = float('inf')
                for oangle in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150]:
                    for ocx in [-0.5, 0, 0.5]:
                        for ocy in [-0.5, 0, 0.5]:
                            s = _outer_hex_side_from_params(cx_opt, cy_opt, ang_opt, ocx, ocy, float(oangle))
                            if s < best_local_side:
                                best_local_side = s
                                best_center = (ocx, ocy)
                                best_angle = float(oangle)
                
                if best_local_side < best_side:
                    best_side = best_local_side
                    best_cfg = [[float(cx_opt[i]), float(cy_opt[i]), float(ang_opt[i])] for i in range(N)]
        except Exception:
            pass
    
    # Final validation and cleanup
    if best_cfg is None:
        best_cfg = configs[0]
        best_side = _outer_hex_side_from_params(
            np.array([p[0] for p in best_cfg]),
            np.array([p[1] for p in best_cfg]),
            np.array([p[2] for p in best_cfg]),
            0.0, 0.0, 0.0
        )
        best_center = (0.0, 0.0)
        best_angle = 0.0
    
    return best_cfg, best_center, best_side, best_angle
