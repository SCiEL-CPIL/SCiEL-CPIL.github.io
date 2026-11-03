"""Place 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
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

def _objective_de(params):
    """
    Objective for differential evolution: outer side length + collision penalty.
    params: flattened [x0,y0,a0, x1,y1,a1, ..., x11,y11,a11]
    """
    cx = params[0::3]
    cy = params[1::3]
    ang = params[2::3]
    
    # Compute outer hexagon side with center at origin and angle 0
    side = _outer_hex_side_from_params(cx, cy, ang, 0.0, 0.0, 0.0)
    
    # Collision penalty: for each pair, compute approximate gap
    penalty = 0.0
    eps = 0.1
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = math.hypot(dx, dy)
            if dist < 1e-12:
                penalty += 1e6
                continue
            nx, ny = dx / dist, dy / dist
            sup_i = _hex_support(cx[i], cy[i], ang[i], nx, ny)
            sup_j = _hex_support(cx[j], cy[j], ang[j], -nx, -ny)
            gap_i = sup_i - (nx * cx[i] + ny * cy[i])
            gap_j = sup_j - (-nx * cx[j] - ny * cy[j])
            sep = dist - gap_i - gap_j
            if sep < eps:
                penalty += (eps - sep) ** 2
    
    return side + 100.0 * penalty

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_side = float('inf')
    best_cfg = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Generate diverse initial configurations
    configs = []
    
    # Flower pattern: center + 6 around + 5 more
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
    
    # Hexagonal lattice
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
    
    # Ring pattern
    pts = [[0.0, 0.0, 0.0]]
    r = 2 * math.sqrt(3)
    for i in range(11):
        angle = math.radians(360 * i / 11 + 15)
        pts.append([r * math.cos(angle), r * math.sin(angle), 0.0])
    configs.append(pts)
    
    # Compact hex arrangement
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
    
    # Random configurations
    for _ in range(4):
        pts = []
        for _ in range(N):
            pts.append([random.uniform(-4, 4), random.uniform(-4, 4), random.uniform(0, 60)])
        configs.append(pts)
    
    # Check initial configs for validity
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
    
    # If no valid config found, use first one as starting point
    if best_cfg is None:
        best_cfg = configs[0]
        cx_arr = np.array([p[0] for p in best_cfg])
        cy_arr = np.array([p[1] for p in best_cfg])
        angle_arr = np.array([p[2] for p in best_cfg])
        best_side = _outer_hex_side_from_params(cx_arr, cy_arr, angle_arr, 0.0, 0.0, 0.0)
        best_center = (0.0, 0.0)
        best_angle = 0.0
    
    # Use differential evolution for global optimization
    if time.time() - t0 < TIME_LIMIT - 30:
        # Define bounds: x, y in [-5, 5], angle in [0, 60]
        bounds = [(-5, 5), (-5, 5), (0, 60)] * N
        
        # Starting points from our best config
        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3*i] = best_cfg[i][0]
            x0[3*i+1] = best_cfg[i][1]
            x0[3*i+2] = best_cfg[i][2]
        
        try:
            result = differential_evolution(
                _objective_de,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                polish=False,
                init='sobol',
                updating='deferred'
            )
            
            if result.fun < best_side:
                x_opt = result.x
                cx_opt = x_opt[0::3]
                cy_opt = x_opt[1::3]
                ang_opt = x_opt[2::3]
                
                # Check validity
                if _is_valid_fast(cx_opt, cy_opt, ang_opt):
                    # Optimize outer hexagon angle and center
                    best_local_side = float('inf')
                    for oangle in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150]:
                        for ocx in [-1.0, -0.5, 0, 0.5, 1.0]:
                            for ocy in [-1.0, -0.5, 0, 0.5, 1.0]:
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
    
    # Final validation
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
