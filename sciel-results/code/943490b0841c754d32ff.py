"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
_DEADLINE = 145.0

def _hex_inradius(s):
    return s * math.cos(math.radians(30))

def _point_in_hex(px, py, cx, cy, s, rot):
    """
    Check if point (px, py) is inside a hexagon centered at (cx, cy) with side s and rotation rot.
    Returns True if inside.
    """
    dx = px - cx
    dy = py - cy
    ca, sa = math.cos(-rot), math.sin(-rot)
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    r_in = _hex_inradius(s)
    c30 = math.cos(math.radians(30))
    d1 = abs(c30 * lx + 0.5 * ly)
    d2 = abs(ly)
    d3 = abs(-c30 * lx + 0.5 * ly)
    return max(d1, d2, d3) <= r_in + 1e-9

def _hex_verts(cx, cy, s, rot):
    vs = np.empty((6, 2))
    for k in range(6):
        a = math.radians(60.0 * k) + rot
        vs[k, 0] = cx + s * math.cos(a)
        vs[k, 1] = cy + s * math.sin(a)
    return vs

def _is_valid_config(inner, center, side, angle):
    """
    Strictly validate the configuration.
    inner: list of (x, y, angle_deg)
    center: (cx, cy)
    side: outer side length
    angle: outer rotation (radians)
    """
    cx, cy = center
    s_out = side
    r_out = _hex_inradius(s_out)
    
    # Check boundary
    for i in range(len(inner)):
        xi, yi, ai_deg = inner[i]
        ai = math.radians(ai_deg)
        verts = _hex_verts(xi, yi, 1.0, ai)
        for v in verts:
            if not _point_in_hex(v[0], v[1], cx, cy, s_out, angle):
                return False

    # Check overlap
    for i in range(len(inner)):
        xi, yi, ai_deg = inner[i]
        ai = math.radians(ai_deg)
        vi = _hex_verts(xi, yi, 1.0, ai)
        for j in range(i + 1, len(inner)):
            xj, yj, aj_deg = inner[j]
            aj = math.radians(aj_deg)
            vj = _hex_verts(xj, yj, 1.0, aj)
            
            # Quick reject
            dc = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dc > 2.0:
                continue
            
            # Check if any vertex of i is in j
            overlap = False
            for v in vi:
                if _point_in_hex(v[0], v[1], xj, yj, 1.0, aj):
                    overlap = True
                    break
            if overlap:
                return False
            
            # Check if any vertex of j is in i
            for v in vj:
                if _point_in_hex(v[0], v[1], xi, yi, 1.0, ai):
                    overlap = True
                    break
            if overlap:
                return False
                
    return True

def _honeycomb_positions():
    """Generate honeycomb packing positions for 11 hexagons."""
    positions = []
    # Center
    positions.append((0.0, 0.0, 0.0))
    # Ring 1: 6 hexagons at distance 2
    for k in range(6):
        a = math.radians(60.0 * k + 30.0)
        positions.append((2.0*math.cos(a), 2.0*math.sin(a), 0.0))
    # Ring 2: 4 hexagons at distance 4, alternating angles
    for k in range(4):
        a = math.radians(60.0 * k + 30.0)
        positions.append((4.0*math.cos(a), 4.0*math.sin(a), 0.0))
    return positions[:N]

def construct():
    t_start = time.time()
    n = N

    def _remaining():
        return _DEADLINE - (time.time() - t_start)

    # Generate initial valid configuration
    init_pos = _honeycomb_positions()
    
    # Calculate required outer side for initial config
    max_r = 0.0
    for (xi, yi, ai) in init_pos:
        verts = _hex_verts(xi, yi, 1.0, ai)
        for v in verts:
            r = math.sqrt(v[0]**2 + v[1]**2)
            if r > max_r:
                max_r = r
    
    s_init = max_r / math.cos(math.radians(30))
    s_init *= 1.01
    
    # Prepare initial parameters for optimization
    # p = [x1, y1, a1, x2, y2, a2, ..., xn, yn, an, S]
    p0 = np.zeros(3*n + 1)
    for i in range(n):
        p0[3*i] = init_pos[i][0]
        p0[3*i+1] = init_pos[i][1]
        p0[3*i+2] = 0.0
    p0[3*n] = s_init
    
    def objective(p):
        x = p[0:3*n:3]
        y = p[1:3*n:3]
        angles = p[2:3*n:3]
        S = p[3*n]
        
        if S < 1.0:
            return 1e6
            
        pen = 0.0
        r_out = _hex_inradius(S)
        
        # Boundary penalty
        for i in range(n):
            verts = _hex_verts(x[i], y[i], 1.0, angles[i])
            for v in verts:
                dx = v[0]
                dy = v[1]
                c30 = math.cos(math.radians(30))
                d1 = abs(c30 * dx + 0.5 * dy)
                d2 = abs(dy)
                d3 = abs(-c30 * dx + 0.5 * dy)
                d_max = max(d1, d2, d3)
                if d_max > r_out:
                    pen += (d_max - r_out)**2
                    
        # Overlap penalty
        for i in range(n):
            for j in range(i+1, n):
                dc = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dc > 2.0:
                    continue
                vi = _hex_verts(x[i], y[i], 1.0, angles[i])
                vj = _hex_verts(x[j], y[j], 1.0, angles[j])
                
                overlap = False
                for v in vi:
                    if _point_in_hex(v[0], v[1], x[j], y[j], 1.0, angles[j]):
                        overlap = True
                        break
                if overlap:
                    pen += 10.0
                    break
                
                if not overlap:
                    for v in vj:
                        if _point_in_hex(v[0], v[1], x[i], y[i], 1.0, angles[i]):
                            overlap = True
                            break
                if overlap:
                    pen += 10.0
                    break
        
        return 0.5 * S * S + 10.0 * pen

    best_val = float('inf')
    best_p = p0.copy()
    
    rng = np.random.RandomState(42)
    
    # Multi-start optimization with diverse initial configurations
    for trial in range(10):
        if _remaining() < 15.0:
            break
            
        if trial == 0:
            p_init = p0.copy()
        elif trial == 1:
            # Random scatter
            p_init = np.zeros(3*n + 1)
            for i in range(n):
                p_init[3*i] = rng.uniform(-2, 2)
                p_init[3*i+1] = rng.uniform(-2, 2)
                p_init[3*i+2] = rng.uniform(-math.pi/6, math.pi/6)
            p_init[3*n] = s_init * 1.05
        else:
            # Perturb best
            p_init = best_p.copy()
            p_init[0:3*n] += rng.randn(3*n) * 0.15
            p_init[3*n] = best_p[3*n] * (1.0 + rng.rand()*0.05)
            
        try:
            res = minimize(objective, p_init, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_val:
                best_val = res.fun
                best_p = res.x.copy()
        except Exception:
            pass

    # Extract best solution
    x = best_p[0:3*n:3]
    y = best_p[1:3*n:3]
    angles = best_p[2:3*n:3]
    S = best_p[3*n]
    
    inner = [[float(x[i]), float(y[i]), float(math.degrees(angles[i]))] for i in range(n)]
    center = (0.0, 0.0)
    angle_out = 0.0
    
    if _is_valid_config(inner, center, S, angle_out):
        return inner, center, float(S), angle_out
    else:
        inner_init = [[float(p[0]), float(p[1]), float(math.degrees(p[2]))] for p in init_pos]
        return inner_init, center, float(s_init), angle_out
