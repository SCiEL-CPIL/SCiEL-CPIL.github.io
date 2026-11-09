"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 11
_DEADLINE = 145.0
_C30 = math.cos(math.radians(30))

def _hex_inradius(s):
    return s * _C30

def _hex_verts_arr(cx, cy, s, rot):
    """Vectorized hexagon vertices."""
    k = np.arange(6)
    a = math.radians(60.0) * k + rot
    return np.column_stack([cx + s * np.cos(a), cy + s * np.sin(a)])

def _point_in_hex_vec(px, py, cx, cy, s, rot):
    """Vectorized point-in-hex test. px, py can be arrays."""
    dx = px - cx
    dy = py - cy
    ca = math.cos(-rot)
    sa = math.sin(-rot)
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    r_in = _hex_inradius(s)
    d1 = np.abs(_C30 * lx + 0.5 * ly)
    d2 = np.abs(ly)
    d3 = np.abs(-_C30 * lx + 0.5 * ly)
    return np.max(np.maximum(np.maximum(d1, d2), d3), axis=-1) <= r_in + 1e-9

def _hex_distance(px, py, cx, cy, s, rot):
    """Distance from point to hexagon (0 if inside). Vectorized."""
    dx = px - cx
    dy = py - cy
    ca = math.cos(-rot)
    sa = math.sin(-rot)
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    r_in = _hex_inradius(s)
    d1 = np.abs(_C30 * lx + 0.5 * ly)
    d2 = np.abs(ly)
    d3 = np.abs(-_C30 * lx + 0.5 * ly)
    d_max = np.maximum(np.maximum(d1, d2), d3)
    return np.maximum(0.0, d_max - r_in)

def _is_valid_config(inner, center, side, angle):
    """Strictly validate the configuration."""
    cx, cy = center
    s_out = side
    r_out = _hex_inradius(s_out)
    
    for i in range(len(inner)):
        xi, yi, ai_deg = inner[i]
        ai = math.radians(ai_deg)
        verts = _hex_verts_arr(xi, yi, 1.0, ai)
        if not np.all(_point_in_hex_vec(verts[:, 0], verts[:, 1], cx, cy, s_out, angle)):
            return False

    for i in range(len(inner)):
        xi, yi, ai_deg = inner[i]
        ai = math.radians(ai_deg)
        vi = _hex_verts_arr(xi, yi, 1.0, ai)
        for j in range(i + 1, len(inner)):
            xj, yj, aj_deg = inner[j]
            aj = math.radians(aj_deg)
            dc = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dc > 2.05:
                continue
            vj = _hex_verts_arr(xj, yj, 1.0, aj)
            if np.any(_point_in_hex_vec(vi[:, 0], vi[:, 1], xj, yj, 1.0, aj)):
                return False
            if np.any(_point_in_hex_vec(vj[:, 0], vj[:, 1], xi, yi, 1.0, ai)):
                return False
                
    return True

def _honeycomb_positions():
    """Generate honeycomb packing positions for 11 hexagons."""
    positions = []
    positions.append((0.0, 0.0, 0.0))
    for k in range(6):
        a = math.radians(60.0 * k + 30.0)
        positions.append((2.0*math.cos(a), 2.0*math.sin(a), 0.0))
    for k in range(4):
        a = math.radians(60.0 * k + 30.0)
        positions.append((4.0*math.cos(a), 4.0*math.sin(a), 0.0))
    return positions[:N]

def _objective(p):
    """
    Objective function for optimization.
    p: [x1, y1, a1, x2, y2, a2, ..., xn, yn, an, S]
    """
    n = N
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
        verts = _hex_verts_arr(x[i], y[i], 1.0, angles[i])
        dists = _hex_distance(verts[:, 0], verts[:, 1], 0.0, 0.0, S, 0.0)
        pen += np.sum(dists ** 2)
                
    # Overlap penalty
    for i in range(n):
        for j in range(i+1, n):
            dc = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if dc > 2.05:
                continue
            vi = _hex_verts_arr(x[i], y[i], 1.0, angles[i])
            vj = _hex_verts_arr(x[j], y[j], 1.0, angles[j])
            
            d1 = _hex_distance(vi[:, 0], vi[:, 1], x[j], y[j], 1.0, angles[j])
            if np.any(d1 > 1e-6):
                pen += 10.0 * np.sum(d1 ** 2)
                continue
            d2 = _hex_distance(vj[:, 0], vj[:, 1], x[i], y[i], 1.0, angles[i])
            if np.any(d2 > 1e-6):
                pen += 10.0 * np.sum(d2 ** 2)
                
    return 0.5 * S * S + 5.0 * pen

def _polish(p, t_remaining):
    """Polish a solution with Nelder-Mead and Powell."""
    best_p = p.copy()
    best_val = _objective(p)
    
    if t_remaining > 20.0:
        try:
            res = minimize(_objective, best_p, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-7})
            if res.fun < best_val:
                best_val = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
    
    if t_remaining > 15.0:
        try:
            res = minimize(_objective, best_p, method='Powell',
                          options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
    
    return best_p, best_val

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
        verts = _hex_verts_arr(xi, yi, 1.0, ai)
        r = np.sqrt(verts[:, 0]**2 + verts[:, 1]**2)
        max_r = max(max_r, float(np.max(r)))
    
    s_init = max_r / _C30 * 1.01
    
    # Prepare initial parameters for optimization
    p0 = np.zeros(3*n + 1)
    for i in range(n):
        p0[3*i] = init_pos[i][0]
        p0[3*i+1] = init_pos[i][1]
        p0[3*i+2] = 0.0
    p0[3*n] = s_init
    
    best_val = _objective(p0)
    best_p = p0.copy()
    
    # Use differential evolution for global optimization
    if _remaining() > 60.0:
        bounds = []
        for i in range(n):
            bounds.append((-5.0, 5.0))
            bounds.append((-5.0, 5.0))
            bounds.append((0.0, 2*math.pi))
        bounds.append((1.5, 6.0))
        
        try:
            res = differential_evolution(_objective, bounds, 
                                         maxiter=100, 
                                         tol=1e-3,
                                         seed=42,
                                         polish=False,
                                         init='sobol',
                                         popsize=15)
            if res.fun < best_val:
                best_val = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
    
    # Polish with multiple methods
    if _remaining() > 25.0:
        best_p, best_val = _polish(best_p, _remaining())
    
    # Try perturbations around best
    if _remaining() > 15.0:
        rng = np.random.RandomState(123)
        for trial in range(5):
            if _remaining() < 5.0:
                break
            p_try = best_p + rng.normal(0, 0.1, size=best_p.shape)
            p_try[3*n] = max(1.5, p_try[3*n])
            p_try[2:3*n:3] = np.mod(p_try[2:3*n:3], 2*math.pi)
            p_try[:3*n:3] = np.clip(p_try[:3*n:3], -5.0, 5.0)
            p_try[1:3*n:3] = np.clip(p_try[1:3*n:3], -5.0, 5.0)
            val_try = _objective(p_try)
            if val_try < best_val:
                best_p, best_val = _polish(p_try, _remaining())
    
    # Extract best solution
    x = best_p[0:3*n:3]
    y = best_p[1:3*n:3]
    angles = best_p[2:3*n:3]
    S = best_p[3*n]
    
    inner = [[float(x[i]), float(y[i]), float(math.degrees(angles[i]))] for i in range(n)]
    center = (0.0, 0.0)
    angle_out = 0.0
    
    # Validate with margin
    if _is_valid_config(inner, center, S, angle_out):
        return inner, center, float(S), angle_out
    else:
        # Try to repair: slightly increase S
        for margin in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
            S_try = S + margin
            if _is_valid_config(inner, center, S_try, angle_out):
                return inner, center, float(S_try), angle_out
        
        # Fallback to initial valid configuration
        inner_init = [[float(p[0]), float(p[1]), float(math.degrees(p[2]))] for p in init_pos]
        return inner_init, center, float(s_init), angle_out
