"""Improved program for packing 11 unit hexagons in a regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11
START_TIME = time.time()
TIME_LIMIT = 140.0

def _hex_in_hex_check(cx1, cy1, a1, cx2, cy2, a2, s1, s2):
    """Check if hexagon 1 is inside hexagon 2."""
    if s2 < 1e-10:
        return cx1 == cx2 and cy1 == cy2 and a1 == a2
    angle1 = np.radians(a1)
    angle2 = np.radians(a2)
    inradius2 = s2 * np.sqrt(3) / 2
    
    # Get vertices of hexagon 1
    for i in range(6):
        va = angle1 + i * np.pi / 3
        vx = cx1 + s1 * np.cos(va)
        vy = cy1 + s1 * np.sin(va)
        # Check if (vx, vy) is inside hexagon 2
        dx = vx - cx2
        dy = vy - cy2
        cos_a = np.cos(-angle2)
        sin_a = np.sin(-angle2)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        for j in range(6):
            na = np.radians(30 + j * 60)
            nx = np.cos(na)
            ny = np.sin(na)
            if lx * nx + ly * ny > inradius2 + 1e-9:
                return False
    return True

def _no_overlap_fast(pos, angles):
    """Fast check: center distance must be at least sqrt(3) for unit hexagons."""
    s3 = np.sqrt(3)
    n = len(pos)
    for i in range(n):
        for j in range(i+1, n):
            dx = pos[i][0] - pos[j][0]
            dy = pos[i][1] - pos[j][1]
            if dx*dx + dy*dy < s3*s3 - 1e-6:
                return False
    return True

def _min_side_for_hexagon(pos, angles, ocx, ocy, oang):
    """Find min side of outer hexagon to contain all inner hexagons."""
    lo, hi = 1.0, 50.0
    for _ in range(50):
        mid = (lo + hi) / 2
        ok = True
        for i in range(len(pos)):
            if not _hex_in_hex_check(pos[i][0], pos[i][1], angles[i], ocx, ocy, oang, 1.0, mid):
                ok = False
                break
        if ok:
            hi = mid
        else:
            lo = mid
    return hi

def construct():
    t0 = time.time()
    s3 = np.sqrt(3)
    
    # Initial symmetric arrangement
    init_pos = [(0.0, 0.0)]
    for k in range(6):
        a = k * np.pi / 3
        init_pos.append((s3 * np.cos(a), s3 * np.sin(a)))
    for k in range(4):
        a = k * np.pi / 2 + np.pi/4
        init_pos.append((2*s3 * np.cos(a), 2*s3 * np.sin(a)))
    
    init_angles = [0.0] * 11
    
    # Objective: minimize outer hexagon side
    def objective(params):
        if time.time() - t0 > TIME_LIMIT:
            return 1e10
        pos = [(params[2*i], params[2*i+1]) for i in range(11)]
        angles = list(params[22:33])
        ocx, ocy = params[33], params[34]
        oang = params[35]
        
        if not _no_overlap_fast(pos, angles):
            return 1e8
        
        return _min_side_for_hexagon(pos, angles, ocx, ocy, oang)
    
    x0 = []
    for p in init_pos:
        x0.extend([p[0], p[1]])
    x0.extend(init_angles)
    x0.extend([0.0, 0.0, 0.0])
    
    best_side = 50.0
    best_x = x0[:]
    
    for method in ['Nelder-Mead', 'Powell']:
        if time.time() - t0 > TIME_LIMIT - 15:
            break
        try:
            res = minimize(objective, x0, method=method,
                          options={'maxiter': 400, 'xatol': 0.005, 'fatol': 0.005})
            if res.fun < best_side:
                best_side = res.fun
                best_x = res.x
        except:
            pass
    
    # Try with fixed angles = 0
    if time.time() - t0 < TIME_LIMIT - 20:
        def objective_fixed_angles(params):
            if time.time() - t0 > TIME_LIMIT:
                return 1e10
            pos = [(params[2*i], params[2*i+1]) for i in range(11)]
            angles = [0.0] * 11
            ocx, ocy = params[22], params[23]
            oang = 0.0
            
            if not _no_overlap_fast(pos, angles):
                return 1e8
            
            return _min_side_for_hexagon(pos, angles, ocx, ocy, oang)
        
        x0_fa = []
        for p in init_pos:
            x0_fa.extend([p[0], p[1]])
        x0_fa.extend([0.0, 0.0])
        
        for method in ['Nelder-Mead', 'Powell']:
            if time.time() - t0 > TIME_LIMIT - 15:
                break
            try:
                res = minimize(objective_fixed_angles, x0_fa, method=method,
                              options={'maxiter': 300, 'xatol': 0.005, 'fatol': 0.005})
                if res.fun < best_side:
                    best_side = res.fun
                    # Store in compatible format
                    tmp = res.x[:]
                    tmp.extend([0.0] * 11)  # angles
                    tmp.extend([0.0])  # oang
                    best_x = tmp
            except:
                pass
    
    if best_side < 49.0:
        pos = [(best_x[2*i], best_x[2*i+1]) for i in range(11)]
        angles = list(best_x[22:33])
        ocx, ocy = best_x[33], best_x[34]
        oang = best_x[35]
        inner = [[p[0], p[1], a] for p, a in zip(pos, angles)]
        return inner, (ocx, ocy), best_side, oang
    else:
        # Fallback
        inner = [[2.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 10.0, 0.0

