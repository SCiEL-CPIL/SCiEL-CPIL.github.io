import numpy as np
from scipy.optimize import minimize
import time

N = 11

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return vertices of regular hexagon centered at (cx,cy) with given angle and side length."""
    angle = np.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle + np.radians(60 * k)
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def project_polygon(vertices, axis):
    """Project polygon vertices onto axis, return min and max."""
    dots = vertices @ axis
    return np.min(dots), np.max(dots)

def hexagons_disjoint(v1, v2):
    """
    Check if two hexagons (arrays of 6 vertices) are disjoint.
    Uses Separating Axis Theorem (SAT).
    """
    axes = []
    for v in (v1, v2):
        for i in range(6):
            p1 = v[i]
            p2 = v[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
    
    for axis in axes:
        min1, max1 = project_polygon(v1, axis)
        min2, max2 = project_polygon(v2, axis)
        if max1 < min2 - 1e-8 or max2 < min1 - 1e-8:
            return True
            
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    positions = []
    angles = [0.0] * N
    
    positions.append([0.0, 0.0])
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([2.0 * np.cos(a), 2.0 * np.sin(a)])
    for k in range(4):
        a = np.radians(60 * k)
        positions.append([4.0 * np.cos(a), 4.0 * np.sin(a)])
        
    while len(positions) < N:
        positions.append([0.0, 0.0])
    positions = positions[:N]

    x0 = []
    for px, py in positions:
        x0.append(px)
        x0.append(py)
    x0.extend(angles)
    x0.extend([0.0, 0.0, 5.0, 0.0])
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        
        pos = np.array(params[:2*N]).reshape(N, 2)
        ang = params[2*N:3*N]
        ocx, ocy = params[3*N], params[3*N+1]
        oside = params[3*N+2]
        oang = params[3*N+3]
        
        if oside < 1.0:
            return 1e10
            
        inner_verts = [hex_vertices(pos[i,0], pos[i,1], ang[i]) for i in range(N)]
        
        penalty = 0.0
        
        outer_inradius = oside * np.sqrt(3) / 2
        angle_rad = np.radians(oang)
        outer_axes = []
        for k in range(6):
            a = angle_rad + np.radians(60 * k)
            outer_axes.append(np.array([np.cos(a), np.sin(a)]))
            
        for i in range(N):
            for j in range(6):
                vx, vy = inner_verts[i][j]
                for axis in outer_axes:
                    dist = (vx - ocx) * axis[0] + (vy - ocy) * axis[1]
                    if dist > outer_inradius:
                        penalty += (dist - outer_inradius) ** 2
        
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint(inner_verts[i], inner_verts[j]):
                    dx = pos[i,0] - pos[j,0]
                    dy = pos[i,1] - pos[j,1]
                    d = np.sqrt(dx*dx + dy*dy)
                    min_d = 1.8 
                    if d < min_d:
                        penalty += (min_d - d) ** 2 * 10.0
                    else:
                        if d < 2.0:
                            penalty += (2.0 - d) ** 2 * 0.1
        
        return oside + 100.0 * penalty

    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
    
    xopt = result.x
    pos = np.array(xopt[:2*N]).reshape(N, 2)
    ang = xopt[2*N:3*N]
    ocx, ocy = xopt[3*N], xopt[3*N+1]
    oside = max(xopt[3*N+2], 1.0)
    oang = xopt[3*N+3]
    
    inner_verts = [hex_vertices(pos[i,0], pos[i,1], ang[i]) for i in range(N)]
    valid = True
    
    outer_inradius = oside * np.sqrt(3) / 2
    angle_rad = np.radians(oang)
    outer_axes = []
    for k in range(6):
        a = angle_rad + np.radians(60 * k)
        outer_axes.append(np.array([np.cos(a), np.sin(a)]))
        
    for i in range(N):
        for j in range(6):
            vx, vy = inner_verts[i][j]
            for axis in outer_axes:
                dist = (vx - ocx) * axis[0] + (vy - ocy) * axis[1]
                if dist > outer_inradius + 1e-6:
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if valid:
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint(inner_verts[i], inner_verts[j]):
                    valid = False
                    break
            if not valid: break
    
    if not valid:
        pos_fallback = []
        for i in range(3):
            for j in range(4):
                if len(pos_fallback) < N:
                    pos_fallback.append([i*2.1, j*2.1])
        while len(pos_fallback) < N:
            pos_fallback.append([0,0])
            
        inner = [[pos_fallback[i][0], pos_fallback[i][1], 0.0] for i in range(N)]
        xs = [p[0] for p in pos_fallback]
        ys = [p[1] for p in pos_fallback]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        width = max(xs) - min(xs) + 4.0
        height = max(ys) - min(ys) + 4.0
        side = max(width, height) / np.sqrt(3) + 1.0
        
        return inner, (cx, cy), side, 0.0

    inner = [[pos[i,0], pos[i,1], ang[i]] for i in range(N)]
    return inner, (ocx, ocy), oside, oang
