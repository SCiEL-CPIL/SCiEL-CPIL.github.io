"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115.0
    sqrt3 = np.sqrt(3.0)
    
    def hex_verts(cx, cy, ang_deg):
        a = np.radians(ang_deg)
        thetas = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + a
        return np.column_stack([cx + np.cos(thetas), cy + np.sin(thetas)])
    
    def pip(pt, poly):
        """Point in convex polygon via cross products."""
        n = len(poly)
        sign = None
        for i in range(n):
            j = (i + 1) % n
            cross = (poly[j][0] - poly[i][0]) * (pt[1] - poly[i][1]) - \
                    (poly[j][1] - poly[i][1]) * (pt[0] - poly[i][0])
            if abs(cross) > 1e-10:
                s = 1 if cross > 0 else -1
                if sign is None:
                    sign = s
                elif s != sign:
                    return False
        return True
    
    def overlap(i, j, centers, angles):
        vi = hex_verts(centers[i][0], centers[i][1], angles[i])
        vj = hex_verts(centers[j][0], centers[j][1], angles[j])
        for v in vi:
            if pip(v, vj):
                return True
        for v in vj:
            if pip(v, vi):
                return True
        # Edge intersection check
        for k in range(6):
            a1, b1 = vi[k], vi[(k+1)%6]
            for m in range(6):
                a2, b2 = vj[m], vj[(m+1)%6]
                if seg_int(a1, b1, a2, b2):
                    return True
        return False
    
    def seg_int(p1, p2, p3, p4):
        def ccw(a, b, c):
            return (c[1]-a[1])*(b[0]-a[0]) - (b[1]-a[1])*(c[0]-a[0])
        d1 = ccw(p3, p4, p1)
        d2 = ccw(p3, p4, p2)
        d3 = ccw(p1, p2, p3)
        d4 = ccw(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False
    
    def min_outer_side(centers, angles):
        """Minimize outer hexagon side length. Optimize over outer angle and center."""
        def cost(p):
            oa = p[0]  # outer angle
            os = p[1]  # outer side
            ox = p[2]
            oy = p[3]
            ov = hex_verts(ox, oy, oa) * os  # scale vertices
            ov = ov + np.array([ox, oy])  # wait, hex_verts already includes center
            # Actually hex_verts(cx, cy, ang) gives vertices at center cx,cy
            # So for outer hex with side os, we need to scale
            # Let me redo: outer hex vertices = center + os * unit_hex_verts(ang)
            unit_v = hex_verts(0, 0, oa)  # unit hex at origin
            ov = np.array([ox, oy]) + os * unit_v
            
            for i in range(N):
                iv = hex_verts(centers[i][0], centers[i][1], angles[i])
                for v in iv:
                    if not pip(v, ov):
                        return 1e6
            return os
        
        x0 = [0.0, 4.5, 0.0, 0.0]
        res = minimize(cost, x0, method='Nelder-Mead', 
                      options={'maxiter': 300, 'xatol': 1e-3, 'fatol': 1e-3})
        return res.fun, res.x
    
    # Honeycomb initial: center + 6 ring + 4 extensions
    pos = [
        (0, 0),
        (sqrt3, 0),
        (sqrt3/2, 1.5),
        (-sqrt3/2, 1.5),
        (-sqrt3, 0),
        (-sqrt3/2, -1.5),
        (sqrt3/2, -1.5),
        (2*sqrt3, 0),
        (sqrt3, 3),
        (-sqrt3, 3),
        (-2*sqrt3, 0),
    ]
    angs = [0.0] * N
    
    best_side = 1e10
    best_pos = pos
    best_angs = angs
    
    for trial in range(4):
        if time.time() > deadline - 10:
            break
        np.random.seed(trial * 999 + 42)
        p = list(pos)
        a = list(angs)
        for i in range(N):
            p[i] = (p[i][0] + np.random.normal(0, 0.3), p[i][1] + np.random.normal(0, 0.3))
            a[i] = np.random.uniform(-15, 15)
        
        def cost_full(params):
            if time.time() > deadline - 3:
                return 1e8
            c = [(params[2*i], params[2*i+1]) for i in range(N)]
            ag = [params[2*N + i] for i in range(N)]
            
            # Overlap penalty
            for i in range(N):
                for j in range(i+1, N):
                    if overlap(i, j, c, ag):
                        return 1e5 + np.sum((np.array(params) - np.array([0]*33))**2)
            
            s, _ = min_outer_side(c, ag)
            return s
        
        x0 = []
        for i in range(N):
            x0.extend([p[i][0], p[i][1]])
        for i in range(N):
            x0.append(a[i])
        
        res = minimize(cost_full, x0, method='Nelder-Mead',
                      options={'maxiter': 200, 'xatol': 1e-3, 'fatol': 1e-3, 'adaptive': True})
        
        if res.fun < best_side and res.fun < 100:
            best_side = res.fun
            best_pos = [(res.x[2*i], res.x[2*i+1]) for i in range(N)]
            best_angs = [res.x[2*N + i] for i in range(N)]
    
    # Final: get optimal outer hexagon
    if best_side < 100:
        os, op = min_outer_side(best_pos, best_angs)
        outer_angle = op[0]
        outer_center = (op[2], op[3])
        outer_side = op[1]
    else:
        outer_angle = 0.0
        outer_center = (0.0, 0.0)
        outer_side = 5.0
        best_pos = pos
        best_angs = angs
    
    inner = [[best_pos[i][0], best_pos[i][1], best_angs[i]] for i in range(N)]
    return inner, outer_center, float(outer_side), float(outer_angle)
# EVOLVE-BLOCK-END
